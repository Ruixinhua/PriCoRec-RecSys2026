import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import polars as pl
import yaml

import cloud_device_recsys.run_preprocess as run_preprocess
import cloud_device_recsys.run_pipeline as run_pipeline
from cloud_device_recsys.config.feature_groups import FeatureGroup, FeatureGroupManager
from cloud_device_recsys.data.item_pool import (
    ensure_train_item_pool,
    extract_item_corpus_from_tfrecord,
)
from fuxictr.pytorch.dataloaders.parquet_dataloader import ParquetDataset


build_dataset = importlib.import_module("fuxictr.preprocess.build_dataset")
item_pool_module = importlib.import_module("cloud_device_recsys.data.item_pool")


class _RecordingFeatureProcessor:
    instances = []

    def __init__(self, feature_cols, label_col, dataset_id, data_root, **kwargs):
        self.feature_cols = feature_cols
        self.label_cols = label_col
        self.data_dir = str(Path(data_root) / dataset_id)
        self.fitted_frame = None
        self.__class__.instances.append(self)

    def read_data(self, data_path, data_format="csv", n_rows=None):
        return pl.scan_csv(data_path, n_rows=n_rows)

    def preprocess(self, ddf):
        # Deliberately mimic FeatureProcessor's projection of non-model fields.
        return ddf.select(["label", "feature"])

    def fit(self, ddf, **kwargs):
        self.fitted_frame = ddf.collect()


class DataPipelineIntegrityTests(unittest.TestCase):
    def test_preprocessor_fits_train_only_and_persists_impression_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            raw_dir = tmp_path / "raw"
            config_dir = tmp_path / "config"
            output_dir = tmp_path / "processed"
            raw_dir.mkdir()
            config_dir.mkdir()

            for split, frame in {
                "train": pd.DataFrame({"label": [1, 0], "feature": [10, 11], "impression_id": [100, 100]}),
                "valid": pd.DataFrame({"label": [1], "feature": [99], "impression_id": [200]}),
                "test": pd.DataFrame({"label": [1], "feature": [77], "impression_id": [300]}),
            }.items():
                frame.to_csv(raw_dir / f"{split}.csv", index=False)

            config = {
                "Toy": {
                    "raw_data_format": "csv",
                    "processed_data_root": str(output_dir),
                    "label_col": {"name": "label", "dtype": "float"},
                    "impression_id_col": "impression_id",
                    "item_id_col": "feature",
                    "preprocessing": {
                        "positive_only_eval": False,
                        "generate_impression_id": True,
                    },
                    "feature_cols": [
                        {"name": "feature", "type": "categorical", "dtype": "int", "feature_group": "FG1"}
                    ],
                }
            }
            with open(config_dir / "dataset_config.yaml", "w", encoding="utf-8") as file_handle:
                yaml.safe_dump(config, file_handle)

            transformed = {}

            def capture_transform(feature_processor, ddf, filename, **kwargs):
                transformed[filename] = ddf.collect()

            _RecordingFeatureProcessor.instances.clear()
            with patch.object(run_preprocess, "FeatureProcessor", _RecordingFeatureProcessor), patch.object(
                run_preprocess, "transform", capture_transform
            ):
                preprocessor = run_preprocess.DataPreprocessor(
                    raw_data_root=str(raw_dir),
                    dataset_id="Toy",
                    config_dir=str(config_dir),
                    output_dir=str(output_dir),
                )
                preprocessor.run()

            fitted = _RecordingFeatureProcessor.instances[0].fitted_frame
            self.assertEqual(fitted.columns, ["label", "feature"])
            self.assertEqual(fitted["feature"].to_list(), [10, 11])
            self.assertNotIn("impression_id", [feature["name"] for feature in preprocessor.feature_cols])
            self.assertEqual(transformed["train"]["impression_id"].to_list(), [100, 100])
            self.assertEqual(transformed["valid"]["impression_id"].to_list(), [200])
            self.assertEqual(transformed["test"]["impression_id"].to_list(), [300])

    def test_eval_item_filter_is_derived_from_train_not_holdout_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            config_dir = tmp_path / "config"
            config_dir.mkdir()
            config = {
                "Toy": {
                    "processed_data_root": str(tmp_path / "processed"),
                    "label_col": {"name": "label", "dtype": "float"},
                    "impression_id_col": "impression_id",
                    "item_id_col": "item",
                    "preprocessing": {
                        "positive_only_eval": False,
                        "generate_impression_id": False,
                        "eval_item_freq_filter": {"enabled": True, "min_positive_count": 2},
                    },
                    "feature_cols": [
                        {"name": "item", "type": "categorical", "dtype": "int", "feature_group": "FG1"}
                    ],
                }
            }
            with open(config_dir / "dataset_config.yaml", "w", encoding="utf-8") as file_handle:
                yaml.safe_dump(config, file_handle)

            preprocessor = run_preprocess.DataPreprocessor(
                raw_data_root=str(tmp_path / "raw"),
                dataset_id="Toy",
                config_dir=str(config_dir),
                output_dir=str(tmp_path / "processed"),
            )
            train_ddf = pl.DataFrame({"label": [1, 1], "item": [1, 1], "impression_id": [1, 2]}).lazy()
            valid_ddf = pl.DataFrame({"label": [1, 1], "item": [99, 99], "impression_id": [3, 4]}).lazy()

            preprocessor._maybe_initialize_eval_filter(train_ddf)
            self.assertTrue(preprocessor.preprocess_split(valid_ddf, "valid").collect().is_empty())

    def test_parquet_loader_requires_persisted_impression_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            feature_map = SimpleNamespace(features={"feature": {}}, labels=["label", "impression_id"])
            missing_id_path = tmp_path / "missing_id.parquet"
            pd.DataFrame({"feature": [1], "label": [1.0]}).to_parquet(missing_id_path, index=False)

            with self.assertRaisesRegex(KeyError, "row indices are not valid request identifiers"):
                ParquetDataset(feature_map, str(missing_id_path))
            with self.assertRaisesRegex(KeyError, "row indices are not valid request identifiers"):
                ParquetDataset(feature_map, str(missing_id_path), low_memory=True)

            present_id_path = tmp_path / "with_id.parquet"
            pd.DataFrame({"feature": [1, 2], "label": [1.0, 0.0], "impression_id": [44, 44]}).to_parquet(
                present_id_path,
                index=False,
            )
            dataset = ParquetDataset(feature_map, str(present_id_path))
            self.assertEqual(dataset.darray[:, 2].tolist(), [44.0, 44.0])

    def test_tfrecord_sequence_serialization_pads_and_truncates(self):
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("tensorflow is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rows.tfrecord"
            feature_encoder = SimpleNamespace(
                feature_cols=[
                    {"name": "seq", "type": "sequence", "max_len": 4, "padding": "pre"},
                    {"name": "cat", "type": "categorical"},
                ],
                label_cols=[{"name": "label"}],
            )
            source = pd.DataFrame(
                {
                    "seq": [[1, 2], [3, 4, 5, 6, 7]],
                    "cat": [9, 10],
                    "label": [1.0, 0.0],
                }
            )

            build_dataset.convert_to_tfrecord(feature_encoder, source, str(output_path))

            records = []
            dataset = tf.data.TFRecordDataset(
                [str(output_path)],
                compression_type="GZIP",
                num_parallel_reads=1,
            )
            for serialized in dataset.as_numpy_iterator():
                example = tf.train.Example()
                example.ParseFromString(serialized)
                records.append(example)

            self.assertEqual(list(records[0].features.feature["seq"].int64_list.value), [0, 0, 1, 2])
            self.assertEqual(list(records[1].features.feature["seq"].int64_list.value), [4, 5, 6, 7])
            self.assertEqual(list(records[0].features.feature["cat"].int64_list.value), [9])
            self.assertEqual(list(records[0].features.feature["label"].float_list.value), [1.0])

    def test_transform_propagates_worker_failures_before_merge(self):
        class FailedResult:
            def get(self):
                raise RuntimeError("worker failed")

        class FakePool:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                self.terminated = False
                self.joined = 0
                self.__class__.instances.append(self)

            def apply_async(self, *_args, **_kwargs):
                return FailedResult()

            def close(self):
                self.closed = True

            def terminate(self):
                self.terminated = True

            def join(self):
                self.joined += 1

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(build_dataset.mp, "Pool", FakePool), patch.object(
            build_dataset.mp, "cpu_count", lambda: 2
        ):
            encoder = SimpleNamespace(data_dir=temp_dir)
            ddf = pl.DataFrame({"value": [1, 2]}).lazy()
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                build_dataset.transform(encoder, ddf, "train", block_size=1)

        pool = FakePool.instances[0]
        self.assertTrue(pool.closed)
        self.assertTrue(pool.terminated)
        self.assertEqual(pool.joined, 1)

    def test_merge_refuses_incomplete_part_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            part_dir = tmp_path / "train"
            part_dir.mkdir()
            pd.DataFrame({"value": [1]}).to_parquet(part_dir / "part_00000.parquet", index=False)

            with self.assertRaisesRegex(RuntimeError, "Expected 2 parquet part files"):
                build_dataset.merge_part_files(str(tmp_path), "train", num_parts=2)
            with self.assertRaisesRegex(RuntimeError, "Expected 1 parquet part files"):
                build_dataset.merge_part_files(str(tmp_path), "missing", num_parts=1)

    def test_train_item_pool_is_train_only_and_invalidates_changed_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            train_path = tmp_path / "train.parquet"
            valid_path = tmp_path / "valid.parquet"
            test_path = tmp_path / "test.parquet"
            pd.DataFrame({"item": [1, 2, 2], "item_feature": [10, 20, 20]}).to_parquet(train_path, index=False)
            pd.DataFrame({"item": [90], "item_feature": [900]}).to_parquet(valid_path, index=False)
            pd.DataFrame({"item": [91], "item_feature": [910]}).to_parquet(test_path, index=False)

            feature_group_manager = FeatureGroupManager()
            feature_group_manager.assign_feature("item_feature", FeatureGroup.FG1)
            data_paths = {
                "train_path": str(train_path),
                "valid_path": str(valid_path),
                "test_path": str(test_path),
            }
            dataset_config = {
                "item_id_col": "item",
                "processed_data_format": "parquet",
                "item_pool": {"train_file": "train_negatives"},
            }

            pool_path = ensure_train_item_pool(data_paths, dataset_config, feature_group_manager)
            self.assertEqual(Path(pool_path).name, "train_negatives.parquet")
            self.assertEqual(set(pd.read_parquet(pool_path)["item"].tolist()), {1, 2})
            self.assertTrue(Path(f"{pool_path}.metadata.json").exists())

            with patch.object(item_pool_module, "extract_item_corpus") as extract_item_corpus:
                self.assertEqual(
                    ensure_train_item_pool(data_paths, dataset_config, feature_group_manager),
                    pool_path,
                )
                extract_item_corpus.assert_not_called()

            pd.DataFrame({"item": [1, 2, 3], "item_feature": [10, 20, 30]}).to_parquet(train_path, index=False)
            refreshed_pool_path = ensure_train_item_pool(data_paths, dataset_config, feature_group_manager)
            self.assertEqual(refreshed_pool_path, pool_path)
            self.assertEqual(set(pd.read_parquet(pool_path)["item"].tolist()), {1, 2, 3})

    def test_tfrecord_item_pool_rejects_missing_configured_features(self):
        feature_map = SimpleNamespace(features={"item": {}}, labels=[])
        with self.assertRaisesRegex(ValueError, "Configured item pool fields are missing"):
            extract_item_corpus_from_tfrecord(
                input_paths=["unused.tfrecord"],
                output_path="unused.parquet",
                item_id_col="item",
                item_feature_cols=["item_feature"],
                feature_map=feature_map,
            )

    def test_pipeline_loads_eval_pool_before_train_negative_pool(self):
        for stage_runner in (
            run_pipeline.run_preranking_stage,
            run_pipeline.run_reranking_stage,
        ):
            with self.subTest(stage_runner=stage_runner.__name__):
                source = inspect.getsource(stage_runner)
                self.assertIn("_ensure_train_negative_pool", source)
                self.assertIn("load_negative_item_features", source)
                self.assertNotIn("load_item_features(neg_pool_path)", source)
                self.assertLess(
                    source.index("load_item_features("),
                    source.index("load_negative_item_features("),
                )


if __name__ == "__main__":
    unittest.main()
