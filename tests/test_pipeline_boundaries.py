"""Regression tests for train/holdout boundaries in pipeline helpers."""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cloud_device_recsys.run_pipeline as run_pipeline


class PipelineBoundaryTests(unittest.TestCase):
    def test_train_negative_pool_ignores_legacy_holdout_pool_options(self):
        logger = logging.getLogger("test.pipeline_boundaries")
        with mock.patch.object(
            run_pipeline,
            "ensure_train_item_pool",
            return_value="/tmp/cand_items_train.parquet",
        ) as ensure_pool, self.assertLogs(logger, level="WARNING") as captured:
            actual = run_pipeline._ensure_train_negative_pool(
                stage_name="Preranking",
                stage_config={"neg_sampling_pool": "full"},
                paths={"train_path": "/tmp/train.parquet"},
                dataset_config={},
                feature_group_manager=object(),
                feature_map=object(),
                logger=logger,
            )

        self.assertEqual(actual, "/tmp/cand_items_train.parquet")
        self.assertIn("must use the train-only item pool", captured.output[0])
        self.assertEqual(ensure_pool.call_args.kwargs["data_paths"]["train_path"], "/tmp/train.parquet")

    def test_runtime_vocab_pruning_scans_only_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train.parquet"
            valid = root / "valid.parquet"
            test = root / "test.parquet"
            for path in (train, valid, test):
                path.touch()

            self.assertEqual(run_pipeline._get_train_vocab_scan_paths(str(root)), [str(train)])

            train_positive = root / "train_positive.parquet"
            train_positive.touch()
            self.assertEqual(
                run_pipeline._get_train_vocab_scan_paths(str(root)),
                [str(train_positive)],
            )


if __name__ == "__main__":
    unittest.main()
