"""Regression tests for offline vocabulary remap artifact boundaries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_device_recsys.data import remap_vocab_data  # noqa: E402
from cloud_device_recsys.data.vocab_pruner import VocabPruneInfo  # noqa: E402
from cloud_device_recsys.utils import load_remap_dicts_json  # noqa: E402


class OfflineRemapTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name) / "data"
        self.data_dir.mkdir()
        (self.data_dir / "feature_map.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_json_remap_round_trip_restores_integer_ids(self):
        output_path = self.data_dir / "mapped" / "remap_dict.json"
        remap_vocab_data.save_remap_dicts_json(
            {
                "item_id": {np.int64(1): np.int64(3), 5: 7},
                "category_id": {2: 4},
            },
            str(output_path),
        )

        raw = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["item_id"], {"1": 3, "5": 7})
        self.assertEqual(
            load_remap_dicts_json(str(output_path)),
            {"item_id": {1: 3, 5: 7}, "category_id": {2: 4}},
        )
        self.assertFalse((output_path.parent / "remap_dict.pkl").exists())

    def test_default_scan_uses_train_positive_or_train_only(self):
        for source_name in ("train_positive.parquet", "train.parquet"):
            with self.subTest(source_name=source_name):
                source_path = self.data_dir / source_name
                source_path.touch()
                proxy = object()
                with (
                    mock.patch.object(remap_vocab_data, "load_feature_map_for_scan", return_value=proxy),
                    mock.patch.object(remap_vocab_data, "scan_parquet_vocab", return_value={}) as scan_vocab,
                    mock.patch.object(remap_vocab_data, "build_vocab_mapping", return_value=VocabPruneInfo()),
                ):
                    remap_vocab_data.run_offline_remap(
                        str(self.data_dir),
                        output_subdir=f"mapped-{source_name}",
                    )
                scan_vocab.assert_called_once_with([str(source_path)], proxy)
                source_path.unlink()

    def test_explicit_scan_paths_remain_caller_controlled(self):
        explicit_paths = [str(self.data_dir / "approved_catalog.parquet")]
        proxy = object()
        with (
            mock.patch.object(remap_vocab_data, "load_feature_map_for_scan", return_value=proxy),
            mock.patch.object(remap_vocab_data, "scan_parquet_vocab", return_value={}) as scan_vocab,
            mock.patch.object(remap_vocab_data, "build_vocab_mapping", return_value=VocabPruneInfo()),
        ):
            remap_vocab_data.run_offline_remap(
                str(self.data_dir),
                output_subdir="mapped-explicit",
                scan_paths=explicit_paths,
            )
        scan_vocab.assert_called_once_with(explicit_paths, proxy)


if __name__ == "__main__":
    unittest.main()
