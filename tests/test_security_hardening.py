"""Focused regression tests for config and artifact trust boundaries."""

import os
import pickle
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_device_recsys.data.vocab_pruner import (  # noqa: E402
    FeaturePruneInfo,
    VocabPruneInfo,
    _deserialize_prune_info,
    _safe_torch_load,
    _serialize_prune_info,
    compute_vocab_pruning,
)
from cloud_device_recsys.pipeline.stage_output import (  # noqa: E402
    StageOutput,
    UnsafeStageOutputFormatError,
)
from cloud_device_recsys.run_hyperparam_search import parse_args as parse_search_args  # noqa: E402
from cloud_device_recsys.utils import (  # noqa: E402
    load_remap_dicts_json,
    load_stage_output,
    load_stage_outputs_from_dir,
    parse_pipeline_args,
    resolve_pipeline_config_path,
    validate_experiment_id,
    validate_pipeline_id,
)
from fuxictr.preprocess.feature_processor import FeatureProcessor  # noqa: E402
from fuxictr.pytorch import layers  # noqa: E402
from fuxictr.pytorch.layers.embeddings.feature_embedding import FeatureEmbeddingDict  # noqa: E402
from fuxictr.pytorch.models.rank_model import BaseModel  # noqa: E402


class _EvilPickle:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        return os.system, (f"touch {shlex.quote(str(self.sentinel))}",)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = nn.Linear(2, 1)


class SecurityHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_dtype_allowlist_accepts_compatible_values_and_rejects_code(self):
        processor = FeatureProcessor(
            feature_cols=[{"name": "feature", "dtype": "int64", "type": "numeric"}],
            label_col={"name": "label", "dtype": "float"},
            dataset_id="security-test",
            data_root=str(self.tmp),
        )
        self.assertIs(processor.dtype_dict["feature"], np.int64)

        sentinel = self.tmp / "dtype-executed"
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            FeatureProcessor(
                feature_cols=[
                    {
                        "name": "feature",
                        "dtype": f"__import__('pathlib').Path({str(sentinel)!r}).touch()",
                        "type": "numeric",
                    }
                ],
                label_col={"name": "label", "dtype": "float"},
                dataset_id="security-test",
                data_root=str(self.tmp),
            )
        self.assertFalse(sentinel.exists())

    def test_feature_encoder_allowlist_rejects_expressions(self):
        owner = object()
        encoder = FeatureEmbeddingDict.get_feature_encoder(owner, "layers.MaskedAveragePooling()")
        self.assertIsInstance(encoder, layers.MaskedAveragePooling)
        sequence = FeatureEmbeddingDict.get_feature_encoder(
            owner,
            ["layers.MaskedAveragePooling()", "layers.MaskedSumPooling()"],
        )
        self.assertIsInstance(sequence, nn.Sequential)

        sentinel = self.tmp / "encoder-executed"
        with self.assertRaisesRegex(ValueError, "not supported"):
            FeatureEmbeddingDict.get_feature_encoder(
                owner,
                f"__import__('pathlib').Path({str(sentinel)!r}).touch()",
            )
        self.assertFalse(sentinel.exists())

    def test_stage_output_refuses_legacy_pickle_artifacts(self):
        legacy_path = self.tmp / "retrieval_valid_stage_output.pkl"
        with legacy_path.open("wb") as handle:
            pickle.dump(_EvilPickle(self.tmp / "stage-executed"), handle)

        with self.assertRaisesRegex(ValueError, "Refusing legacy pickle"):
            load_stage_output(str(legacy_path))
        with self.assertRaisesRegex(ValueError, "Refusing legacy pickle"):
            load_stage_outputs_from_dir(str(self.tmp), "retrieval", load_test=False)
        with self.assertRaises(UnsafeStageOutputFormatError):
            StageOutput.load(str(legacy_path))
        self.assertFalse((self.tmp / "stage-executed").exists())

    def test_checkpoint_loader_accepts_state_dict_and_rejects_executable_artifact(self):
        checkpoint_path = self.tmp / "weights.model"
        source = _TinyModel()
        torch.save(source.state_dict(), checkpoint_path)
        target = _TinyModel()
        with torch.no_grad():
            target.linear.weight.zero_()
            target.linear.bias.zero_()
        BaseModel.load_weights(target, checkpoint_path)
        self.assertTrue(torch.equal(target.linear.weight, source.linear.weight))

        sentinel = self.tmp / "checkpoint-executed"
        malicious_path = self.tmp / "malicious.model"
        torch.save(_EvilPickle(sentinel), malicious_path)
        with self.assertRaisesRegex(RuntimeError, "unsafe or unsupported"):
            BaseModel.load_weights(_TinyModel(), malicious_path)
        self.assertFalse(sentinel.exists())

    def test_vocab_cache_uses_tensor_payload_and_ignores_legacy_pickle(self):
        info = VocabPruneInfo(
            features={
                "item": FeaturePruneInfo(
                    feature_name="item",
                    original_vocab_size=3,
                    compact_vocab_size=3,
                    remap_table=torch.tensor([0, 1, 2], dtype=torch.int32),
                    reduction_ratio=0.0,
                )
            },
            shared_groups={"item": {"item"}},
        )
        cache_path = self.tmp / "vocab_prune_cache.pt"
        torch.save(_serialize_prune_info(info), cache_path)
        restored = _deserialize_prune_info(_safe_torch_load(str(cache_path)))
        self.assertTrue(torch.equal(restored.features["item"].remap_table, info.features["item"].remap_table))
        cache_path.unlink()

        sentinel = self.tmp / "vocab-executed"
        with (self.tmp / "vocab_prune_cache.pkl").open("wb") as handle:
            pickle.dump(_EvilPickle(sentinel), handle)
        empty_feature_map = SimpleNamespace(features={}, total_features=0, data_dir=str(self.tmp))
        result = compute_vocab_pruning(empty_feature_map, str(self.tmp / "missing.parquet"))
        self.assertEqual(result.features, {})
        self.assertFalse(sentinel.exists())

    def test_safe_identifiers_and_remap_schema(self):
        self.assertEqual(validate_experiment_id("run-20260710"), "run-20260710")
        self.assertEqual(validate_pipeline_id("pipeline_config/smoke_pipeline"), "pipeline_config/smoke_pipeline")
        with self.assertRaises(ValueError):
            validate_experiment_id("../escape")
        with self.assertRaises(ValueError):
            validate_pipeline_id("../escape")
        with self.assertRaises(ValueError):
            validate_pipeline_id(r"pipeline_config\\escape")

        config_dir = self.tmp / "config"
        config_dir.mkdir()
        resolved = Path(resolve_pipeline_config_path(str(config_dir), "nested/pipeline"))
        self.assertTrue(resolved.is_relative_to(config_dir.resolve()))

        remap_path = self.tmp / "remap_dict.json"
        remap_path.write_text('{"item_id": {"1": 2}}', encoding="utf-8")
        self.assertEqual(load_remap_dicts_json(str(remap_path)), {"item_id": {1: 2}})
        remap_path.write_text('{"item_id": {"1": -2}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_remap_dicts_json(str(remap_path))

    def test_cli_modes_are_consistent(self):
        with mock.patch.object(sys, "argv", ["prog", "--mode", "save_preranking_outputs"]):
            self.assertEqual(parse_pipeline_args().mode, "save_preranking_outputs")
        with mock.patch.object(
            sys,
            "argv",
            ["prog", "--search_config", "search.yaml", "--base_config", "base.yaml", "--mode", "dtcn_preranking"],
        ):
            with self.assertRaises(SystemExit):
                parse_search_args()

if __name__ == "__main__":
    unittest.main()
