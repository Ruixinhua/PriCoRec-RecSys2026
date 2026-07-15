"""Regression tests for non-evaluating loss, initializer, and metric registries."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuxictr.metrics import _resolve_group_metric, evaluate_metrics, gAUC  # noqa: E402
from fuxictr.pytorch.torch_utils import get_initializer, get_loss  # noqa: E402


class SafeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_loss_registry_preserves_supported_bce_alias(self):
        loss = get_loss("binary_crossentropy")
        expected = torch.nn.functional.binary_cross_entropy(
            torch.tensor([0.8]), torch.tensor([1.0])
        )
        self.assertTrue(torch.isclose(loss(torch.tensor([0.8]), torch.tensor([1.0])), expected))
        with self.assertRaises(NotImplementedError):
            get_loss("not_a_loss")

    def test_initializer_registry_supports_default_partial_without_execution(self):
        initializer = get_initializer("partial(nn.init.normal_, std=1e-4)")
        tensor = torch.empty(8, 4)
        initializer(tensor)
        self.assertTrue(torch.isfinite(tensor).all())

        zero_initializer = get_initializer("nn.init.zeros_")
        zero_initializer(tensor)
        self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))

        sentinel = self.tmp / "initializer-executed"
        with self.assertRaises(ValueError):
            get_initializer(f"__import__('pathlib').Path({str(sentinel)!r}).touch()")
        self.assertFalse(sentinel.exists())

    def test_group_metric_registry_rejects_expression_payloads(self):
        self.assertIs(_resolve_group_metric("gAUC"), gAUC)
        self.assertEqual(_resolve_group_metric("NDCG(3)").topk, 3)
        self.assertEqual(_resolve_group_metric("NDCG@5").topk, 5)

        sentinel = self.tmp / "metric-executed"
        payload = f"NDCG(__import__('pathlib').Path({str(sentinel)!r}).touch())"
        with self.assertRaises(NotImplementedError):
            evaluate_metrics(
                np.array([0, 1]),
                np.array([0.1, 0.9]),
                [payload],
                group_id=np.array([1, 1]),
            )
        self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
