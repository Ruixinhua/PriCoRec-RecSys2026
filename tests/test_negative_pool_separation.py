"""Regression tests for separating train negative pools from evaluation pools."""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_device_recsys.pipeline.preranking_stage import PrerankingStage
from cloud_device_recsys.pipeline.reranking_stage import RerankingStage


class NegativePoolSeparationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.eval_pool_path = self.root / "eval_items.parquet"
        self.train_pool_path = self.root / "train_items.parquet"
        pd.DataFrame(
            {"item_id": [100, 101], "item_feature": [10, 11], "pool": ["eval", "eval"]}
        ).to_parquet(self.eval_pool_path, index=False)
        pd.DataFrame(
            {"item_id": [1, 2], "item_feature": [20, 21], "pool": ["train", "train"]}
        ).to_parquet(self.train_pool_path, index=False)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _bare_stage(stage_type):
        stage = object.__new__(stage_type)
        stage.feature_map = SimpleNamespace(dataset_config={"item_id_col": "item_id"})
        stage.logger = logging.getLogger(f"test.{stage_type.__name__}")
        stage.item_features_df = None
        stage.train_negative_item_features_df = None
        stage.negative_sampler = None
        stage.num_negatives = 1
        return stage

    def _assert_train_pool_is_used_for_negatives(self, stage):
        stage.load_item_features(str(self.eval_pool_path))
        self.assertEqual(stage.item_features_df.index.tolist(), [100, 101])
        self.assertEqual(stage.item_features_df.loc[100, "pool"], "eval")

        # Loading only an evaluation pool must not make holdout items eligible
        # for training negatives.
        self.assertFalse(stage._initialize_negative_sampler("item_id"))
        self.assertIsNone(stage.negative_sampler)

        stage.load_negative_item_features(str(self.train_pool_path))
        self.assertEqual(stage.item_features_df.index.tolist(), [100, 101])
        self.assertEqual(stage.train_negative_item_features_df.index.tolist(), [1, 2])

        self.assertTrue(stage._initialize_negative_sampler("item_id"))
        sampled = stage.negative_sampler.sample_negatives_batch(
            np.asarray([999], dtype=np.int64), num_negatives=1
        )
        self.assertTrue(set(sampled.reshape(-1)).issubset({1, 2}))
        sampled_features = stage.negative_sampler.get_features_by_ids(sampled.reshape(-1))
        self.assertEqual(set(sampled_features["pool"].tolist()), {"train"})

    def _assert_process_uses_eval_pool(self, stage):
        stage.model = object()
        stage.best_weights_path = str(self.root / "missing.model")
        stage.stage_name = "test-stage"
        stage.metrics_k = [1]
        stage.inference_batch_size = 8
        if isinstance(stage, PrerankingStage):
            stage.top_k = 1
            stage.evaluate_pool_diversity = False
            stage.popularity_blend_weight = 0.0
            stage.popularity_blend_transform = "log1p"
            stage.popularity_blend_normalize = "zscore"
            stage.popularity_model_bias_weight = 0.0
            process_kwargs = {"load_best_model": False}
        else:
            stage.top_k = 1
            stage.cloud_teacher_model = None
            stage.cloud_score_teacher = None
            stage.cloud_teacher_mode = None
            stage.use_cloud_score = False
            process_kwargs = {}

        with mock.patch(
            "cloud_device_recsys.metric_utils.process_and_rank_candidates",
            return_value=(None, {"Recall@1": 1.0}),
        ) as process_candidates:
            stage.process(SimpleNamespace(), **process_kwargs)
        self.assertIs(
            process_candidates.call_args.kwargs["item_features_df"],
            stage.item_features_df,
        )
        self.assertEqual(
            process_candidates.call_args.kwargs["item_features_df"].loc[100, "pool"],
            "eval",
        )

    def test_preranking_uses_train_pool_without_overwriting_eval_pool(self):
        stage = self._bare_stage(PrerankingStage)
        stage.negative_sampling_strategy = "uniform"
        stage.negative_sampling_popularity_alpha = 0.75
        stage.popularity_prior_counts = None
        stage.loss_type = "bpr"
        self._assert_train_pool_is_used_for_negatives(stage)
        self._assert_process_uses_eval_pool(stage)

    def test_reranking_uses_train_pool_without_overwriting_eval_pool(self):
        stage = self._bare_stage(RerankingStage)
        stage.loss_type = "bpr"
        self._assert_train_pool_is_used_for_negatives(stage)
        self._assert_process_uses_eval_pool(stage)


if __name__ == "__main__":
    unittest.main()
