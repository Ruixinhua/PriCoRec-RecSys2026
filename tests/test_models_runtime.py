import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_device_recsys.config.config_parser import ConfigParser
from cloud_device_recsys.metric_utils import _process_lazy_all_items_candidates
from cloud_device_recsys.models.din_ranker import DINRanker
from cloud_device_recsys.pipeline.stage_output import StageOutput
from cloud_device_recsys.run_hyperparam_search import parse_args as parse_hyperparam_args
from cloud_device_recsys.utils import parse_pipeline_args
from model_zoo.PrivacyPreserving.src.DPSGD import DPSGD


class _FeatureMap:
    dataset_config = {"item_id_col": "cand_item_id"}
    features = {
        "cand_item_id": {"type": "categorical"},
        "item_value": {"type": "numeric"},
        "user_value": {"type": "numeric"},
    }


class _RecordingScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.max_batch_rows = 0
        self.forward_calls = 0

    def forward(self, inputs):
        row_count = next(iter(inputs.values())).shape[0]
        self.max_batch_rows = max(self.max_batch_rows, row_count)
        self.forward_calls += 1
        scores = inputs["item_value"].float() + inputs["user_value"].float()
        return {"logit": scores.reshape(-1, 1)}


class _FixedEmbedding(nn.Module):
    def forward(self, inputs):
        batch_size = inputs["history"].shape[0]
        return {
            "target": torch.zeros(batch_size, 1, dtype=torch.float32),
            "history": inputs["history_embedding"].float(),
        }


class _ZeroAttention(nn.Module):
    def forward(self, inputs):
        return torch.zeros(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)


class _TinyDIN(DINRanker):
    def __init__(self):
        nn.Module.__init__(self)
        self.sequence_features = ["history"]
        self.target_item_attention_key = "target"
        self.embedding_layer = _FixedEmbedding()
        self.attention_mlp = _ZeroAttention()
        self.final_feature_names = []
        self.mlp = nn.Linear(1, 1, bias=False)
        self.mlp.weight.data.fill_(1.0)
        self.output_activation = nn.Identity()
        self.use_diversity_loss = False

    def get_inputs(self, inputs):
        return inputs


class ModelsRuntimeTests(unittest.TestCase):
    def test_lazy_all_items_honors_item_side_inference_cap(self):
        item_features = pd.DataFrame(
            {"item_value": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=pd.Index([1, 2, 3, 4, 5], name="cand_item_id"),
        )
        stage_output = StageOutput.from_dataframes(
            stage_name="preranking",
            user_features_df=pd.DataFrame(
                {"request_id": [101, 202], "user_value": [10.0, 20.0]}
            ),
            candidates_df=pd.DataFrame(
                {
                    "request_id": [101, 202],
                    "item_id": [5, 5],
                    "label": [1, 1],
                }
            ),
            metadata={"split": "valid"},
        )
        model = _RecordingScoreModel()

        output, metrics = _process_lazy_all_items_candidates(
            model=model,
            feature_map=_FeatureMap(),
            input_data=stage_output,
            item_features_df=item_features,
            stage_name="preranking",
            return_output=True,
            compute_metrics=True,
            metrics_k=[1, 2],
            top_k=2,
            inference_batch_size=3,
        )

        self.assertLessEqual(model.max_batch_rows, 3)
        self.assertEqual(model.forward_calls, 4)
        self.assertEqual(
            output.candidates_df.groupby("request_id")["item_id"].apply(list).to_dict(),
            {101: [5, 4], 202: [5, 4]},
        )
        self.assertEqual(metrics["Recall@1"], 1.0)

    def test_din_attention_excludes_padding_and_handles_empty_history(self):
        model = _TinyDIN()
        result = model(
            {
                "history": torch.tensor([[7, 0]], dtype=torch.long),
                "history_embedding": torch.tensor([[[2.0], [100.0]]]),
            }
        )
        self.assertTrue(torch.allclose(result["y_pred"], torch.tensor([[2.0]])))

        empty_result = model(
            {
                "history": torch.tensor([[0, 0]], dtype=torch.long),
                "history_embedding": torch.tensor([[[2.0], [100.0]]]),
            }
        )
        self.assertTrue(torch.isfinite(empty_result["y_pred"]).all())
        self.assertTrue(torch.allclose(empty_result["y_pred"], torch.zeros((1, 1))))

    def test_dpsgd_is_disabled_until_verified_dp_implementation_exists(self):
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            DPSGD(object())

    def test_default_config_parser_uses_tracked_smoke_pipeline(self):
        config_dir = PROJECT_ROOT / "cloud_device_recsys" / "config"
        config = ConfigParser(config_dir).load_pipeline_config()
        self.assertEqual(config["dataset_id"], "SmokeTest")

    def test_pipeline_cli_accepts_implemented_save_preranking_mode(self):
        with mock.patch.object(sys, "argv", ["run_pipeline", "--mode", "save_preranking_outputs"]):
            args = parse_pipeline_args()
        self.assertEqual(args.mode, "save_preranking_outputs")

    def test_hyperparameter_cli_rejects_unimplemented_stage_modes(self):
        for mode in ("dtcn_preranking", "joint_train"):
            with self.subTest(mode=mode), mock.patch.object(
                sys, "argv", ["run_hyperparam_search", "--mode", mode]
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_hyperparam_args()


if __name__ == "__main__":
    unittest.main()
