import unittest

import torch

from cloud_device_recsys.models.losses import compute_diversity_loss_per_user


class _DiversityModel:
    _diversity_enabled = True
    _diversity_item_features = ["item_embedding"]

    @staticmethod
    def get_inputs(batch):
        return batch

    @staticmethod
    def _embedding_dict(inputs):
        return {"item_embedding": inputs["item_embedding"]}

    _diversity_emb_dict_layer = _embedding_dict


class DiversityLossTests(unittest.TestCase):
    def test_pairwise_logit_prediction_mean_uses_probability_scale(self):
        model = _DiversityModel()
        pos_inputs = {"item_embedding": torch.tensor([[1.0, 0.0]])}
        neg_inputs = {"item_embedding": torch.tensor([[0.0, 1.0]])}
        scores = torch.zeros(1, 1)

        logit_loss = compute_diversity_loss_per_user(
            model,
            pos_inputs,
            neg_inputs,
            scores,
            scores,
            num_negatives=1,
            theta=1.0,
            lambda_=1.0,
            kernel="rbf",
            scores_are_logits=True,
        )
        probability_loss = compute_diversity_loss_per_user(
            model,
            pos_inputs,
            neg_inputs,
            scores,
            scores,
            num_negatives=1,
            theta=1.0,
            lambda_=1.0,
            kernel="rbf",
            scores_are_logits=False,
        )

        self.assertAlmostEqual(float(logit_loss), -0.5, places=6)
        self.assertAlmostEqual(float(probability_loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
