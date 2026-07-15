# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
DualRec: A Collaborative Training Framework for Device and Cloud Recommendation Models

Implements a cloud-device collaborative training framework where a large cloud
model (using all features) and a lightweight device model (using only
non-personalized features) are jointly trained with bidirectional knowledge
distillation.

Architecture:
  - Cloud model: full-feature backbone (uses personalized + non-personalized features)
  - Device model: NP-feature backbone (uses only non-personalized features)
  - Both models are trained end-to-end simultaneously.

Training:
  1. Both models compute predictions on all data.
  2. Cloud model is supervised with BCE on labeled data.
  3. Device model is supervised with BCE + two distillation objectives:
     (a) Cloud-to-Device KD: soft predictions from cloud guide device model
         via KL divergence (on personalized samples where cloud has richer info).
     (b) Output Distribution Regularization (ODR): a bidirectional KL term
         that aligns the overall output distributions of cloud and device models,
         encouraging the device model to capture complementary knowledge.
  4. The cloud model also receives a mild regularization signal from the device
     model's gradient (mutual regularization), preventing overfitting.

Scope note:
  This is an in-process research adapter. It does not implement or verify a
  distributed federated transport, secure aggregation, or deployment privacy
  boundary, and it must not be used as evidence of a formal privacy guarantee.

Routing (inference):
  - group_id=1 (personalized): cloud model prediction
  - group_id!=1 (non-personalized): device model prediction

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Zhang et al., "DualRec: A Collaborative Training Framework for Device and
  Cloud Recommendation Models", IEEE TKDE, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from fuxictr.pytorch.backbone import build_backbone


class DualRec(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="DualRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # DualRec KD parameters
                 kd_temperature=4.0,
                 kd_loss_weight=1.0,         # cloud->device KD weight
                 odr_loss_weight=0.5,        # output distribution regularization weight
                 mutual_reg_weight=0.1,      # device->cloud mutual regularization weight
                 base_loss_weight=1.0,       # device BCE weight
                 cloud_loss_weight=1.0,      # cloud BCE weight
                 kd_loss_type="kl",          # "kl", "mse", "cosine"
                 # Inference routing
                 inference_model="auto",     # "auto", "cloud", "device"
                 # Feature separation
                 personalization_feature_list=None,
                 personalization_field="is_personalization",
                 # Backbone config
                 backbone_type="PNN",
                 # PNN params
                 hidden_units=[400, 400, 400],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 # FinalNet params
                 block_type="2B",
                 use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64],
                 block1_hidden_activations=None,
                 block1_dropout=0,
                 block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None,
                 block2_dropout=0,
                 residual_type="concat",
                 # DCNv3 params
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 num_heads=1,
                 # Standard params
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):

        super(DualRec, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.odr_loss_weight = odr_loss_weight
        self.mutual_reg_weight = mutual_reg_weight
        self.base_loss_weight = base_loss_weight
        self.cloud_loss_weight = cloud_loss_weight
        self.kd_loss_type = kd_loss_type
        self.inference_model = inference_model
        self.personalization_feature_list = personalization_feature_list or []
        self.personalization_field = personalization_field

        self.personalization_feature_list = [
            f for f in self.personalization_feature_list
            if f in self.feature_map.features
        ]
        self.feature_separator = FeatureSeparator(
            self.personalization_feature_list, self.feature_map
        )

        backbone_kwargs = dict(
            embedding_dim=embedding_dim,
            hidden_units=hidden_units, hidden_activations=hidden_activations,
            net_dropout=net_dropout, batch_norm=batch_norm, product_type=product_type,
            block_type=block_type, use_feature_gating=use_feature_gating,
            block1_hidden_units=block1_hidden_units,
            block1_hidden_activations=block1_hidden_activations,
            block1_dropout=block1_dropout,
            block2_hidden_units=block2_hidden_units,
            block2_hidden_activations=block2_hidden_activations,
            block2_dropout=block2_dropout,
            residual_type=residual_type,
            num_deep_cross_layers=num_deep_cross_layers,
            num_shallow_cross_layers=num_shallow_cross_layers,
            deep_net_dropout=deep_net_dropout, shallow_net_dropout=shallow_net_dropout,
            layer_norm=layer_norm, num_heads=num_heads,
        )

        # Cloud model: uses ALL features
        self.cloud_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Device model: uses only NP features
        self.device_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"DualRec initialized: backbone={backbone_type}, "
                     f"T={kd_temperature}, kd_weight={kd_loss_weight}, "
                     f"odr_weight={odr_loss_weight}, mutual_reg={mutual_reg_weight}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        # Device model sees only NP features
        _, device_X = self.feature_separator.separate_features(X, personalized_mask)
        device_logit = self.device_backbone(device_X)
        device_y_pred = self.output_activation(device_logit)

        # Cloud model sees ALL features
        cloud_logit = self.cloud_backbone(X)
        cloud_y_pred = self.output_activation(cloud_logit)

        # Route predictions based on inference_model setting
        if self.inference_model == "cloud":
            final_pred = cloud_y_pred
        elif self.inference_model == "device":
            final_pred = device_y_pred
        else:  # "auto" — original routing
            final_pred = torch.zeros_like(device_y_pred)
            if personalized_mask.any():
                final_pred[personalized_mask] = cloud_y_pred[personalized_mask]
            if non_personalized_mask.any():
                final_pred[non_personalized_mask] = device_y_pred[non_personalized_mask]

        return_dict = {
            "y_pred": final_pred,
            "device_y_pred": device_y_pred,
            "device_logit": device_logit,
            "cloud_y_pred": cloud_y_pred,
            "cloud_logit": cloud_logit,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
        return return_dict

    def add_loss(self, return_dict, y_true):
        personalized_mask = return_dict["personalized_mask"]

        # 1. Device BCE loss on all data
        device_bce = self.loss_fn(return_dict["device_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * device_bce

        # 2. Cloud BCE loss on all data
        cloud_bce = self.loss_fn(return_dict["cloud_y_pred"], y_true, reduction='mean')
        total_loss = total_loss + self.cloud_loss_weight * cloud_bce

        if personalized_mask.any():
            # 3. Cloud-to-Device KD loss (on personalized samples only)
            #    Cloud has richer features -> distill to device
            c2d_kd_loss = self._compute_kd_loss(
                return_dict["device_logit"][personalized_mask],
                return_dict["cloud_logit"][personalized_mask].detach())
            total_loss = total_loss + self.kd_loss_weight * c2d_kd_loss

            # 4. Mutual regularization: device -> cloud (mild)
            #    Prevents cloud from overfitting, stabilizes training
            if self.mutual_reg_weight > 0:
                d2c_reg_loss = self._compute_kd_loss(
                    return_dict["cloud_logit"][personalized_mask],
                    return_dict["device_logit"][personalized_mask].detach())
                total_loss = total_loss + self.mutual_reg_weight * d2c_reg_loss

        # 5. Output Distribution Regularization (ODR) on ALL samples
        #    Aligns device and cloud output distributions bidirectionally
        if self.odr_loss_weight > 0:
            odr_loss = self._compute_odr_loss(
                return_dict["device_logit"],
                return_dict["cloud_logit"])
            total_loss = total_loss + self.odr_loss_weight * odr_loss

        return total_loss

    def _compute_kd_loss(self, student_logit, teacher_logit):
        """Compute KD loss from teacher to student."""
        T = self.kd_temperature
        if self.kd_loss_type == "kl":
            s_prob = torch.sigmoid(student_logit / T).clamp(1e-7, 1 - 1e-7)
            t_prob = torch.sigmoid(teacher_logit / T).clamp(1e-7, 1 - 1e-7)
            kd_loss = (t_prob * torch.log(t_prob / s_prob) +
                       (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob)))
            return kd_loss.mean() * (T * T)
        elif self.kd_loss_type == "mse":
            return F.mse_loss(student_logit, teacher_logit)
        elif self.kd_loss_type == "cosine":
            return 1 - F.cosine_similarity(student_logit, teacher_logit, dim=-1).mean()
        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")

    def _compute_odr_loss(self, device_logit, cloud_logit):
        """Output Distribution Regularization: symmetric KL between
        device and cloud output distributions across the batch.

        This computes the mean absolute difference of the batch-level
        output statistics (mean, variance) between cloud and device models,
        encouraging them to produce similarly distributed predictions.
        """
        device_prob = torch.sigmoid(device_logit).clamp(1e-7, 1 - 1e-7)
        cloud_prob = torch.sigmoid(cloud_logit).clamp(1e-7, 1 - 1e-7)

        # Symmetric KL divergence per sample, then average
        kl_dc = (device_prob * torch.log(device_prob / cloud_prob.detach()) +
                 (1 - device_prob) * torch.log((1 - device_prob) / (1 - cloud_prob.detach())))
        kl_cd = (cloud_prob * torch.log(cloud_prob / device_prob.detach()) +
                 (1 - cloud_prob) * torch.log((1 - cloud_prob) / (1 - device_prob.detach())))

        return (kl_dc.mean() + kl_cd.mean()) / 2.0
