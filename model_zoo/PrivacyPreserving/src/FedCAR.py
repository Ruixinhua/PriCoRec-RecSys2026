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
FedCAR: Federated Context-Aware Personalized Recommendation

Adapted from the AAAI 2026 paper by Wang et al.

Original Method:
  FedCAR proposes a federated recommendation framework that leverages users'
  recent interactions as behavioral context for prediction. Instead of static
  user embeddings, FedCAR dynamically constructs context representations by
  aggregating recent item embeddings. A contrastive learning strategy aligns
  local and global behavioral structures while maintaining personalized
  preferences, enhancing both generalization and robustness in heterogeneous
  federated environments.

Adaptation for Feature-Based CTR Prediction:
  In our cloud-device feature-separation scenario:

  1. Cloud model (global) uses ALL features and learns complete interaction
     patterns across all users.
  2. Device model (local) uses only NP features and captures device-side
     behavioral patterns.
  3. A projection layer maps each model's latent representation to a shared
     contrastive space.
  4. Contrastive learning (InfoNCE) treats (cloud_repr, device_repr) of the
     same sample as positive pairs and representations from different samples
     as negatives. This transfers behavioral structure knowledge from cloud
     to device without sharing raw features or predictions.
  5. An optional momentum-updated global prototype (EMA of cloud
     representations) provides stable alignment targets for the device model.

Scope note:
  This is an in-process research adapter. It does not implement or verify a
  distributed federated transport, secure aggregation, or deployment privacy
  boundary, and it must not be used as evidence of a formal privacy guarantee.

Routing (inference):
  - group_id=1 (personalized): cloud model prediction.
  - group_id!=1 (non-personalized): device model prediction.

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Wang et al., "Federated Context-Aware Personalized Recommendation",
  Proceedings of the AAAI Conference on Artificial Intelligence, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from fuxictr.pytorch.backbone import build_backbone


class FedCAR(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="FedCAR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # FedCAR parameters
                 contrastive_weight=0.5,         # weight for contrastive loss
                 contrastive_temperature=0.1,    # InfoNCE temperature
                 projection_dim=128,             # contrastive projection head dim
                 use_prototype=True,             # use EMA prototype for stability
                 prototype_momentum=0.99,        # EMA momentum for prototype
                 prototype_weight=0.1,           # prototype alignment weight
                 base_loss_weight=1.0,           # device BCE weight
                 cloud_loss_weight=1.0,          # cloud BCE weight
                 # Inference routing
                 inference_model="auto",         # "auto", "cloud", "device"
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

        super(FedCAR, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_prototype = use_prototype
        self.prototype_momentum = prototype_momentum
        self.prototype_weight = prototype_weight
        self.base_loss_weight = base_loss_weight
        self.cloud_loss_weight = cloud_loss_weight
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

        # Cloud model: uses ALL features (global model)
        self.cloud_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Device model: uses only NP features (local model)
        self.device_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        # Contrastive projection heads: map latent to shared contrastive space
        latent_dim = self.cloud_backbone.latent_dim
        self.cloud_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim),
        )
        self.device_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim),
        )

        # EMA global prototype (momentum-updated mean of cloud projections)
        if self.use_prototype:
            self.register_buffer('global_prototype',
                                 torch.zeros(projection_dim))
            self._prototype_initialized = False

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"FedCAR initialized: backbone={backbone_type}, "
                     f"contrastive_weight={contrastive_weight}, "
                     f"temperature={contrastive_temperature}, "
                     f"projection_dim={projection_dim}, "
                     f"prototype={'on' if use_prototype else 'off'}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _update_prototype(self, cloud_proj):
        """Update global prototype with EMA of cloud projections."""
        batch_mean = cloud_proj.detach().mean(dim=0)
        if not self._prototype_initialized:
            self.global_prototype.copy_(batch_mean)
            self._prototype_initialized = True
        else:
            self.global_prototype.mul_(self.prototype_momentum).add_(
                batch_mean, alpha=1 - self.prototype_momentum)

    def _info_nce_loss(self, z_i, z_j):
        """Compute InfoNCE contrastive loss.

        Args:
            z_i: (B, projection_dim) — anchor representations
            z_j: (B, projection_dim) — positive representations
            Negatives are all other samples in the batch.

        Returns:
            Scalar InfoNCE loss.
        """
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        batch_size = z_i.size(0)

        # Similarity matrix: (2B, 2B)
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        similarity = torch.mm(representations, representations.t()) / self.contrastive_temperature

        # Mask out self-similarity
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity = similarity.masked_fill(~mask, float('-inf'))

        # For each z_i[k], positive is z_j[k] (at index k + B)
        # For each z_j[k], positive is z_i[k] (at index k)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device),
        ])

        loss = F.cross_entropy(similarity, labels)
        return loss

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        personalized_mask, non_personalized_mask = self._get_personalized_mask(X)

        # Device model sees only NP features
        _, device_X = self.feature_separator.separate_features(X, personalized_mask)
        device_logit, device_latent = self.device_backbone.forward_with_latent(device_X)
        device_y_pred = self.output_activation(device_logit)

        # Cloud model sees ALL features
        cloud_logit, cloud_latent = self.cloud_backbone.forward_with_latent(X)
        cloud_y_pred = self.output_activation(cloud_logit)

        # Project to contrastive space
        cloud_proj = self.cloud_projector(cloud_latent)
        device_proj = self.device_projector(device_latent)

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
            "cloud_y_pred": cloud_y_pred,
            "cloud_proj": cloud_proj,
            "device_proj": device_proj,
            "personalized_mask": personalized_mask,
            "non_personalized_mask": non_personalized_mask,
        }
        return return_dict

    def add_loss(self, return_dict, y_true):
        # 1. Device BCE on all data
        device_bce = self.loss_fn(return_dict["device_y_pred"], y_true, reduction='mean')
        total_loss = self.base_loss_weight * device_bce

        # 2. Cloud BCE on all data
        cloud_bce = self.loss_fn(return_dict["cloud_y_pred"], y_true, reduction='mean')
        total_loss = total_loss + self.cloud_loss_weight * cloud_bce

        # 3. Contrastive alignment (FedCAR core)
        #    InfoNCE between cloud and device projections
        cl_loss = self._info_nce_loss(
            return_dict["cloud_proj"].detach(),  # stop gradient to cloud
            return_dict["device_proj"])
        total_loss = total_loss + self.contrastive_weight * cl_loss

        # 4. Prototype alignment (optional)
        #    Device projections align to global prototype
        if self.use_prototype and self.training:
            self._update_prototype(return_dict["cloud_proj"])
            device_proj_norm = F.normalize(return_dict["device_proj"], dim=-1)
            proto_norm = F.normalize(self.global_prototype.unsqueeze(0), dim=-1)
            proto_loss = 1 - (device_proj_norm * proto_norm).sum(dim=-1).mean()
            total_loss = total_loss + self.prototype_weight * proto_loss

        return total_loss
