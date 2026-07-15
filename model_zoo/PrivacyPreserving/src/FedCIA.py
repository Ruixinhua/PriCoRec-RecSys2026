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
FedCIA: Federated Collaborative Information Aggregation for
Privacy-Preserving Recommendation

Adapted from the SIGIR 2025 paper by Han et al.

Original Method:
  FedCIA proposes a novel federated aggregation paradigm that shares
  collaborative information (item similarity matrices) instead of raw model
  parameters between clients. Each client computes a local item similarity
  matrix from its embeddings, adds Laplace noise for privacy, and uploads
  it. The server averages these matrices, and each client aligns its local
  embeddings to reproduce the aggregated similarity via L2 loss.

Adaptation for Feature-Based CTR Prediction:
  In our cloud-device feature-separation scenario, we treat the cloud model
  and the device model as two "silos" that need to share collaborative
  information without exchanging raw features:

  1. Both models extract latent representations via forward_with_latent().
  2. Each model computes a batch-level latent similarity matrix
     (latent @ latent^T), capturing feature interaction co-occurrence patterns.
  3. Optional Laplace noise is added to the cloud similarity matrix before
     sharing (simulating the privacy mechanism in FedCIA).
  4. The device model aligns its similarity matrix to the cloud's via an
     L2 alignment loss, learning the collaborative interaction patterns
     without accessing personalized features directly.
  5. The cloud model also receives a mild alignment signal from the device
     similarity (bidirectional but asymmetric).

Scope note:
  This is an in-process research adapter, not a reference implementation of a
  distributed federated protocol. Optional Laplace noise is not paired with a
  sensitivity proof or privacy accountant, so this code makes no formal local
  differential-privacy guarantee.

Routing (inference):
  - group_id=1 (personalized): cloud model prediction.
  - group_id!=1 (non-personalized): device model prediction.

Supported backbones: PNN, FinalNet, DCNv3 (FCN).

Reference:
  Han et al., "FedCIA: Federated Collaborative Information Aggregation for
  Privacy-Preserving Recommendation", SIGIR 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import FeatureSeparator
from fuxictr.pytorch.backbone import build_backbone


class FedCIA(BaseModel):

    def __init__(self,
                 feature_map,
                 model_id="FedCIA",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 # FedCIA parameters
                 similarity_align_weight=1.0,    # weight for similarity alignment loss
                 reverse_align_weight=0.1,       # cloud aligns to device (mild)
                 laplace_noise_scale=0.001,      # Laplace noise scale for privacy
                 similarity_normalize=True,      # normalize similarity matrix
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

        super(FedCIA, self).__init__(
            feature_map, model_id=model_id, gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer, **kwargs)

        self.similarity_align_weight = similarity_align_weight
        self.reverse_align_weight = reverse_align_weight
        self.laplace_noise_scale = laplace_noise_scale
        self.similarity_normalize = similarity_normalize
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

        # Cloud model: uses ALL features
        self.cloud_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)
        # Device model: uses only NP features
        self.device_backbone = build_backbone(backbone_type, feature_map, **backbone_kwargs)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        logging.info(f"FedCIA initialized: backbone={backbone_type}, "
                     f"sim_align={similarity_align_weight}, "
                     f"reverse_align={reverse_align_weight}, "
                     f"laplace_noise={laplace_noise_scale}")

    def _get_personalized_mask(self, X):
        if self.personalization_field in X:
            flag = X[self.personalization_field]
            return (flag == 1), (flag != 1)
        batch_size = list(X.values())[0].size(0)
        device = list(X.values())[0].device
        return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.ones(batch_size, dtype=torch.bool, device=device))

    def _compute_similarity_matrix(self, latent, add_noise=False):
        """Compute batch-level latent similarity matrix.

        Args:
            latent: (batch_size, latent_dim) tensor
            add_noise: if True, add Laplace noise for privacy

        Returns:
            (batch_size, batch_size) similarity matrix
        """
        # L2 normalize latent for stable similarity
        if self.similarity_normalize:
            latent = F.normalize(latent, p=2, dim=-1)
        sim = torch.mm(latent, latent.t())  # (B, B)
        if add_noise and self.laplace_noise_scale > 0 and self.training:
            noise = torch.distributions.Laplace(0, self.laplace_noise_scale).sample(sim.shape)
            sim = sim + noise.to(sim.device)
        return sim

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
            "device_latent": device_latent,
            "cloud_latent": cloud_latent,
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

        # 3. Collaborative information alignment (FedCIA core)
        #    Cloud computes similarity matrix (with privacy noise) -> device aligns
        cloud_sim = self._compute_similarity_matrix(
            return_dict["cloud_latent"].detach(), add_noise=True)
        device_sim = self._compute_similarity_matrix(
            return_dict["device_latent"], add_noise=False)
        align_loss = F.mse_loss(device_sim, cloud_sim)
        total_loss = total_loss + self.similarity_align_weight * align_loss

        # 4. Reverse alignment: cloud mildly aligns to device (bidirectional)
        if self.reverse_align_weight > 0:
            cloud_sim_live = self._compute_similarity_matrix(
                return_dict["cloud_latent"], add_noise=False)
            device_sim_detached = self._compute_similarity_matrix(
                return_dict["device_latent"].detach(), add_noise=False)
            reverse_loss = F.mse_loss(cloud_sim_live, device_sim_detached)
            total_loss = total_loss + self.reverse_align_weight * reverse_loss

        return total_loss
