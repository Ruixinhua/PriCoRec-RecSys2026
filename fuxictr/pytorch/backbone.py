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
Unified backbone factory for CTR prediction models.

Supports PNN, FinalNet, and DCNv3 (FCN) backbones with a consistent interface.
Each backbone wraps: embedding -> feature interaction -> logit.

Interface (all backbones):
  - forward(X) -> raw logit tensor [B, 1]
  - forward_with_latent(X) -> (raw_logit, latent_features)
  - forward_features(X) -> intermediate features before prediction head
  - get_model_return_dict(X) -> dict with "y_pred" (activated if output_activation set)
  - has_custom_loss() -> bool
  - compute_custom_loss(return_dict, y_true, loss_fn) -> scalar loss
  - latent_dim: int -- dimension of latent features
  - embedding_layer: FeatureEmbedding -- the embedding layer (for CL methods)

Used by: KD, PrivacyPreserving, DTDN, and DTCN model series.
"""

import torch
import torch.nn as nn

from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import get_activation


# =====================================================================
# PNN Backbone
# =====================================================================

class PNNBackbone(nn.Module):
    """PNN backbone: embedding -> inner product -> MLP -> logit."""

    def __init__(self, feature_map, embedding_dim=10, hidden_units=[400, 400, 400],
                 hidden_activations="ReLU", net_dropout=0, batch_norm=False,
                 product_type="inner", output_activation=None, **kwargs):
        super(PNNBackbone, self).__init__()
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        if product_type != "inner":
            raise NotImplementedError(f"product_type={product_type} not implemented.")
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
        input_dim = int(num_fields * (num_fields - 1) / 2) + num_fields * embedding_dim
        # Accept both net_dropout and dropout_rates (alias used by DTDN)
        dropout = kwargs.get("dropout_rates", net_dropout)
        self.dnn = MLP_Block(input_dim=input_dim, output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=dropout,
                             batch_norm=batch_norm)
        self.latent_dim = hidden_units[-1] if hidden_units else input_dim
        self._output_activation = output_activation
        self._feature_input_dim = input_dim

    def forward(self, X):
        """Forward pass returning raw logit."""
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        return self.dnn(dense_input)

    def forward_with_latent(self, X):
        """Return (raw_logit, latent_features) for latent alignment."""
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        layers = list(self.dnn.mlp)
        x = dense_input
        for layer in layers[:-1]:
            x = layer(x)
        logit = layers[-1](x)
        return logit, x

    def forward_features(self, X):
        """Return feature-interaction output before MLP (embedding + inner products)."""
        feature_emb = self.embedding_layer(X)
        inner_products = self.inner_product_layer(feature_emb)
        return torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)

    def _apply_activation(self, logit):
        if self._output_activation is not None:
            return self._output_activation(logit)
        return logit

    def get_model_return_dict(self, X):
        """Return dict with activated y_pred."""
        logit = self.forward(X)
        return {"y_pred": self._apply_activation(logit)}

    def has_custom_loss(self):
        return False

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        return loss_fn(return_dict["y_pred"], y_true, reduction='mean')


# =====================================================================
# FinalNet Backbone
# =====================================================================

class FinalNetBackbone(nn.Module):
    """FinalNet backbone: embedding -> FinalBlock(s) -> logit."""

    def __init__(self, feature_map, embedding_dim=10, block_type="2B",
                 batch_norm=True, use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64], block1_hidden_activations=None,
                 block1_dropout=0, block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None, block2_dropout=0,
                 residual_type="concat", output_activation=None, **kwargs):
        super(FinalNetBackbone, self).__init__()
        assert block_type in ["1B", "2B"], f"block_type={block_type} not supported."
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.use_feature_gating = use_feature_gating
        self.block_type = block_type
        self._output_activation = output_activation

        if use_feature_gating:
            self.feature_gating = _FeatureGating(num_fields, gate_residual="concat")
            gate_out_dim = embedding_dim * num_fields * 2
        else:
            gate_out_dim = embedding_dim * num_fields

        self.block1 = _FinalBlock(input_dim=gate_out_dim,
                                  hidden_units=block1_hidden_units,
                                  hidden_activations=block1_hidden_activations,
                                  dropout_rates=block1_dropout,
                                  batch_norm=batch_norm,
                                  residual_type=residual_type)
        self.fc1 = nn.Linear(block1_hidden_units[-1], 1)

        if block_type == "2B":
            self.block2 = _FinalBlock(input_dim=embedding_dim * num_fields,
                                      hidden_units=block2_hidden_units,
                                      hidden_activations=block2_hidden_activations,
                                      dropout_rates=block2_dropout,
                                      batch_norm=batch_norm,
                                      residual_type=residual_type)
            self.fc2 = nn.Linear(block2_hidden_units[-1], 1)

        self.latent_dim = block1_hidden_units[-1]

    def forward(self, X):
        """Forward pass returning raw logit."""
        feature_emb = self.embedding_layer(X)
        if self.block_type == "1B":
            logit = self._forward1(feature_emb)
        else:
            y1 = self._forward1(feature_emb)
            y2 = self._forward2(feature_emb)
            logit = 0.5 * (y1 + y2)
        return logit

    def forward_with_latent(self, X):
        """Return (raw_logit, block1_output_as_latent)."""
        feature_emb = self.embedding_layer(X)
        block1_out = self.block1(self._gate_input(feature_emb))
        latent = block1_out
        y1 = self.fc1(block1_out)
        if self.block_type == "2B":
            block2_out = self.block2(feature_emb.flatten(start_dim=1))
            y2 = self.fc2(block2_out)
            logit = 0.5 * (y1 + y2)
        else:
            logit = y1
        return logit, latent

    def forward_features(self, X):
        """Return block1 output (intermediate features before prediction head)."""
        feature_emb = self.embedding_layer(X)
        return self.block1(self._gate_input(feature_emb))

    def _gate_input(self, feature_emb):
        if self.use_feature_gating:
            return self.feature_gating(feature_emb).flatten(start_dim=1)
        return feature_emb.flatten(start_dim=1)

    def _forward1(self, feature_emb):
        x = self._gate_input(feature_emb)
        block1_out = self.block1(x)
        return self.fc1(block1_out)

    def _forward2(self, feature_emb):
        block2_out = self.block2(feature_emb.flatten(start_dim=1))
        return self.fc2(block2_out)

    def _apply_activation(self, logit):
        if self._output_activation is not None:
            return self._output_activation(logit)
        return logit

    def get_model_return_dict(self, X):
        """Return dict with activated y_pred and raw y1/y2 for custom loss."""
        feature_emb = self.embedding_layer(X)
        return_dict = {}
        if self.block_type == "1B":
            y_pred = self._forward1(feature_emb)
            y_pred = self._apply_activation(y_pred)
            return_dict["y_pred"] = y_pred
        else:
            y1 = self._forward1(feature_emb)
            y2 = self._forward2(feature_emb)
            y_pred = 0.5 * (y1 + y2)
            y_pred = self._apply_activation(y_pred)
            return_dict["y_pred"] = y_pred
            return_dict["y1"] = y1  # raw (not activated)
            return_dict["y2"] = y2  # raw (not activated)
        return return_dict

    def has_custom_loss(self):
        return self.block_type == "2B"

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        """FinalNet 2B custom loss: main + block-level distillation."""
        loss = loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.block_type == "2B":
            y1 = self._apply_activation(return_dict["y1"])
            y2 = self._apply_activation(return_dict["y2"])
            loss1 = loss_fn(y1, return_dict["y_pred"].detach(), reduction='mean')
            loss2 = loss_fn(y2, return_dict["y_pred"].detach(), reduction='mean')
            loss = loss + loss1 + loss2
        return loss


# =====================================================================
# DCNv3 (FCN) Backbone
# =====================================================================

class DCNv3Backbone(nn.Module):
    """DCNv3 (FCN) backbone: MultiHeadEmbedding -> ECN + LCN -> logit."""

    def __init__(self, feature_map, embedding_dim=10,
                 num_deep_cross_layers=4, num_shallow_cross_layers=4,
                 deep_net_dropout=0.1, shallow_net_dropout=0.3,
                 layer_norm=True, batch_norm=False, num_heads=1,
                 output_activation=None, **kwargs):
        super(DCNv3Backbone, self).__init__()
        num_fields = feature_map.num_fields
        self.num_heads = num_heads
        self._output_activation = output_activation

        self.embedding_layer = _MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads)

        cross_input_dim = num_fields * embedding_dim

        self.ECN = _ExponentialCrossNetwork(
            input_dim=cross_input_dim, num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout, layer_norm=layer_norm,
            batch_norm=batch_norm, num_heads=num_heads)
        self.LCN = _LinearCrossNetwork(
            input_dim=cross_input_dim, num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout, layer_norm=layer_norm,
            batch_norm=batch_norm, num_heads=num_heads)

        self.latent_dim = cross_input_dim

    def forward(self, X):
        """Forward pass returning raw logit."""
        feature_emb = self.embedding_layer(X)
        logit_d = self.ECN(feature_emb).mean(dim=1)
        logit_s = self.LCN(feature_emb).mean(dim=1)
        logit = (logit_d + logit_s) * 0.5
        return logit

    def forward_with_latent(self, X):
        """Return (raw_logit, ecn_latent_mean_over_heads)."""
        feature_emb = self.embedding_layer(X)
        ecn_latent = self.ECN.forward_latent(feature_emb)  # B x H x D
        latent = ecn_latent.mean(dim=1)  # B x D
        logit_d = self.ECN(feature_emb).mean(dim=1)
        logit_s = self.LCN(feature_emb).mean(dim=1)
        logit = (logit_d + logit_s) * 0.5
        return logit, latent

    def forward_features(self, X):
        """Return (ECN_latent_flat, LCN_latent_flat) before final linear layers."""
        feature_emb = self.embedding_layer(X)
        ecn_latent = self.ECN.forward_latent(feature_emb)
        lcn_latent = self.LCN.forward_latent(feature_emb)
        return (ecn_latent.reshape(ecn_latent.size(0), -1),
                lcn_latent.reshape(lcn_latent.size(0), -1))

    def _apply_activation(self, logit):
        if self._output_activation is not None:
            return self._output_activation(logit)
        return logit

    def get_model_return_dict(self, X):
        """Return dict with activated y_pred, y_d, y_s."""
        feature_emb = self.embedding_layer(X)
        logits_xld = self.ECN(feature_emb).mean(dim=1)
        logits_xls = self.LCN(feature_emb).mean(dim=1)
        logit = (logits_xld + logits_xls) * 0.5
        return {
            "y_pred": self._apply_activation(logit),
            "y_d": self._apply_activation(logits_xld),
            "y_s": self._apply_activation(logits_xls),
        }

    def has_custom_loss(self):
        return True

    def compute_custom_loss(self, return_dict, y_true, loss_fn):
        """DCNv3 custom loss with adaptive weighting for ECN/LCN."""
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        loss = loss_fn(y_pred, y_true, reduction='mean')
        loss_d = loss_fn(y_d, y_true, reduction='mean')
        loss_s = loss_fn(y_s, y_true, reduction='mean')
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        return loss + loss_d * weight_d + loss_s * weight_s


# =====================================================================
# Backbone Registry & Factory
# =====================================================================

BACKBONE_REGISTRY = {
    "PNN": PNNBackbone,
    "FinalNet": FinalNetBackbone,
    "DCNv3": DCNv3Backbone,
    "FCN": DCNv3Backbone,
}


def build_backbone(backbone_type, feature_map, **kwargs):
    """Build a backbone network by type name.

    Args:
        backbone_type: One of "PNN", "FinalNet", "DCNv3", "FCN"
        feature_map: FuxiCTR FeatureMap object
        **kwargs: Backbone-specific parameters

    Returns:
        backbone (nn.Module) with .forward(X) -> logit,
        .forward_with_latent(X) -> (logit, latent),
        .get_model_return_dict(X) -> dict, etc.
    """
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. "
                         f"Supported: {list(BACKBONE_REGISTRY.keys())}")
    cls = BACKBONE_REGISTRY[backbone_type]
    return cls(feature_map, **kwargs)


# =====================================================================
# Internal components (self-contained to avoid cross-model_zoo imports)
# =====================================================================

class _FeatureGating(nn.Module):
    def __init__(self, num_fields, gate_residual="concat"):
        super(_FeatureGating, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        self.gate_residual = gate_residual

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        if self.gate_residual == "concat":
            return torch.cat([feature_emb, feature_emb * gates], dim=1)
        return feature_emb + feature_emb * gates


class _FactorizedInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        super(_FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            return torch.cat([h2, h1 * h2], dim=-1)
        return h2 + h1 * h2


class _FinalBlock(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None,
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        super(_FinalBlock, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        dims = [input_dim] + hidden_units
        for idx in range(len(hidden_units)):
            self.layer.append(_FactorizedInteraction(dims[idx], dims[idx + 1],
                                                     residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(dims[idx + 1]))
            if idx < len(dropout_rates) and dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            else:
                self.dropout.append(None)
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        x = X
        for i in range(len(self.layer)):
            x = self.layer[i](x)
            if i < len(self.norm):
                x = self.norm[i](x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if self.dropout[i] is not None:
                x = self.dropout[i](x)
        return x


class _MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2, sequence_pooling="sum"):
        super(_MultiHeadFeatureEmbedding, self).__init__()
        self.feature_map = feature_map
        self.target_embedding_dim = int(embedding_dim)
        self.num_heads = num_heads
        self.sequence_pooling = str(sequence_pooling or "sum").lower()
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.feature_projections = nn.ModuleDict()
        for feature, feature_spec in self.feature_map.features.items():
            feature_dim = self._infer_feature_embedding_dim(feature_spec, self.target_embedding_dim)
            if feature_dim != self.target_embedding_dim:
                self.feature_projections[feature] = nn.Linear(feature_dim, self.target_embedding_dim, bias=False)

    @staticmethod
    def _infer_feature_embedding_dim(feature_spec, default_dim):
        try:
            return int(feature_spec.get("embedding_dim", default_dim) or default_dim)
        except (TypeError, ValueError):
            return int(default_dim)

    def _align_feature_embedding(self, feature, embedding):
        if embedding.size(-1) == self.target_embedding_dim:
            return embedding
        if feature in self.feature_projections:
            return self.feature_projections[feature](embedding)
        raise RuntimeError(
            f"DCNv3 feature '{feature}' produced embedding_dim={embedding.size(-1)}, "
            f"expected {self.target_embedding_dim}. Add a feature projection or align feature_map embedding_dim."
        )

    def forward(self, X):
        feature_emb_dict = self.embedding_layer.embedding_layer(X)
        feature_emb_list = []
        for feature, feature_spec in self.feature_map.features.items():
            if feature not in feature_emb_dict:
                continue
            embedding = feature_emb_dict[feature]
            if embedding.dim() == 3:
                padding_idx = int(feature_spec.get("padding_idx", 0) or 0)
                feature_input = X.get(feature)
                if feature_input is not None:
                    mask = (feature_input.long() != padding_idx).float().unsqueeze(-1)
                    if self.sequence_pooling == "mean":
                        embedding = (embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    elif self.sequence_pooling == "max":
                        embedding = embedding.masked_fill(mask == 0, float("-inf"))
                        embedding = embedding.max(dim=1).values
                        embedding = torch.where(torch.isfinite(embedding), embedding, torch.zeros_like(embedding))
                    else:
                        embedding = (embedding * mask).sum(dim=1)
                else:
                    embedding = embedding.sum(dim=1)
            embedding = self._align_feature_embedding(feature, embedding)
            feature_emb_list.append(embedding)
        feature_emb = torch.stack(feature_emb_list, dim=1)
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
            multihead_feature_emb, 2, dim=-1)
        multihead_feature_emb1 = multihead_feature_emb1.flatten(start_dim=2)
        multihead_feature_emb2 = multihead_feature_emb2.flatten(start_dim=2)
        return torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)


class _ExponentialCrossNetwork(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True,
                 batch_norm=False, net_dropout=0.1, num_heads=1):
        super(_ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.dfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.dfc(x)

    def forward_latent(self, x):
        """Return features before final linear layer."""
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return x


class _LinearCrossNetwork(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True,
                 batch_norm=True, net_dropout=0.1, num_heads=1):
        super(_LinearCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.sfc(x)

    def forward_latent(self, x):
        """Return features before final linear layer."""
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return x
