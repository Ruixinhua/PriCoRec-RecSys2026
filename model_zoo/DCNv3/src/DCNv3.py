# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024 salmon@github
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

import torch
import logging
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MultiTowerModule


class DCNv3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DCNv3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 sequence_pooling="sum",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_domain_aware_structure=False,
                 **kwargs):
        super(DCNv3, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.hparams = kwargs
        self.use_domain_aware_structure = use_domain_aware_structure
        self.num_heads = num_heads

        self.embedding_layer = MultiHeadFeatureEmbedding(
            feature_map,
            embedding_dim * num_heads,
            num_heads,
            sequence_pooling=sequence_pooling,
        )

        cross_input_dim = self.num_fields * embedding_dim

        self.ECN = ExponentialCrossNetwork(input_dim=cross_input_dim,
                                           num_cross_layers=num_deep_cross_layers,
                                           net_dropout=deep_net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads,
                                           output_intermediate_features=self.use_domain_aware_structure)
        self.LCN = LinearCrossNetwork(input_dim=cross_input_dim,
                                      num_cross_layers=num_shallow_cross_layers,
                                      net_dropout=shallow_net_dropout,
                                      layer_norm=layer_norm,
                                      batch_norm=batch_norm,
                                      num_heads=num_heads,
                                      output_intermediate_features=self.use_domain_aware_structure)
        self.logits_xld = None
        self.logits_xls = None

        if self.use_domain_aware_structure:
            tower_input_dim = num_heads * cross_input_dim
            self._init_domain_aware_structure_params_pytorch(tower_input_dim)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def _init_domain_aware_structure_params_pytorch(self, tower_input_dim):
        # 使用 MultiTowerModule 替换原有的多塔实现
        self.multi_tower_module = MultiTowerModule(
            input_dim=tower_input_dim,
            tower_hidden_units_list=self.hparams.get("tower_hidden_units_list"),
            tower_activation=self.hparams.get("tower_activation", "ReLU"),
            tower_l2_reg_list=self.hparams.get("tower_l2_reg_list"),
            tower_dropout_list=self.hparams.get("tower_dropout_list"),
            use_bn_tower=self.hparams.get("use_bn_tower", True),
            scene_name=self.hparams.get("scene_name", "scene_id"),
            scene_num_shift=self.hparams.get("scene_num_shift", 1),
            use_scene_id_mapping=self.hparams.get("use_scene_id_mapping", False),
            mapping_feature_name=self.hparams.get("mapping_feature_name"),
            mapping_feature_type=self.hparams.get("mapping_feature_type"),
            feature2id_dict=self.hparams.get("feature2id_dict"),
            default_value=self.hparams.get("default_value"),
            feature_map_dict=self.hparams.get("feature_map_dict")
        )

        # 保存场景数量信息，用于其他方法
        self.scene_num = self.multi_tower_module.scene_num

        logging.info(f"Domain-aware structure initialized with {self.scene_num} towers using MultiTowerModule for PyTorch DCNv3.")


    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        if self.use_domain_aware_structure:
            xld_intermediate = self.ECN(feature_emb)
            xls_intermediate = self.LCN(feature_emb)

            xld_flat = xld_intermediate.view(xld_intermediate.size(0), -1)
            xls_flat = xls_intermediate.view(xls_intermediate.size(0), -1)

            self.logits_xld = self._generate_domain_aware_logits_pytorch(X, xld_flat)
            self.logits_xls = self._generate_domain_aware_logits_pytorch(X, xls_flat)
        else:
            self.logits_xld = self.ECN(feature_emb).mean(dim=1)
            self.logits_xls = self.LCN(feature_emb).mean(dim=1)

        logit = (self.logits_xld + self.logits_xls) * 0.5

        # 应用输出激活函数
        y_pred = self.output_activation(logit)
        y_d = self.output_activation(self.logits_xld)
        y_s = self.output_activation(self.logits_xls)

        # 轻微的数值稳定化，避免过度限制模型表达能力
        # 只处理极端的数值问题，不影响正常的梯度流
        # eps = 1e-6
        # y_pred = torch.clamp(y_pred, min=eps, max=1.0-eps)
        # y_d = torch.clamp(y_d, min=eps, max=1.0-eps)
        # y_s = torch.clamp(y_s, min=eps, max=1.0-eps)

        return_dict = {"y_pred": y_pred,
                       "y_d": y_d,
                       "y_s": y_s,
                       "logit": logit}
        return return_dict

    def _generate_domain_aware_logits_pytorch(self, X_features, net_output):
        # 使用 MultiTowerModule 处理多塔逻辑
        final_logits = self.multi_tower_module(net_output, X_features)
        return final_logits

    def add_loss(self, return_dict, y_true):
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]

        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss_d = self.loss_fn(y_d, y_true, reduction='mean')
        loss_s = self.loss_fn(y_s, y_true, reduction='mean')

        weight_d = loss_d - loss
        weight_s = loss_s - loss

        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))

        total_loss = loss + loss_d * weight_d + loss_s * weight_s
        return total_loss


class MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2, sequence_pooling="sum"):
        super(MultiHeadFeatureEmbedding, self).__init__()
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
        if self.feature_projections:
            preview = ", ".join(
                f"{name}:{module.in_features}->{module.out_features}"
                for name, module in list(self.feature_projections.items())[:10]
            )
            logging.info(
                "DCNv3 MultiHeadFeatureEmbedding added %d feature projections to align embedding_dim=%d: %s",
                len(self.feature_projections),
                self.target_embedding_dim,
                preview,
            )

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

    def forward(self, X):  # H = num_heads
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
        feature_emb = torch.stack(feature_emb_list, dim=1)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        return multihead_feature_emb  # B × H × FD/H


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=False,
                 net_dropout=0.1,
                 num_heads=1,
                 output_intermediate_features=False):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.output_intermediate_features = output_intermediate_features
        self.intermediate_output_dim = input_dim

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

        if self.output_intermediate_features:
            return x

        logit = self.dfc(x)
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1,
                 num_heads=1,
                 output_intermediate_features=False):
        super(LinearCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.output_intermediate_features = output_intermediate_features
        self.intermediate_output_dim = input_dim

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

        if self.output_intermediate_features:
            return x

        logit = self.sfc(x)
        return logit
