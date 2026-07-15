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

import torch
import logging
from torch import nn
import torch.nn.functional as F
from .mlp_block import MLP_Block


class MultiTowerModule(nn.Module):
    """
    多塔模块，支持域感知结构的多塔网络

    Args:
        input_dim: 输入特征维度
        tower_hidden_units_list: 每个塔的隐藏层单元数列表，例如 [[128, 64], [128, 64]]
        tower_activation: 塔内激活函数，默认为 'ReLU'
        tower_l2_reg_list: 每个塔的L2正则化系数列表，默认为 [0.0] * scene_num
        tower_dropout_list: 每个塔的dropout率列表，默认为 [0.0] * scene_num
        use_bn_tower: 是否在塔内使用批归一化，默认为 True
        scene_name: 场景特征名称，默认为 'scene_id'
        scene_num_shift: 场景ID的偏移量，默认为 1（1-based转0-based）
        use_scene_id_mapping: 是否使用场景ID映射，默认为 False
        mapping_feature_name: 用于映射的特征名称
        mapping_feature_type: 映射特征类型 ('sparse' 或 'dense')
        feature2id_dict: 特征值到场景ID的映射字典
        default_value: 默认场景ID值（1-based）
        feature_map_dict: 稀疏特征的映射字典
    """

    def __init__(self,
                 input_dim,
                 tower_hidden_units_list,
                 tower_activation='ReLU',
                 tower_l2_reg_list=None,
                 tower_dropout_list=None,
                 use_bn_tower=True,
                 scene_name='scene_id',
                 scene_num_shift=1,
                 use_scene_id_mapping=False,
                 mapping_feature_name=None,
                 mapping_feature_type=None,
                 feature2id_dict=None,
                 default_value=None,
                 feature_map_dict=None,
                 **kwargs):
        super(MultiTowerModule, self).__init__()

        # 验证基本参数
        if not tower_hidden_units_list or len(tower_hidden_units_list) == 0:
            raise ValueError("`tower_hidden_units_list` cannot be empty.")

        self.input_dim = input_dim
        self.tower_hidden_units_list = tower_hidden_units_list
        self.scene_num = len(tower_hidden_units_list)
        self.tower_activation = tower_activation
        self.tower_l2_reg_list = tower_l2_reg_list or [0.0] * self.scene_num
        self.tower_dropout_list = tower_dropout_list or [0.0] * self.scene_num
        self.use_bn_tower = use_bn_tower
        self.output_activation = kwargs.get("output_activation", None)

        # 场景ID相关参数
        self.scene_name = scene_name
        self.scene_num_shift = scene_num_shift
        self.use_scene_id_mapping = use_scene_id_mapping

        # 场景ID映射相关参数
        if self.use_scene_id_mapping:
            self.mapping_feature_name = mapping_feature_name
            self.mapping_feature_type = mapping_feature_type
            self.feature2id_dict = feature2id_dict
            self.default_value = default_value
            self.feature_map_dict = feature_map_dict

            # 验证映射参数
            if not self.mapping_feature_name:
                raise ValueError("`mapping_feature_name` required for scene_id mapping.")
            if not self.feature2id_dict:
                raise ValueError("`feature2id_dict` required for scene_id mapping.")
            if self.default_value is None or not (1 <= self.default_value <= self.scene_num):
                raise ValueError(f"`default_value` ({self.default_value}) must be a valid 1-based scene_id (1 to {self.scene_num}).")
            if self.mapping_feature_type == "sparse" and not self.feature_map_dict:
                raise ValueError("`feature_map_dict` required for sparse scene_id mapping.")

        # 创建多塔网络
        self._build_towers()

        logging.info(f"MultiTowerModule initialized with {self.scene_num} towers.")

    def _build_towers(self):
        """构建多塔网络结构"""
        self.tower_dnns = nn.ModuleList()
        self.tower_output_layers = nn.ModuleList()

        for i in range(self.scene_num):
            # 使用 MLP_Block 构建每个塔
            if self.tower_hidden_units_list[i]:  # 如果有隐藏层
                tower_dnn = MLP_Block(
                    input_dim=self.input_dim,
                    hidden_units=self.tower_hidden_units_list[i],
                    hidden_activations=self.tower_activation,
                    dropout_rates=self.tower_dropout_list[i],
                    batch_norm=self.use_bn_tower,
                    use_bias=True
                )
                dnn_output_dim = self.tower_hidden_units_list[i][-1]
            else:  # 没有隐藏层，直接连接
                tower_dnn = nn.Identity()
                dnn_output_dim = self.input_dim

            self.tower_dnns.append(tower_dnn)
            self.tower_output_layers.append(nn.Linear(dnn_output_dim, 1))

    def forward(self, net_output, X_features=None):
        """
        前向传播

        Args:
            net_output: 网络输出特征 (batch_size, input_dim)
            X_features: 原始输入特征字典，用于场景ID映射

        Returns:
            final_logits: 最终的 logits (batch_size, 1)
        """
        # 获取场景ID
        scene_id_0_indexed = self._scene_id_mapping(X_features)  # checked

        # 计算每个塔的输出
        tower_logits_list = []
        for i in range(self.scene_num):
            tower_dnn_out = self.tower_dnns[i](net_output)
            tower_logit = self.tower_output_layers[i](tower_dnn_out)
            tower_logits_list.append(tower_logit)

        # 合并所有塔的输出
        scene_tower_output_concat = torch.cat(tower_logits_list, dim=1)

        # 路由到对应的塔
        final_logits = self._logits_routing(scene_tower_output_concat, scene_id_0_indexed)

        return final_logits

    def _scene_id_mapping(self, X_features):
        """场景ID映射"""
        if X_features is None:
            raise ValueError("X_features is required for scene_id mapping.")

        if self.use_scene_id_mapping:
            # 使用自定义映射
            feature_values = X_features.get(self.mapping_feature_name)
            if feature_values is None:
                raise ValueError(f"Mapping feature '{self.mapping_feature_name}' not found in input features.")

            if feature_values.ndim > 1:
                feature_values = feature_values.squeeze(-1)

            # 初始化为默认值
            default_scene_id_0_indexed = torch.tensor(
                self.default_value - self.scene_num_shift,
                device=feature_values.device,
                dtype=torch.long
            )
            scene_ids = torch.full_like(feature_values, default_scene_id_0_indexed, dtype=torch.long)

            # 应用映射
            for feat_val_str, scene_id_1_based in self.feature2id_dict.items():
                target_scene_id_0_indexed = torch.tensor(
                    scene_id_1_based - self.scene_num_shift,
                    device=feature_values.device,
                    dtype=torch.long
                )

                # 获取比较值
                if self.mapping_feature_type == 'sparse':
                    mapped_int_val = self.feature_map_dict.get(feat_val_str)
                    if mapped_int_val is None:
                        logging.warning(f"Feature value '{feat_val_str}' not in feature_map_dict. Skipping.")
                        continue
                    compare_value = torch.tensor(mapped_int_val, device=feature_values.device, dtype=feature_values.dtype)
                else:
                    try:
                        if feature_values.dtype != torch.float32:
                            compare_value = torch.tensor(int(feat_val_str), device=feature_values.device, dtype=feature_values.dtype)
                        else:
                            compare_value = torch.tensor(float(feat_val_str), device=feature_values.device, dtype=feature_values.dtype)
                    except ValueError:
                        logging.warning(f"Cannot convert feature value '{feat_val_str}' to numeric type. Skipping.")
                        continue

                scene_ids = torch.where(feature_values == compare_value, target_scene_id_0_indexed, scene_ids)

            return scene_ids
        else:
            # 直接使用场景ID
            scene_id_tensor = X_features.get(self.scene_name)
            if scene_id_tensor is None:
                scene_id_tensor = X_features.get('scene_id')
                if scene_id_tensor is None:
                    raise ValueError(f"Scene feature '{self.scene_name}' (and fallback 'scene_id') not found in input features.")

            if scene_id_tensor.ndim > 1:
                scene_id_tensor = scene_id_tensor.squeeze(-1)

            scene_id_tensor = scene_id_tensor.long()
            return scene_id_tensor - self.scene_num_shift

    def _logits_routing(self, scene_tower_output_concat, scene_id_0_indexed):
        """Logits路由"""
        num_towers = scene_tower_output_concat.size(1)
        scene_select = F.one_hot(scene_id_0_indexed, num_classes=num_towers)
        scene_select = scene_select.to(scene_tower_output_concat.dtype)

        final_logits = torch.sum(scene_tower_output_concat * scene_select, dim=-1, keepdim=True)
        return final_logits

    def get_tower_outputs(self, net_output):
        """
        获取所有塔的输出（用于调试或分析）

        Args:
            net_output: 网络输出特征

        Returns:
            tower_outputs: 所有塔的输出列表
        """
        tower_outputs = []
        for i in range(self.scene_num):
            tower_dnn_out = self.tower_dnns[i](net_output)
            tower_logit = self.tower_output_layers[i](tower_dnn_out)
            tower_outputs.append(tower_logit)
        return tower_outputs
