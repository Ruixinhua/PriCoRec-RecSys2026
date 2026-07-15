# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import ast
import sys
import os
import logging
import numpy as np
import torch
from torch import nn
import random
from functools import partial
import re


_LOSS_ALIASES = {
    "bce": "binary_cross_entropy",
    "binary_crossentropy": "binary_cross_entropy",
    "binary_cross_entropy": "binary_cross_entropy",
    "bce_with_logits": "binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits": "binary_cross_entropy_with_logits",
    "mse": "mse_loss",
    "mean_squared_error": "mse_loss",
    "mae": "l1_loss",
    "mean_absolute_error": "l1_loss",
}
_LOSS_REGISTRY = {
    "binary_cross_entropy": torch.nn.functional.binary_cross_entropy,
    "binary_cross_entropy_with_logits": torch.nn.functional.binary_cross_entropy_with_logits,
    "cross_entropy": torch.nn.functional.cross_entropy,
    "nll_loss": torch.nn.functional.nll_loss,
    "mse_loss": torch.nn.functional.mse_loss,
    "l1_loss": torch.nn.functional.l1_loss,
    "smooth_l1_loss": torch.nn.functional.smooth_l1_loss,
    "huber_loss": torch.nn.functional.huber_loss,
}
_INITIALIZER_REGISTRY = {
    "normal_": nn.init.normal_,
    "uniform_": nn.init.uniform_,
    "zeros_": nn.init.zeros_,
    "ones_": nn.init.ones_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "orthogonal_": nn.init.orthogonal_,
}


def _qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _resolve_initializer_function(node):
    name = _qualified_name(node)
    if not name:
        raise ValueError("initializer must name an approved torch.nn.init function")
    short_name = name.rsplit(".", 1)[-1]
    allowed_prefixes = ("", "nn.init.", "torch.nn.init.")
    if not any(name == prefix + short_name for prefix in allowed_prefixes):
        raise ValueError(f"initializer={name!r} is not supported")
    try:
        return _INITIALIZER_REGISTRY[short_name]
    except KeyError as exc:
        raise ValueError(f"initializer={name!r} is not supported") from exc


def _safe_initializer_literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValueError("initializer arguments must be literals") from exc


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(gpu=-1):
    requested_gpu = int(gpu)
    torch_cuda_version = getattr(torch.version, "cuda", None)
    cuda_available = torch.cuda.is_available()
    if requested_gpu >= 0 and cuda_available:
        device_count = torch.cuda.device_count()
        if requested_gpu >= device_count:
            logging.warning(
                "PyTorch requested gpu=%d but only %d CUDA device(s) are visible; using CPU. "
                "torch.version.cuda=%s",
                requested_gpu,
                device_count,
                torch_cuda_version,
            )
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(requested_gpu))
            try:
                device_name = torch.cuda.get_device_name(device)
            except Exception:
                device_name = "unknown"
            logging.info(
                "PyTorch device selected: requested_gpu=%d, actual_device=%s, cuda_available=True, "
                "torch.version.cuda=%s, device_name=%s",
                requested_gpu,
                device,
                torch_cuda_version,
                device_name,
            )
    else:
        device = torch.device("cpu")
        if requested_gpu >= 0:
            logging.warning(
                "PyTorch requested gpu=%d but CUDA is unavailable; using CPU. "
                "torch.version.cuda=%s. Check NVIDIA driver compatibility with the installed torch wheel.",
                requested_gpu,
                torch_cuda_version,
            )
        else:
            logging.info(
                "PyTorch device selected: requested_gpu=%d, actual_device=cpu, torch.version.cuda=%s",
                requested_gpu,
                torch_cuda_version,
            )
    return device

def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

def get_loss(loss):
    if callable(loss):
        return loss
    if not isinstance(loss, str):
        raise NotImplementedError("loss={} is not supported.".format(loss))
    loss_name = _LOSS_ALIASES.get(loss.strip().lower(), loss.strip().lower())
    try:
        return _LOSS_REGISTRY[loss_name]
    except KeyError as exc:
        raise NotImplementedError("loss={} is not supported.".format(loss)) from exc

def get_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "dice":
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if callable(initializer):
        return initializer
    if not isinstance(initializer, str):
        raise ValueError("initializer={} is not supported.".format(initializer))
    try:
        expression = ast.parse(initializer.strip(), mode="eval").body
        if isinstance(expression, ast.Call):
            if _qualified_name(expression.func) not in {"partial", "functools.partial"}:
                raise ValueError("initializer calls must use functools.partial")
            if not expression.args:
                raise ValueError("partial initializer requires a target function")
            target = _resolve_initializer_function(expression.args[0])
            args = [_safe_initializer_literal(arg) for arg in expression.args[1:]]
            kwargs = {}
            for keyword in expression.keywords:
                if keyword.arg is None:
                    raise ValueError("initializer does not support **kwargs")
                kwargs[keyword.arg] = _safe_initializer_literal(keyword.value)
            return partial(target, *args, **kwargs)
        return _resolve_initializer_function(expression)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("initializer={} is not supported.".format(initializer)) from exc


class FeatureSeparator:
    """
    特征分离器：为非个性化模型创建mask后的特征

    新逻辑：
    - 个性化塔：使用全部特征
    - 非个性化塔：使用全部特征，但对个性化数据的个性化特征进行mask
    """

    def __init__(self, personalization_feature_list=None, feature_map=None):
        """
        初始化特征分离器

        Args:
            personalization_feature_list: 个性化特征列表
            feature_map: 特征映射，用于获取特征类型信息
        """
        self.personalization_feature_list = personalization_feature_list or []
        self.feature_map = feature_map
        self.mask_values = {feature: self._get_mask_value(feature)
                            for feature in personalization_feature_list if feature in feature_map.features}

    def _get_mask_value(self, feature_name):
        """
        获取特征的mask值

        Args:
            feature_name: 特征名称

        Returns:
            mask_value: 用于mask的值
        """
        if self.feature_map and feature_name in self.feature_map.features:
            feature_spec = self.feature_map.features[feature_name]
            feature_type = feature_spec.get("type", "categorical")

            if feature_type == "categorical":
                # 分类特征使用padding_idx作为mask值
                return feature_spec.get("padding_idx", 0)
            elif feature_type == "sequence":
                # 序列特征使用padding_idx作为mask值
                return feature_spec.get("padding_idx", 0)
            elif feature_type == "numeric":
                # 数值特征使用0作为mask值
                return 0.0
            else:
                # 其他类型默认使用0
                return 0
        else:
            # 如果没有特征映射信息，默认使用0
            return 0

    def separate_features(self, inputs, personalized_mask=None):
        """
        分离输入特征，为非个性化模型创建mask后的特征

        Args:
            inputs: 输入特征字典
            personalized_mask: 个性化用户掩码 [batch_size]

        Returns:
            tuple: (personalized_features, non_personalized_features)
                - personalized_features: 包含所有特征（给个性化塔）
                - non_personalized_features: 全特征，但个性化数据的个性化特征被mask（给非个性化塔）
        """
        # 个性化塔使用全部特征
        personalized_features = inputs.copy()

        # 非个性化塔使用全特征，但需要mask个性化数据的个性化特征
        non_personalized_features = {}

        for field_name, field_data in inputs.items():
            if field_name in self.personalization_feature_list:
                # 对于个性化特征，需要mask掉个性化用户的数据
                masked_data = field_data.clone()

                if torch.sum(personalized_mask) > 0:  # 如果有个性化用户，只对个性化用户进行操作
                    mask_value = self.mask_values[field_name]

                    # 将个性化用户的个性化特征设置为mask值
                    if field_data.dtype in [torch.long, torch.int32, torch.int64]:
                        # 整数类型特征
                        masked_data[personalized_mask] = int(mask_value)
                    else:
                        # 浮点类型特征
                        masked_data[personalized_mask] = float(mask_value)

                non_personalized_features[field_name] = masked_data
            else:
                # 非个性化特征保持不变
                non_personalized_features[field_name] = field_data

        return personalized_features, non_personalized_features
