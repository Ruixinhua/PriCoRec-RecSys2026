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


import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import numpy as np
from collections import OrderedDict
from .pretrained_embedding import PretrainedEmbedding
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.utils import not_in_whitelist
from fuxictr.pytorch import layers


# Feature maps are data artifacts and must not be able to execute arbitrary
# expressions.  These are the only encoder expressions emitted by the project
# configuration generators; aliases keep existing hand-written feature maps
# compatible without accepting parameters or arbitrary modules.
_FEATURE_ENCODER_FACTORIES = {
    "layers.MaskedAveragePooling()": layers.MaskedAveragePooling,
    "MaskedAveragePooling()": layers.MaskedAveragePooling,
    "MaskedAveragePooling": layers.MaskedAveragePooling,
    "layers.MaskedSumPooling()": layers.MaskedSumPooling,
    "MaskedSumPooling()": layers.MaskedSumPooling,
    "MaskedSumPooling": layers.MaskedSumPooling,
}


def _build_allowed_feature_encoder(encoder):
    if isinstance(encoder, nn.Module):
        # Programmatic callers may supply an already-built module. Serialized
        # feature-map data can only contain the string forms handled below.
        return encoder
    if not isinstance(encoder, str):
        raise ValueError(
            "feature_encoder must be an allowed encoder name or an nn.Module; "
            f"got {type(encoder).__name__}."
        )
    factory = _FEATURE_ENCODER_FACTORIES.get(encoder.strip())
    if factory is None:
        raise ValueError(
            f"feature_encoder={encoder!r} is not supported. "
            f"Allowed values: {sorted(_FEATURE_ENCODER_FACTORIES)}."
        )
    return factory()


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int_list(value):
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    parsed = []
    for item in values:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


class AutoDisNumericEmbedding(nn.Module):
    """AutoDis-style encoder for dense numeric features.

    The layer maps each scalar through learnable meta-embeddings while keeping
    the FuxiCTR embedding interface unchanged. Each input becomes one embedding
    vector with shape [batch_size, embedding_dim].
    """

    def __init__(self, embedding_dim, bins=80, temp=0.02, hidden_units=None,
                 use_log_transform=False, dropout=0.0):
        super(AutoDisNumericEmbedding, self).__init__()
        self.embedding_dim = int(embedding_dim)
        self.bins = max(1, _as_int(bins, 80))
        self.temp = max(_as_float(temp, 0.02), 1e-6)
        self.use_log_transform = _as_bool(use_log_transform, False)

        layers_list = []
        input_dim = 1
        for hidden_dim in _as_int_list(hidden_units):
            if hidden_dim <= 0:
                continue
            layers_list.append(nn.Linear(input_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            if _as_float(dropout, 0.0) > 0:
                layers_list.append(nn.Dropout(_as_float(dropout, 0.0)))
            input_dim = hidden_dim
        layers_list.append(nn.Linear(input_dim, self.bins))
        self.logit_net = nn.Sequential(*layers_list)
        self.meta_embeddings = nn.Parameter(torch.empty(self.bins, self.embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.logit_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.meta_embeddings, mean=0.0, std=1e-4)

    def forward(self, x):
        x = x.float().view(-1, 1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_log_transform:
            x = torch.sign(x) * torch.log1p(torch.abs(x))
        logits = self.logit_net(x)
        weights = F.softmax(logits / self.temp, dim=-1)
        return torch.matmul(weights, self.meta_embeddings)


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map,
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing)

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb


class FeatureEmbeddingDict(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = get_initializer(embedding_initializer)
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        self._warned_invalid_id_features = set()
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_dim = 1 # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])
                    else:
                        if feature_spec["type"] == "embedding": # add embedding projection
                            pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                            self.feature_encoders[feature] = nn.Linear(pretrain_dim, feat_dim, bias=False)

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    dense_encoder = str(feature_spec.get("dense_encoder", "") or "").lower()
                    autodis_conf = feature_spec.get("autodis") or feature_spec.get("autodis_params") or {}
                    if dense_encoder == "autodis" or _as_bool(autodis_conf.get("enabled"), False):
                        self.embedding_layers[feature] = AutoDisNumericEmbedding(
                            feat_dim,
                            bins=autodis_conf.get("bins", 80),
                            temp=autodis_conf.get("temp", 0.02),
                            hidden_units=autodis_conf.get("hidden_units"),
                            use_log_transform=autodis_conf.get("use_log_transform", False),
                            dropout=autodis_conf.get("dropout", 0.0),
                        )
                    else:
                        self.embedding_layers[feature] = nn.Linear(1, feat_dim, bias=False)
                elif feature_spec["type"] in ["categorical", "sequence"]:
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        pretrain_path = os.path.join(feature_map.data_dir,
                                                     feature_spec["pretrained_emb"])
                        vocab_path = os.path.join(feature_map.data_dir,
                                                  "feature_vocab.json")
                        pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                        pretrain_usage = feature_spec.get("pretrain_usage", "init")
                        self.embedding_layers[feature] = PretrainedEmbedding(feature,
                                                                             feature_spec,
                                                                             pretrain_path,
                                                                             vocab_path,
                                                                             feat_dim,
                                                                             pretrain_dim,
                                                                             pretrain_usage,
                                                                             embedding_initializer)
                    else:
                        padding_idx = feature_spec.get("padding_idx", None)
                        self.embedding_layers[feature] = nn.Embedding(feature_spec["vocab_size"],
                                                                      feat_dim,
                                                                      padding_idx=padding_idx)
                elif feature_spec["type"] == "embedding":
                    self.embedding_layers[feature] = nn.Identity()
        self.init_weights()

    def get_feature_encoder(self, encoder):
        if isinstance(encoder, list):
            return nn.Sequential(*[_build_allowed_feature_encoder(enc) for enc in encoder])
        return _build_allowed_feature_encoder(encoder)

    def init_weights(self):
        for k, v in self.embedding_layers.items():
            if "share_embedding" in self._feature_map.features[k]:
                continue
            if type(v) == PretrainedEmbedding: # skip pretrained
                v.init_weights()
            elif type(v) == nn.Embedding:
                if v.padding_idx is not None:
                    self.embedding_initializer(v.weight[1:, :]) # set padding_idx to zero
                else:
                    self.embedding_initializer(v.weight)

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def dict2tensor(self, embedding_dict, flatten_emb=False, feature_list=[], feature_source=[],
                    feature_type=[]):
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_list and not_in_whitelist(feature, feature_list):
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        feature_emb_dict = OrderedDict()
        for feature in inputs.keys():
            feature_spec = self._feature_map.features[feature]
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    inp = self._sanitize_embedding_indices(feature, inp)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    inp = self._sanitize_embedding_indices(feature, inp)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "embedding":
                    inp = inputs[feature].float()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict

    def _sanitize_embedding_indices(self, feature, inp):
        if feature not in self.embedding_layers:
            return inp
        embedding_layer = self.embedding_layers[feature]
        if not isinstance(embedding_layer, nn.Embedding) or inp.numel() == 0:
            return inp

        min_id = int(torch.min(inp).detach().item())
        max_id = int(torch.max(inp).detach().item())
        sanitized = inp
        if min_id < 0:
            padding_idx = embedding_layer.padding_idx
            if padding_idx is None or padding_idx < 0:
                padding_idx = 0
            sanitized = inp.clone()
            sanitized[sanitized < 0] = int(padding_idx)
            self._warn_invalid_ids_once(
                feature,
                f"FeatureEmbedding input for '{feature}' contains negative ids "
                f"(min_id={min_id}); mapping them to padding index {padding_idx}.",
            )
            max_id = int(torch.max(sanitized).detach().item())

        vocab_size = int(embedding_layer.num_embeddings)
        if max_id >= vocab_size:
            raise ValueError(
                f"FeatureEmbedding input id exceeds vocab_size for feature '{feature}': "
                f"min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}, "
                f"input_shape={tuple(inp.shape)}. Check TFRecord/item-pool feature values "
                "against feature_map vocab_size."
            )
        return sanitized

    def _warn_invalid_ids_once(self, feature, message):
        if feature in self._warned_invalid_id_features:
            return
        self._warned_invalid_id_features.add(feature)
        logging.warning(message)
