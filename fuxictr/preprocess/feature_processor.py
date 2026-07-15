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


import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import polars as pl
import os
import logging
import json
import re
import shutil
import glob
from pathlib import Path
import sklearn.preprocessing as sklearn_preprocess
from fuxictr.features import FeatureMap
from .tokenizer import Tokenizer
from .normalizer import Normalizer
import psutil  # 添加内存监控
import gc  # 添加垃圾回收


# Dataset configuration is an input boundary.  Keep the accepted dtype syntax
# deliberately small instead of evaluating arbitrary Python from YAML.
_DTYPE_ALLOWLIST = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "bool": bool,
    "boolean": bool,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "np.int8": np.int8,
    "np.int16": np.int16,
    "np.int32": np.int32,
    "np.int64": np.int64,
    "np.uint8": np.uint8,
    "np.uint16": np.uint16,
    "np.uint32": np.uint32,
    "np.uint64": np.uint64,
    "np.float16": np.float16,
    "np.float32": np.float32,
    "np.float64": np.float64,
}
_ALLOWED_DTYPE_OBJECTS = frozenset(_DTYPE_ALLOWLIST.values())


def _resolve_dtype(dtype, feature_name):
    """Resolve an explicitly supported dtype without executing configuration."""
    if isinstance(dtype, str):
        resolved = _DTYPE_ALLOWLIST.get(dtype.strip().lower())
        if resolved is not None:
            return resolved
    else:
        try:
            if dtype in _ALLOWED_DTYPE_OBJECTS:
                return dtype
        except TypeError:
            # Unhashable config values are invalid dtypes as well, but should
            # still receive the same useful validation error below.
            pass
    raise ValueError(
        f"Unsupported dtype for feature {feature_name!r}: {dtype!r}. "
        f"Expected one of {sorted(_DTYPE_ALLOWLIST)} or its corresponding Python/numpy type."
    )


class FeatureProcessor(object):
    def __init__(self,
                 feature_cols=[],
                 label_col=[],
                 dataset_id=None,
                 data_root="../data/",
                 **kwargs):
        logging.info("Set up feature processor...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_processor.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.vocab_file = os.path.join(self.data_dir, "feature_vocab.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_cols = label_col if type(label_col) == list else [label_col]
        self.feature_map = FeatureMap(dataset_id, self.data_dir)
        self.feature_map.labels = [col["name"] for col in self.label_cols]
        self.feature_map.group_id = kwargs.get("group_id", None)
        self.feature_map.feature_group_id = kwargs.get("feature_group_id", None)
        self.dtype_dict = {
            feat["name"]: _resolve_dtype(feat["dtype"], feat["name"])
            for feat in self.feature_cols + self.label_cols
        }
        self.processor_dict = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_data(self, data_path, data_format="csv", sep=",", n_rows=None, **kwargs):
        if not data_path.endswith(data_format):
            data_path = os.path.join(data_path, f"*.{data_format}")
        logging.info("Reading files: " + data_path)
        file_names = sorted(glob.glob(data_path))
        assert len(file_names) > 0, f"Invalid data path: {data_path}"
        if data_format == "csv":
            dfs = [
                pl.scan_csv(source=file_name, separator=sep, dtypes=self.dtype_dict,
                            low_memory=False, n_rows=n_rows)
                for file_name in file_names
            ]
            ddf = pl.concat(dfs)
        elif data_format == "parquet":
            dfs = [
                pl.scan_parquet(source=file_name, low_memory=False, n_rows=n_rows)
                for file_name in file_names
            ]
            ddf = pl.concat(dfs)
            # Use collect_schema() to avoid PerformanceWarning
            schema = ddf.collect_schema()
            seq_cols = [x for x in schema.names() if isinstance(schema[x], pl.List)]
            for col in seq_cols:
                # Convert list to "^" seperated string for the same preprocessing as csv format
                ddf = ddf.with_columns(pl.col(col).map_elements(lambda x: "^".join(map(str, x)), return_dtype=pl.String))
        else:
            NotImplementedError(f"data_format={data_format} not supported.")
        return ddf

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        # Get schema once to avoid repeated schema resolution
        if hasattr(ddf, 'collect_schema'):  # LazyFrame
            schema = ddf.collect_schema()
            column_names = schema.names()
        else:  # DataFrame
            column_names = ddf.columns

        all_cols = self.label_cols + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            fill_na = col.get("fill_na",
                              "" if col["dtype"] in ["str", str] else 0)
            col_exist = name in column_names
            if col_exist:
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na))
            if col.get("preprocess"):
                preprocess_args = re.split(r"\(|\)", col["preprocess"])
                preprocess_fn = getattr(self, preprocess_args[0])
                if len(preprocess_args) == 1:
                    preprocess_args = [name] # use col_name as args when not being explicitly set
                else:
                    preprocess_args = preprocess_args[1:-1]
                ddf = ddf.with_columns(
                    preprocess_fn(*preprocess_args)
                    .alias(name)
                    .cast(self.dtype_dict[name])
                )
            if not col_exist:
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na))
            selected_dtype = ddf.select(name).collect_schema().dtypes()[0]
            if col.get("type") == "sequence" and isinstance(selected_dtype, pl.List):
                # Convert list to a "^"-separated string for unified preprocessing of parquet and CSV formats.
                ddf = ddf.with_columns(pl.col(name).apply(lambda x: "^".join(map(str, x))))
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.select(active_cols)
        return ddf

    def fit(self, train_ddf, min_categr_count=1, num_buckets=10, rebuild_dataset=True, **kwargs):
        logging.info("Fit feature processor...")
        self.rebuild_dataset = rebuild_dataset

        # 内存监控 - 开始
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        logging.info(f"Available memory at start: {available_gb:.2f} GB ({memory_info.percent:.1f}% used)")

        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                col_series = (
                    train_ddf.select(name).collect().to_series().to_pandas() if self.rebuild_dataset
                    else None
                )

                if col["type"] == "meta": # e.g. set group_id in gAUC
                    self.fit_meta_col(col)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, col_series)
                elif col["type"] == "embedding":
                    self.fit_embedding_col(col)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, col_series,
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, col_series,
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature type={}".format(col["type"]))

                # 手动释放内存
                if col_series is not None:
                    del col_series
                gc.collect()

                # 内存监控 - 每列处理后
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:
                    logging.warning(f"High memory usage after processing {name}: {memory_info.percent:.1f}%")
                    gc.collect()

        # 内存监控 - 结束
        memory_info = psutil.virtual_memory()
        logging.info(f"Memory usage after fit: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.2f} GB used)")

        # Expand vocab from pretrained_emb
        os.makedirs(self.data_dir, exist_ok=True)
        for col in self.feature_cols:
            name = col["name"]
            if "pretrained_emb" in col:
                logging.info("Loading pretrained embedding: " + name)
                if "pretrain_dim" in col:
                    self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]
                ext = Path(col["pretrained_emb"]).suffix
                shutil.copy(col["pretrained_emb"],
                            os.path.join(self.data_dir, "pretrained_{}{}".format(name, ext)))
                self.feature_map.features[name]["pretrained_emb"] = "pretrained_{}{}".format(name, ext)
                self.feature_map.features[name]["freeze_emb"] = col.get("freeze_emb", True)
                self.feature_map.features[name]["pretrain_usage"] = col.get("pretrain_usage", "init")
                tokenizer = self.processor_dict[name + "::tokenizer"]
                tokenizer.load_pretrained_vocab(self.dtype_dict[name], col["pretrained_emb"])
                self.processor_dict[name + "::tokenizer"] = tokenizer
                self.feature_map.features[name]["vocab_size"] = tokenizer.vocab_size()

        # Handle share_embedding vocab re-assign
        for name, spec in self.feature_map.features.items():
            if spec["type"] == "numeric":
                self.feature_map.total_features += 1
            elif spec["type"] in ["categorical", "sequence"]:
                if "share_embedding" in spec:
                    # sync vocab from the shared_embedding field
                    tokenizer = self.processor_dict[name + "::tokenizer"]
                    tokenizer.vocab = self.processor_dict[spec["share_embedding"] + "::tokenizer"].vocab
                    self.processor_dict[name + "::tokenizer"] = tokenizer
                    self.feature_map.features[name].update({"oov_idx": tokenizer.vocab["__OOV__"],
                                                            "vocab_size": tokenizer.vocab_size()})
                else:
                    self.feature_map.total_features += self.feature_map.features[name]["vocab_size"]
                if "pretrained_emb" not in spec: # "oov_idx" not used without pretrained_emb
                    del self.feature_map.features[name]["oov_idx"]

        self.feature_map.num_fields = self.feature_map.get_num_fields()
        self.feature_map.set_column_index()
        self.feature_map.save(self.json_file)
        self.save_pickle(self.pickle_file)
        self.save_vocab(self.vocab_file)
        logging.info("Set feature processor done.")

    def fit_meta_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        self.feature_map.features[name] = {"type": feature_type}
        if col.get("remap", True):
            # No need to fit, update vocab in encode_meta()
            tokenizer = Tokenizer(min_freq=1, remap=True)
            self.processor_dict[name + "::tokenizer"] = tokenizer

    def fit_numeric_col(self, col, col_series):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "normalizer" in col:
            normalizer = Normalizer(col["normalizer"])
            if self.rebuild_dataset:
                normalizer.fit(col_series.dropna().values)
            self.processor_dict[name + "::normalizer"] = normalizer

    def fit_embedding_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "pretrain_dim" in col:
            self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]

    def fit_categorical_col(self, col, col_series, min_categr_count=1, num_buckets=10):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        if "category_processor" not in col:
            tokenizer = Tokenizer(min_freq=min_categr_count,
                                  na_value=col.get("fill_na", ""),
                                  remap=col.get("remap", True))
            if self.rebuild_dataset:
                tokenizer.fit_on_texts(col_series)
            else:
                if "vocab_size" in col:
                    tokenizer.update_vocab(range(col["vocab_size"] - 1))
                else:
                    raise ValueError(f"{name}: vocab_size is required when rebuild_dataset=False")
            if "share_embedding" in col:
                self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
                tknzr_name = col["share_embedding"] + "::tokenizer"
                # update vocab of both tokenizers
                self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
                self.feature_map.features[col["share_embedding"]] \
                                .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                         "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
            self.processor_dict[name + "::tokenizer"] = tokenizer
            self.feature_map.features[name].update({"padding_idx": 0,
                                                    "oov_idx": tokenizer.vocab["__OOV__"],
                                                    "vocab_size": tokenizer.vocab_size()})
        else:
            category_processor = col["category_processor"]
            self.feature_map.features[name]["category_processor"] = category_processor
            if category_processor == "quantile_bucket": # transform numeric value to bucket
                num_buckets = col.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                if self.rebuild_dataset:
                    qtf.fit(col_series.values)
                    boundaries = qtf.quantiles_[1:-1]
                    self.processor_dict[name + "::boundaries"] = boundaries
                self.feature_map.features[name]["vocab_size"] = num_buckets
            elif category_processor == "hash_bucket":
                num_buckets = col.get("num_buckets", num_buckets)
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::num_buckets"] = num_buckets
            else:
                raise NotImplementedError("category_processor={} not supported.".format(category_processor))

    def fit_sequence_col(self, col, col_series, min_categr_count=1):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        feature_encoder = col.get("feature_encoder", "layers.MaskedAveragePooling()")
        if feature_encoder not in [None, "null", "None", "none"]:
            self.feature_map.features[name]["feature_encoder"] = feature_encoder
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        splitter = col.get("splitter", "^")
        na_value = col.get("fill_na", "")
        max_len = col.get("max_len", 0)
        padding = col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter,
                              na_value=na_value, max_len=max_len, padding=padding,
                              remap=col.get("remap", True))
        if self.rebuild_dataset:
            tokenizer.fit_on_texts(col_series)
        else:
            if "vocab_size" in col:
                tokenizer.update_vocab(range(col["vocab_size"] - 1))
            else:
                raise ValueError(f"{name}: vocab_size is required when rebuild_dataset=False")
        if "share_embedding" in col:
            self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
            tknzr_name = col["share_embedding"] + "::tokenizer"
            # update vocab of both tokenizers
            self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
            self.feature_map.features[col["share_embedding"]] \
                            .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                     "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
        self.processor_dict[name + "::tokenizer"] = tokenizer
        self.feature_map.features[name].update({"padding_idx": 0,
                                                "oov_idx": tokenizer.vocab["__OOV__"],
                                                "max_len": tokenizer.max_len,
                                                "vocab_size": tokenizer.vocab_size()})

        # Handle compute_sequence_length option for 0-padded sequences
        if col.get("compute_sequence_length", False):
            len_feature_name = name + "_len"
            self.feature_map.features[name]["compute_sequence_length"] = True
            self.feature_map.features[len_feature_name] = {
                "source": feature_source,
                "type": "numeric",
                "derived_from": name
            }

    def transform(self, ddf):
        logging.info("Transform feature columns to IDs...")
        for feature, feature_spec in self.feature_map.features.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                col_series = ddf[feature]
                if feature_type == "meta":
                    if feature + "::tokenizer" in self.processor_dict:
                        tokenizer = self.processor_dict[feature + "::tokenizer"]
                        ddf[feature] = tokenizer.encode_meta(col_series)
                        # Update vocab in tokenizer
                        self.processor_dict[feature + "::tokenizer"] = tokenizer
                elif feature_type == "numeric":
                    normalizer = self.processor_dict.get(feature + "::normalizer")
                    if normalizer:
                        ddf[feature] = normalizer.transform(col_series.values)
                elif feature_type == "categorical":
                    category_processor = feature_spec.get("category_processor")
                    if category_processor is None:
                        ddf[feature] = (
                            self.processor_dict.get(feature + "::tokenizer")
                            .encode_category(col_series)
                        )
                    elif category_processor == "numeric_bucket":
                        raise NotImplementedError
                    elif category_processor == "hash_bucket":
                        raise NotImplementedError
                elif feature_type == "sequence":
                    tokenizer = self.processor_dict.get(feature + "::tokenizer")
                    ddf[feature] = tokenizer.encode_sequence(col_series)
                    # Compute sequence length if enabled
                    if feature_spec.get("compute_sequence_length", False):
                        len_feature_name = feature + "_len"
                        ddf[len_feature_name] = tokenizer.compute_sequence_lengths(col_series)
                elif feature_type == "embedding":
                    continue
                else:
                    raise NotImplementedError
        return ddf

    def load_pickle(self, pickle_file=None):
        """Refuse legacy pickle caches, which can execute code when deserialized."""
        del pickle_file
        raise RuntimeError(
            "Loading FeatureProcessor pickle caches is disabled for security. "
            "Rebuild the processor from dataset_config.yaml and feature_map.json instead."
        )

    def save_pickle(self, pickle_file):
        # JSON feature-map/vocab artifacts written by fit() are the supported,
        # data-only interchange format.  Do not create a new unsafe cache.
        logging.info("Skipping legacy FeatureProcessor pickle cache: " + pickle_file)

    def save_vocab(self, vocab_file):
        logging.info("Save feature_vocab to json: " + vocab_file)
        vocab = dict()
        for feature, spec in self.feature_map.features.items():
            if spec["type"] in ["categorical", "sequence"]:
                # Skip features that use share_embedding, as they share vocab with another feature
                if "share_embedding" in spec:
                    continue
                vocab[feature] = OrderedDict(
                    sorted(self.processor_dict[feature + "::tokenizer"].vocab.items(), key=lambda x:x[1]))
        with open(vocab_file, "w") as fd:
            fd.write(json.dumps(vocab, indent=4))

    def copy_from(self, src_col):
        return pl.col(src_col)
