# Modified by the PriCoRec authors in 2026.
import hashlib
import logging
import math
import os

import numpy as np
import tensorflow as tf
import torch

from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only


configure_tensorflow_cpu_only(tf)


class TFRecordDataLoader(object):
    def __init__(self, feature_map, data_path, split="train", batch_size=32,
                 shuffle=False, drop_remainder=False, tfrecord_load_conf=None, **kwargs):
        self.feature_map = feature_map
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.tfrecord_load_conf = dict(tfrecord_load_conf or {})
        self.map_negative_ids_to_zero = self._as_bool(
            self._get_loader_conf("map_negative_ids_to_zero", True),
            True,
        )
        self.validate_id_range = self._as_bool(
            self._get_loader_conf("validate_id_range", True),
            True,
        )
        self.clip_oov_ids = self._as_bool(
            self._get_loader_conf("clip_oov_ids", False),
            False,
        )
        self._warned_invalid_id_features = set()
        self.autotune = getattr(tf.data, "AUTOTUNE", tf.data.experimental.AUTOTUNE)
        self.num_parallel_reads = self._resolve_runtime_value(
            self._get_loader_conf("num_parallel_reads", self.autotune)
        )
        self.num_parallel_calls = self._resolve_runtime_value(
            self._get_loader_conf("num_parallel_calls", self.autotune)
        )
        self.prefetch_size = self._resolve_runtime_value(
            self._get_loader_conf("prefetch_size", self.autotune)
        )
        self.buffer_size = self._get_loader_conf("buffer_size")
        self.shuffle_size = self._get_loader_conf("shuffle_size")
        self.deterministic = self._get_loader_conf("deterministic", not shuffle)
        self.count_samples = bool(self._get_loader_conf("count_samples", True))
        self.filenames = self._resolve_filenames(data_path)
        self.compression_type = self._resolve_compression_type(
            self._get_loader_conf("compression_type", "AUTO"),
            self.filenames,
        )
        self.example_feature_kinds = self._infer_example_feature_kinds()
        self.schema = self._build_schema()
        self._log_label_schema()
        self.num_samples = self._count_samples() if self.count_samples else -1
        self.num_blocks = len(self.filenames)
        if self.num_samples < 0:
            self.num_batches = 0
        elif self.drop_remainder:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = int(math.ceil(self.num_samples / self.batch_size))

    def _build_schema(self):
        schema = dict()
        for feat, feat_spec in self.feature_map.features.items():
            if feat_spec["type"] == "numeric":
                schema[feat] = tf.io.FixedLenFeature(shape=1, dtype=tf.float32)
            elif feat_spec["type"] in ["categorical", "meta"]:
                schema[feat] = tf.io.FixedLenFeature(shape=1, dtype=self._id_feature_dtype(feat))
            elif feat_spec["type"] == "sequence":
                schema[feat] = tf.io.FixedLenFeature(
                    shape=feat_spec["max_len"],
                    dtype=self._id_feature_dtype(feat),
                )
        for label in self.feature_map.labels:
            schema[label] = self._build_label_feature(label)
        return schema

    def _build_label_feature(self, label):
        dtype = self._label_dtype(label)
        if label == self._impression_id_col():
            if dtype == tf.string:
                return tf.io.FixedLenFeature(shape=1, dtype=dtype, default_value=[b""])
            return tf.io.FixedLenFeature(shape=1, dtype=dtype, default_value=[-1])
        if dtype == tf.int64:
            return tf.io.FixedLenFeature(shape=1, dtype=dtype, default_value=[0])
        return tf.io.FixedLenFeature(shape=1, dtype=dtype)

    def _label_dtype(self, label):
        dataset_config = getattr(self.feature_map, "dataset_config", {}) or {}
        label_col = dataset_config.get("label_col", {})
        label_names = []
        if isinstance(label_col, dict):
            label_names.append(label_col.get("name", "label"))
        elif isinstance(label_col, list):
            label_names.extend([col.get("name") for col in label_col if isinstance(col, dict)])
        else:
            label_names.append(str(label_col))

        if label in label_names or label in {"label", "clk", "click"}:
            return tf.float32
        if label == self._impression_id_col():
            if self._is_string_dtype(self._impression_id_dtype()):
                return tf.string
            inferred_dtype = self._inferred_tf_dtype(label)
            if inferred_dtype is not None:
                return inferred_dtype
            return tf.int64
        id_labels = {
            self._config_name(dataset_config.get("group_id")),
            self._config_name(dataset_config.get("feature_group_id")),
        }
        if label in id_labels:
            inferred_dtype = self._inferred_tf_dtype(label)
            if inferred_dtype is not None:
                return inferred_dtype
            return tf.int64
        return tf.float32

    def _impression_id_col(self):
        dataset_config = getattr(self.feature_map, "dataset_config", {}) or {}
        return self._config_name(dataset_config.get("impression_id_col"), "impression_id")

    def _impression_id_dtype(self):
        dataset_config = getattr(self.feature_map, "dataset_config", {}) or {}
        configured_dtype = dataset_config.get("impression_id_dtype")
        impression_id_col = dataset_config.get("impression_id_col")
        if configured_dtype is None and isinstance(impression_id_col, dict):
            configured_dtype = impression_id_col.get("dtype")
        return str(configured_dtype or "int64").lower()

    @staticmethod
    def _config_name(value, default=None):
        if isinstance(value, dict):
            return value.get("name", default)
        if value in (None, ""):
            return default
        return str(value)

    @staticmethod
    def _is_string_dtype(dtype_name):
        return str(dtype_name).lower() in {"str", "string", "bytes", "byte"}

    def _inferred_tf_dtype(self, feature):
        kind = self.example_feature_kinds.get(feature)
        if kind == "bytes_list":
            return tf.string
        if kind == "int64_list":
            return tf.int64
        if kind == "float_list":
            return tf.float32
        return None

    def _id_feature_dtype(self, feature):
        inferred_dtype = self._inferred_tf_dtype(feature)
        if inferred_dtype == tf.int64 or inferred_dtype == tf.float32:
            return inferred_dtype
        return tf.int64

    def _infer_example_feature_kinds(self):
        if not self.filenames:
            return {}
        try:
            dataset = tf.data.TFRecordDataset(
                [self.filenames[0]],
                compression_type=self.compression_type,
                buffer_size=self.buffer_size,
                num_parallel_reads=1,
            )
            for raw_record in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                return {
                    name: feature.WhichOneof("kind")
                    for name, feature in example.features.feature.items()
                }
        except Exception as exc:
            logging.warning("Failed to infer TFRecord feature kinds from %s: %s", self.filenames[0], exc)
        return {}

    def _log_label_schema(self):
        if not self.feature_map.labels:
            return
        schema_summary = {}
        for label in self.feature_map.labels:
            dtype = self._label_dtype(label)
            schema_summary[label] = getattr(dtype, "name", str(dtype))
        logging.info("TFRecord label schema: %s", schema_summary)

    def _resolve_runtime_value(self, value):
        if value is None:
            return None
        if isinstance(value, int) and value <= 0:
            return self.autotune
        return value

    def _get_loader_conf(self, key, default=None):
        return self.tfrecord_load_conf.get(key, default)

    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_compression_type(self, compression_type, filenames):
        if compression_type is None:
            return ""
        compression_type = str(compression_type).strip()
        if compression_type == "":
            return ""
        compression_upper = compression_type.upper()
        if compression_upper in {"GZIP", "ZLIB"}:
            return compression_upper
        if compression_upper != "AUTO":
            return compression_type
        if not filenames:
            return ""
        first_file = filenames[0]
        if first_file.endswith(".gz"):
            return "GZIP"
        try:
            with tf.io.gfile.GFile(first_file, "rb") as fp:
                magic = fp.read(2)
            if magic == b"\x1f\x8b":
                return "GZIP"
        except Exception:
            pass
        return ""

    def _resolve_filenames(self, filenames):
        if isinstance(filenames, (list, tuple)):
            resolved_files = []
            for filename in filenames:
                resolved_files.extend(self._resolve_filenames(filename))
            return resolved_files
        if os.path.isdir(filenames) or tf.io.gfile.isdir(filenames):
            matched_files = []
            for pattern in ("*.tfrecord", "*.tfrecord.gz"):
                matched_files.extend(sorted(tf.io.gfile.glob(os.path.join(filenames, pattern))))
            if not matched_files:
                for name in sorted(tf.io.gfile.listdir(filenames)):
                    if name.startswith(("_", ".")):
                        continue
                    candidate = os.path.join(filenames, name)
                    if tf.io.gfile.isdir(candidate):
                        continue
                    matched_files.append(candidate)
            if not matched_files:
                raise FileNotFoundError(f"No TFRecord files found in directory: {filenames}")
            return matched_files
        matched_files = sorted(tf.io.gfile.glob(filenames))
        if matched_files:
            return matched_files
        candidate_files = [filenames]
        if not filenames.endswith(".tfrecord") and not filenames.endswith(".tfrecord.gz"):
            candidate_files.extend([f"{filenames}.tfrecord", f"{filenames}.tfrecord.gz"])
        for candidate in candidate_files:
            matched_files = sorted(tf.io.gfile.glob(candidate))
            if matched_files:
                return matched_files
        raise FileNotFoundError(f"TFRecord file not found: {filenames}")

    def _count_samples(self):
        dataset = tf.data.TFRecordDataset(
            self.filenames,
            compression_type=self.compression_type,
            buffer_size=self.buffer_size,
            num_parallel_reads=self.num_parallel_reads
        )
        return sum(1 for _ in dataset)

    def _build_dataset_options(self):
        options = tf.data.Options()
        options.experimental_deterministic = self.deterministic
        return options

    def _build_reader_dataset(self):
        dataset = tf.data.TFRecordDataset(
            self.filenames,
            compression_type=self.compression_type,
            buffer_size=self.buffer_size,
            num_parallel_reads=self.num_parallel_reads
        )
        return dataset.with_options(self._build_dataset_options())

    def _build_dataset(self):
        dataset = self._build_reader_dataset()
        if self.shuffle:
            buffer_size = self.shuffle_size
            if buffer_size is None:
                if self.num_samples > 0:
                    buffer_size = min(self.num_samples, max(self.batch_size * 100, 10000))
                else:
                    buffer_size = max(self.batch_size * 100, 10000)
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.map(
            lambda batch_examples: tf.io.parse_example(batch_examples, features=self.schema),
            num_parallel_calls=self.num_parallel_calls
        )
        if self.prefetch_size is not None:
            dataset = dataset.prefetch(buffer_size=self.prefetch_size)
        return dataset

    def _to_torch_tensor(self, feature, value):
        if feature in self.feature_map.labels:
            dtype = self._label_dtype(feature)
            if dtype == tf.string:
                tensor = self._string_array_to_long_tensor(value.numpy())
            else:
                tensor = torch.from_numpy(value.numpy())
            if dtype == tf.float32:
                tensor = tensor.float()
            else:
                tensor = tensor.long()
            return self._squeeze_last_dim(tensor)
        feat_spec = self.feature_map.features.get(feature, {})
        if feat_spec.get("type") == "numeric":
            return self._squeeze_last_dim(torch.from_numpy(value.numpy()).float())
        tensor = torch.from_numpy(value.numpy()).long()
        if feat_spec.get("type") in {"categorical", "sequence"}:
            tensor = self._sanitize_id_tensor(feature, tensor, feat_spec)
        if feat_spec.get("type") != "sequence":
            tensor = self._squeeze_last_dim(tensor)
        return tensor

    def _sanitize_id_tensor(self, feature, tensor, feat_spec):
        if tensor.numel() == 0:
            return tensor
        min_id = int(torch.min(tensor).item())
        max_id = int(torch.max(tensor).item())
        sanitized = tensor
        if min_id < 0:
            if not self.map_negative_ids_to_zero:
                raise ValueError(
                    f"TFRecord feature '{feature}' contains negative ids: min_id={min_id}, max_id={max_id}. "
                    "Set map_negative_ids_to_zero=true to map missing ids to padding index 0."
                )
            sanitized = tensor.clone()
            sanitized[sanitized < 0] = 0
            self._warn_invalid_ids_once(
                feature,
                f"TFRecord feature '{feature}' contains negative ids (min_id={min_id}); "
                "mapping them to padding index 0.",
            )
            max_id = int(torch.max(sanitized).item())

        vocab_size = int(feat_spec.get("vocab_size", 0) or 0)
        if self.validate_id_range and vocab_size > 0 and max_id >= vocab_size:
            message = (
                f"TFRecord feature '{feature}' id exceeds embedding vocab_size: "
                f"max_id={max_id}, vocab_size={vocab_size}, min_id={min_id}. "
                "Check the encoded ids and feature_map vocab_size alignment."
            )
            if not self.clip_oov_ids:
                raise ValueError(message)
            sanitized = sanitized.clone()
            sanitized[sanitized >= vocab_size] = vocab_size - 1
            self._warn_invalid_ids_once(feature, message + " Clipping to vocab_size - 1.")
        return sanitized

    def _warn_invalid_ids_once(self, feature, message):
        if feature in self._warned_invalid_id_features:
            return
        self._warned_invalid_id_features.add(feature)
        logging.warning(message)

    def _squeeze_last_dim(self, tensor):
        if tensor.dim() > 1 and tensor.size(-1) == 1:
            return tensor.squeeze(-1)
        return tensor

    def _string_array_to_long_tensor(self, array):
        np_array = np.asarray(array)
        hashed_values = [self._hash_string_value(value) for value in np_array.reshape(-1)]
        return torch.tensor(hashed_values, dtype=torch.long).reshape(np_array.shape)

    @staticmethod
    def _hash_string_value(value):
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, str):
            raw_value = value.encode("utf-8")
        elif isinstance(value, bytes):
            raw_value = value
        else:
            raw_value = str(value).encode("utf-8")
        if raw_value == b"":
            return -1
        digest = hashlib.blake2b(raw_value, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF

    def __iter__(self):
        row_offset = 0
        impression_id_col = self._impression_id_col()
        for batch_data in self._build_dataset():
            torch_batch = {
                feature: self._to_torch_tensor(feature, value)
                for feature, value in batch_data.items()
            }
            if impression_id_col in torch_batch:
                impression_ids = torch_batch[impression_id_col]
                batch_size = impression_ids.size(0)
                if torch.all(impression_ids == -1).item():
                    torch_batch[impression_id_col] = torch.arange(
                        row_offset,
                        row_offset + batch_size,
                        dtype=torch.long,
                    )
                row_offset += batch_size
            yield torch_batch

    def __len__(self):
        return self.num_batches
