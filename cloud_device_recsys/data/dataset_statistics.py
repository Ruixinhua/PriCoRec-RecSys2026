import json
import os
import hashlib
import math
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional


_CACHE_VERSION = 4


def write_dataset_statistics(
    dataset_config: Dict[str, Any],
    data_paths: Dict[str, Any],
    output_dir: str,
    logger,
) -> Optional[str]:
    """Scan encoded TFRecord splits and write item/user/request statistics."""
    stats_conf = dict(dataset_config.get("dataset_stats", {}) or {})
    data_format = str(
        dataset_config.get("processed_data_format", dataset_config.get("data_format", data_paths.get("data_format", "")))
    ).lower()
    enabled = _as_bool(stats_conf.get("enabled", data_format in {"tfrecord", "tf_record"}), False)
    if not enabled:
        logger.info("Dataset statistics output disabled.")
        return None
    if data_format not in {"tfrecord", "tf_record"}:
        logger.info("Dataset statistics output currently supports TFRecord data; skipped for %s.", data_format)
        return None

    item_id_col = _config_name(dataset_config.get("item_id_col"), "item_id")
    user_id_col = _config_name(dataset_config.get("user_id_col"), "deviceid")
    impression_id_col = _config_name(dataset_config.get("impression_id_col"), "impression_id")
    label_col = _label_name(dataset_config)
    dataset_id = str(dataset_config.get("dataset_id") or "")
    top_k = int(stats_conf.get("top_k", 20) or 20)
    max_records_per_split = int(stats_conf.get("max_records_per_split", 0) or 0)
    batch_size = int(stats_conf.get("batch_size", 8192) or 8192)
    cache_enabled = _as_bool(stats_conf.get("cache_enabled", True), True)
    cache_policy = str(stats_conf.get("cache_policy", "prefer") or "prefer").lower()
    log_top = _as_bool(stats_conf.get("log_top", False), False)
    progress_interval = int(stats_conf.get("progress_interval", 0) or 0)

    split_paths = {
        "train": data_paths.get("train_path") or dataset_config.get("train_data"),
        "valid": data_paths.get("valid_path") or dataset_config.get("valid_data"),
        "test": data_paths.get("test_path") or dataset_config.get("test_data"),
    }

    logger.info(
        "Writing dataset statistics: item_id_col=%s, user_id_col=%s, request_id_col=%s, label_col=%s, batch_size=%d",
        item_id_col,
        user_id_col,
        impression_id_col,
        label_col,
        batch_size,
    )

    import tensorflow as tf
    from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only

    configure_tensorflow_cpu_only(tf, logger)

    tfrecord_load_conf = dict(dataset_config.get("tfrecord_load_conf", {}) or {})
    split_inputs = {}
    for split, path in split_paths.items():
        if not path:
            continue
        filenames = sorted(dict.fromkeys(_resolve_tfrecord_filenames(str(path), tf)))
        compression_type = _resolve_compression_type(
            tfrecord_load_conf.get("compression_type", "AUTO"),
            filenames,
            tf,
        )
        split_inputs[split] = {
            "path": str(path),
            "filenames": filenames,
            "compression_type": compression_type,
        }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_statistics.json")
    cache_path = _statistics_cache_path(stats_conf, dataset_config, output_path)
    cache_key = _statistics_cache_key(
        dataset_id=dataset_id,
        split_inputs=split_inputs,
        item_id_col=item_id_col,
        user_id_col=user_id_col,
        impression_id_col=impression_id_col,
        label_col=label_col,
        top_k=top_k,
        max_records_per_split=max_records_per_split,
        batch_size=batch_size,
    )
    if cache_enabled:
        cached = _load_cached_statistics(
            cache_path,
            cache_key,
            logger,
            cache_policy=cache_policy,
            expected_dataset_id=dataset_id,
            expected_splits=split_inputs.keys(),
        )
        if cached is not None:
            if os.path.abspath(cache_path) != os.path.abspath(output_path):
                _write_statistics_payload(output_path, cached, logger, "Copied dataset statistics to")
            _log_summary(logger, "combined", cached["combined"])
            return output_path

    split_stats = {}
    combined_counters = {
        "items": Counter(),
        "users": Counter(),
        "requests": Counter(),
    }
    combined_label_distribution = _init_label_distribution()
    combined_rows = 0
    combined_positive = 0

    for split, split_input in split_inputs.items():
        stats = _scan_tfrecord_split(
            split=split,
            path=split_input["path"],
            filenames=split_input["filenames"],
            compression_type=split_input["compression_type"],
            item_id_col=item_id_col,
            user_id_col=user_id_col,
            impression_id_col=impression_id_col,
            label_col=label_col,
            dataset_config=dataset_config,
            tfrecord_load_conf=tfrecord_load_conf,
            top_k=top_k,
            max_records=max_records_per_split,
            batch_size=batch_size,
            log_top=log_top,
            progress_interval=progress_interval,
            logger=logger,
            tf=tf,
        )
        split_stats[split] = _serialise_split_stats(stats, top_k)
        combined_rows += stats["rows"]
        combined_positive += stats["positive_rows"]
        combined_counters["items"].update(stats["item_counter"])
        combined_counters["users"].update(stats["user_counter"])
        combined_counters["requests"].update(stats["request_counter"])
        _merge_label_distribution(combined_label_distribution, stats["label_distribution"])

    combined_stats = {
        "rows": combined_rows,
        "positive_rows": combined_positive,
        "positive_rate": _ratio(combined_positive, combined_rows),
        "label_distribution": _finalise_label_distribution(combined_label_distribution),
        "item_side": _side_stats(combined_counters["items"], top_k),
        "user_side": _side_stats(combined_counters["users"], top_k),
        "request_side": _side_stats(combined_counters["requests"], top_k),
    }

    output = {
        "cache_version": _CACHE_VERSION,
        "cache_key": cache_key,
        "dataset_id": dataset_id,
        "data_format": data_format,
        "item_id_col": item_id_col,
        "user_id_col": user_id_col,
        "impression_id_col": impression_id_col,
        "label_col": label_col,
        "top_k": top_k,
        "max_records_per_split": max_records_per_split,
        "batch_size": batch_size,
        "splits": split_stats,
        "combined": combined_stats,
    }

    primary_path = cache_path if cache_enabled else output_path
    _write_statistics_payload(primary_path, output, logger, "Saved dataset statistics to")
    if os.path.abspath(primary_path) != os.path.abspath(output_path):
        _write_statistics_payload(output_path, output, logger, "Copied dataset statistics to")
    _log_summary(logger, "combined", combined_stats)
    return output_path


def _scan_tfrecord_split(
    split: str,
    path: str,
    filenames: List[str],
    compression_type: str,
    item_id_col: str,
    user_id_col: str,
    impression_id_col: str,
    label_col: str,
    dataset_config: Dict[str, Any],
    tfrecord_load_conf: Dict[str, Any],
    top_k: int,
    max_records: int,
    batch_size: int,
    log_top: bool,
    progress_interval: int,
    logger,
    tf,
) -> Dict[str, Any]:
    if not filenames:
        logger.warning("No TFRecord files found for %s statistics: %s", split, path)
        return _empty_scan_result(split, path)

    field_names = [item_id_col, user_id_col, impression_id_col, label_col]
    configured_kinds = _configured_feature_kinds(dataset_config, field_names, label_col, impression_id_col)
    feature_kinds = _infer_feature_kinds(
        filenames,
        compression_type,
        field_names,
        tf,
        configured_kinds=configured_kinds,
    )
    schema = {
        name: tf.io.VarLenFeature(_tf_dtype_for_feature_kind(kind, tf))
        for name, kind in feature_kinds.items()
        if kind in {"int64_list", "float_list", "bytes_list"}
    }
    unknown_fields = [name for name in field_names if name and name not in schema]
    if unknown_fields:
        logger.warning("[%s] Could not infer TFRecord field type(s); counted as missing: %s", split, unknown_fields)

    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type=compression_type,
        buffer_size=(tfrecord_load_conf or {}).get("buffer_size"),
        num_parallel_reads=(tfrecord_load_conf or {}).get("num_parallel_reads"),
    )
    if max_records > 0:
        dataset = dataset.take(max_records)
    dataset = dataset.batch(batch_size)
    prefetch_size = (tfrecord_load_conf or {}).get("stats_prefetch_size", (tfrecord_load_conf or {}).get("prefetch_size"))
    if prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)

    item_counter = Counter()
    user_counter = Counter()
    request_counter = Counter()
    rows = 0
    positive_rows = 0
    label_distribution = _init_label_distribution()
    missing = Counter()

    import time

    logger.info(
        "[%s] dataset statistics scan started: files=%d, max_records=%s",
        split,
        len(filenames),
        max_records if max_records > 0 else "all",
    )
    start_time = time.time()
    next_progress = progress_interval if progress_interval > 0 else None
    for batch_records in dataset:
        batch_rows = int(batch_records.shape[0] or tf.shape(batch_records)[0].numpy())
        rows += batch_rows
        if schema:
            parsed = tf.io.parse_example(batch_records, features=schema)
        else:
            parsed = {}

        item_values = _first_values_from_sparse(parsed.get(item_id_col), batch_rows, feature_kinds.get(item_id_col))
        user_values = _first_values_from_sparse(parsed.get(user_id_col), batch_rows, feature_kinds.get(user_id_col))
        request_values = _first_values_from_sparse(parsed.get(impression_id_col), batch_rows, feature_kinds.get(impression_id_col))
        label_values = _first_values_from_sparse(parsed.get(label_col), batch_rows, feature_kinds.get(label_col))

        _update_counter_and_missing(item_counter, missing, item_id_col, item_values)
        _update_counter_and_missing(user_counter, missing, user_id_col, user_values)
        _update_counter_and_missing(request_counter, missing, impression_id_col, request_values)
        if label_col:
            label_missing = sum(value is None for value in label_values)
            if label_missing:
                missing[label_col] += label_missing
            _update_label_distribution(label_distribution, label_values)
        positive_rows += sum(1 for value in label_values if _is_positive_label(value))

        if next_progress is not None and rows >= next_progress:
            elapsed = max(time.time() - start_time, 1e-9)
            logger.info("[%s] dataset statistics scanned %d rows (%.1f rows/s)", split, rows, rows / elapsed)
            while rows >= next_progress:
                next_progress += progress_interval

    result = {
        "split": split,
        "path": path,
        "files": filenames,
        "rows": rows,
        "positive_rows": positive_rows,
        "truncated": max_records > 0 and rows >= max_records,
        "missing": dict(missing),
        "label_distribution": label_distribution,
        "item_counter": item_counter,
        "user_counter": user_counter,
        "request_counter": request_counter,
    }
    logger.info(
        "[%s] dataset statistics: rows=%d positives=%d unique_items=%d unique_users=%d unique_requests=%d",
        split,
        rows,
        positive_rows,
        len(item_counter),
        len(user_counter),
        len(request_counter),
    )
    _log_label_distribution(logger, split, _finalise_label_distribution(label_distribution))
    if missing:
        logger.warning("[%s] Missing statistics fields: %s", split, dict(missing))
    if log_top and top_k > 0:
        logger.info("[%s] Top items: %s", split, _format_top(item_counter, top_k))
        logger.info("[%s] Top users: %s", split, _format_top(user_counter, top_k))
    return result


def _configured_feature_kinds(
    dataset_config: Dict[str, Any],
    field_names: List[str],
    label_col: str,
    impression_id_col: str,
) -> Dict[str, str]:
    configured = {}
    label_dtype = None
    label_conf = dataset_config.get("label_col")
    if isinstance(label_conf, dict):
        label_dtype = label_conf.get("dtype")
    label_kind = _feature_kind_from_dtype(label_dtype or "float")
    if label_col and label_kind:
        configured[label_col] = label_kind

    dtype_keys = [
        (dataset_config.get("item_id_col"), dataset_config.get("item_id_dtype")),
        (dataset_config.get("user_id_col"), dataset_config.get("user_id_dtype")),
        (impression_id_col, dataset_config.get("impression_id_dtype")),
    ]
    for name, dtype in dtype_keys:
        normalised_name = _config_name(name)
        if not normalised_name or normalised_name not in field_names:
            continue
        kind = _feature_kind_from_dtype(dtype)
        if kind:
            configured[normalised_name] = kind
    return configured


def _feature_kind_from_dtype(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    value = str(dtype).strip().lower()
    if not value:
        return None
    if value in {"string", "str", "bytes", "byte"}:
        return "bytes_list"
    if value.startswith(("float", "double")):
        return "float_list"
    if value.startswith(("int", "uint", "long")):
        return "int64_list"
    return None


def _infer_feature_kinds(
    filenames: List[str],
    compression_type: str,
    field_names: List[str],
    tf,
    configured_kinds: Optional[Dict[str, str]] = None,
    max_examples: int = 1000,
) -> Dict[str, str]:
    fields = {name for name in field_names if name}
    inferred: Dict[str, str] = {}
    if not fields or not filenames:
        return inferred

    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression_type)
    for record in dataset.take(max_examples):
        example = tf.train.Example.FromString(record.numpy())
        features = example.features.feature
        for name in list(fields - set(inferred)):
            feature = features.get(name)
            if feature is None:
                continue
            kind = feature.WhichOneof("kind")
            if kind in {"int64_list", "float_list", "bytes_list"}:
                inferred[name] = kind
        if fields.issubset(inferred):
            break

    for name, kind in (configured_kinds or {}).items():
        inferred.setdefault(name, kind)
    return inferred


def _tf_dtype_for_feature_kind(kind: str, tf):
    if kind == "bytes_list":
        return tf.string
    if kind == "float_list":
        return tf.float32
    return tf.int64


def _first_values_from_sparse(sparse_tensor: Any, batch_rows: int, kind: Optional[str]) -> List[Any]:
    values = [None] * batch_rows
    if sparse_tensor is None:
        return values
    indices = sparse_tensor.indices.numpy()
    raw_values = sparse_tensor.values.numpy()
    if len(raw_values) == 0:
        return values
    for index, raw_value in zip(indices, raw_values):
        row_index = int(index[0])
        if row_index < 0 or row_index >= batch_rows or values[row_index] is not None:
            continue
        values[row_index] = _normalise_feature_value(raw_value, kind)
    return values


def _normalise_feature_value(value: Any, kind: Optional[str]) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        value = value.item()
    if kind == "bytes_list" or isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.hex()
    if kind == "float_list":
        return float(value)
    if kind == "int64_list":
        return int(value)
    return value


def _update_counter_and_missing(counter: Counter, missing: Counter, field_name: str, values: List[Any]) -> None:
    if not field_name:
        return
    missing_count = sum(value is None for value in values)
    if missing_count:
        missing[field_name] += missing_count
    counter.update(value for value in values if value is not None)


def _statistics_cache_key(
    dataset_id: str,
    split_inputs: Dict[str, Dict[str, Any]],
    item_id_col: str,
    user_id_col: str,
    impression_id_col: str,
    label_col: str,
    top_k: int,
    max_records_per_split: int,
    batch_size: int,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "dataset_id": str(dataset_id or ""),
        "columns": {
            "item_id_col": item_id_col,
            "user_id_col": user_id_col,
            "impression_id_col": impression_id_col,
            "label_col": label_col,
        },
        "top_k": top_k,
        "max_records_per_split": max_records_per_split,
        "batch_size": batch_size,
        "splits": {
            split: {
                "path": value.get("path"),
                "compression_type": value.get("compression_type"),
                "files": [_file_fingerprint(filename) for filename in value.get("filenames", [])],
            }
            for split, value in sorted(split_inputs.items())
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_fingerprint(filename: str) -> Dict[str, Any]:
    try:
        stat = os.stat(filename)
        return {
            "path": filename,
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }
    except OSError:
        return {"path": filename, "size": None, "mtime_ns": None}


def _statistics_cache_path(
    stats_conf: Dict[str, Any],
    dataset_config: Dict[str, Any],
    default_path: str,
) -> str:
    configured_path = stats_conf.get("cache_path")
    if configured_path:
        return os.path.expanduser(os.path.expandvars(str(configured_path)))
    cache_dir = stats_conf.get("cache_dir") or dataset_config.get("cache_dir")
    filename = f"dataset_statistics_{_cache_slug(dataset_config.get('dataset_id'))}.json"
    if cache_dir:
        cache_dir = os.path.expanduser(os.path.expandvars(str(cache_dir)))
        return os.path.join(cache_dir, filename)
    default_dir = os.path.dirname(default_path) or "."
    return os.path.join(default_dir, filename)


def _write_statistics_payload(path: str, payload: Dict[str, Any], logger, action: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("%s %s", action, path)


def _load_cached_statistics(
    output_path: str,
    cache_key: str,
    logger,
    cache_policy: str = "prefer",
    expected_dataset_id: Optional[str] = None,
    expected_splits: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(output_path):
        return None
    if cache_policy in {"refresh", "rebuild", "ignore"}:
        logger.info("Ignoring dataset statistics cache due to cache_policy=%s: %s", cache_policy, output_path)
        return None
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("cache_version") != _CACHE_VERSION:
        return None
    if "combined" not in cached or "splits" not in cached:
        return None
    if expected_dataset_id is not None:
        cached_dataset_id = cached.get("dataset_id")
        if cached_dataset_id != str(expected_dataset_id):
            logger.info(
                "Dataset statistics cache dataset_id mismatch; rebuilding. expected=%s cached=%s path=%s",
                expected_dataset_id,
                cached_dataset_id if cached_dataset_id is not None else "<missing>",
                output_path,
            )
            return None
    if not _cached_statistics_complete(cached, expected_splits, logger, output_path):
        return None
    cache_key_matches = cached.get("cache_key") == cache_key
    if not cache_key_matches and cache_policy in {"strict", "validate", "validated"}:
        logger.info(
            "Dataset statistics cache key mismatch; rebuilding. cache_policy=%s path=%s",
            cache_policy,
            output_path,
        )
        return None
    if not cache_key_matches:
        logger.info(
            "Loaded dataset statistics from cache despite cache key mismatch "
            "(cache_policy=%s): %s",
            cache_policy,
            output_path,
        )
    else:
        logger.info("Loaded dataset statistics from cache: %s", output_path)
    return cached


def _cached_statistics_complete(
    cached: Dict[str, Any],
    expected_splits: Optional[Any],
    logger,
    output_path: str,
) -> bool:
    splits = cached.get("splits")
    if not isinstance(splits, dict):
        logger.info("Dataset statistics cache has no split statistics; rebuilding: %s", output_path)
        return False

    required_splits = [split for split in (expected_splits or []) if split]
    for split in required_splits:
        split_stats = splits.get(split)
        if not isinstance(split_stats, dict):
            logger.info(
                "Dataset statistics cache missing required split=%s; rebuilding: %s",
                split,
                output_path,
            )
            return False
        if split_stats.get("truncated"):
            logger.info(
                "Dataset statistics cache split=%s is truncated; rebuilding full statistics: %s",
                split,
                output_path,
            )
            return False
        try:
            rows = int(split_stats.get("rows", 0) or 0)
        except (TypeError, ValueError):
            rows = 0
        if rows <= 0:
            logger.info(
                "Dataset statistics cache split=%s has no rows; rebuilding: %s",
                split,
                output_path,
            )
            return False

    combined = cached.get("combined") or {}
    try:
        combined_rows = int(combined.get("rows", 0) or 0)
    except (TypeError, ValueError):
        combined_rows = 0
    if required_splits and combined_rows <= 0:
        logger.info("Dataset statistics cache has no combined rows; rebuilding: %s", output_path)
        return False
    return True


def _cache_slug(value: Optional[Any]) -> str:
    text = str(value or "dataset").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", ".", "-"} else "_" for ch in text)
    safe = safe.strip("._-")
    return safe or "dataset"


def _serialise_split_stats(stats: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    rows = stats["rows"]
    positive_rows = stats["positive_rows"]
    return {
        "path": stats["path"],
        "files": stats["files"],
        "rows": rows,
        "positive_rows": positive_rows,
        "positive_rate": _ratio(positive_rows, rows),
        "label_distribution": _finalise_label_distribution(stats.get("label_distribution") or _init_label_distribution()),
        "truncated": stats["truncated"],
        "missing": stats["missing"],
        "item_side": _side_stats(stats["item_counter"], top_k),
        "user_side": _side_stats(stats["user_counter"], top_k),
        "request_side": _side_stats(stats["request_counter"], top_k),
    }


def _side_stats(counter: Counter, top_k: int) -> Dict[str, Any]:
    counts = list(counter.values())
    total = sum(counts)
    return {
        "unique": len(counter),
        "total_events": total,
        "events_per_id": _count_distribution(counts),
        "top": [{"id": _json_value(key), "count": int(value)} for key, value in counter.most_common(top_k)],
    }


def _count_distribution(counts: List[int]) -> Dict[str, Any]:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p99": 0}
    ordered = sorted(int(v) for v in counts)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": float(mean(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
    }


def _percentile(ordered_values: List[int], q: float) -> int:
    if not ordered_values:
        return 0
    index = min(len(ordered_values) - 1, max(0, int(round((len(ordered_values) - 1) * q))))
    return int(ordered_values[index])


def _read_feature_value(features: Any, name: str) -> Any:
    if not name:
        return None
    feature = features.get(name)
    if feature is None:
        return None
    if feature.int64_list.value:
        return int(feature.int64_list.value[0])
    if feature.float_list.value:
        return float(feature.float_list.value[0])
    if feature.bytes_list.value:
        raw = feature.bytes_list.value[0]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.hex()
    return None


def _init_label_distribution() -> Dict[str, Any]:
    return {
        "count": 0,
        "missing_count": 0,
        "non_numeric_count": 0,
        "non_finite_count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "zero_count": 0,
        "one_count": 0,
        "positive_count_gt0": 0,
        "negative_count_lt0": 0,
        "non_binary_count": 0,
    }


def _label_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return number
    return number


def _is_zero_or_one(value: float) -> bool:
    return math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12) or math.isclose(
        value, 1.0, rel_tol=0.0, abs_tol=1e-12
    )


def _update_label_distribution(stats: Dict[str, Any], values: List[Any]) -> None:
    for value in values:
        if value is None:
            stats["missing_count"] += 1
            continue
        number = _label_to_float(value)
        if number is None:
            stats["non_numeric_count"] += 1
            continue
        if not math.isfinite(number):
            stats["non_finite_count"] += 1
            continue
        stats["count"] += 1
        stats["sum"] += number
        stats["min"] = number if stats["min"] is None else min(stats["min"], number)
        stats["max"] = number if stats["max"] is None else max(stats["max"], number)
        if math.isclose(number, 0.0, rel_tol=0.0, abs_tol=1e-12):
            stats["zero_count"] += 1
        if math.isclose(number, 1.0, rel_tol=0.0, abs_tol=1e-12):
            stats["one_count"] += 1
        if number > 0:
            stats["positive_count_gt0"] += 1
        if number < 0:
            stats["negative_count_lt0"] += 1
        if not _is_zero_or_one(number):
            stats["non_binary_count"] += 1


def _merge_label_distribution(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in [
        "count",
        "missing_count",
        "non_numeric_count",
        "non_finite_count",
        "zero_count",
        "one_count",
        "positive_count_gt0",
        "negative_count_lt0",
        "non_binary_count",
    ]:
        target[key] += int(source.get(key, 0) or 0)
    target["sum"] += float(source.get("sum", 0.0) or 0.0)
    source_min = source.get("min")
    source_max = source.get("max")
    if source_min is not None:
        target["min"] = source_min if target["min"] is None else min(target["min"], source_min)
    if source_max is not None:
        target["max"] = source_max if target["max"] is None else max(target["max"], source_max)


def _finalise_label_distribution(stats: Dict[str, Any]) -> Dict[str, Any]:
    count = int(stats.get("count", 0) or 0)
    positive_count = int(stats.get("positive_count_gt0", 0) or 0)
    one_count = int(stats.get("one_count", 0) or 0)
    non_binary_count = int(stats.get("non_binary_count", 0) or 0)
    return {
        "count": count,
        "missing_count": int(stats.get("missing_count", 0) or 0),
        "non_numeric_count": int(stats.get("non_numeric_count", 0) or 0),
        "non_finite_count": int(stats.get("non_finite_count", 0) or 0),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "mean": float(stats.get("sum", 0.0) or 0.0) / count if count > 0 else None,
        "sum": float(stats.get("sum", 0.0) or 0.0),
        "zero_count": int(stats.get("zero_count", 0) or 0),
        "one_count": one_count,
        "positive_count_gt0": positive_count,
        "negative_count_lt0": int(stats.get("negative_count_lt0", 0) or 0),
        "non_binary_count": non_binary_count,
        "positive_rate_gt0": _ratio(positive_count, count),
        "positive_rate_eq1": _ratio(one_count, count),
        "non_binary_rate": _ratio(non_binary_count, count),
    }


def _fmt_optional_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _log_label_distribution(logger, split: str, stats: Dict[str, Any]) -> None:
    logger.info(
        "[%s] label distribution: count=%d missing=%d min=%s max=%s mean=%s "
        "zero=%d one=%d gt0=%d neg=%d non_binary=%d non_numeric=%d non_finite=%d",
        split,
        stats["count"],
        stats["missing_count"],
        _fmt_optional_float(stats["min"]),
        _fmt_optional_float(stats["max"]),
        _fmt_optional_float(stats["mean"]),
        stats["zero_count"],
        stats["one_count"],
        stats["positive_count_gt0"],
        stats["negative_count_lt0"],
        stats["non_binary_count"],
        stats["non_numeric_count"],
        stats["non_finite_count"],
    )


def _is_positive_label(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _empty_scan_result(split: str, path: str) -> Dict[str, Any]:
    return {
        "split": split,
        "path": path,
        "files": [],
        "rows": 0,
        "positive_rows": 0,
        "label_distribution": _init_label_distribution(),
        "truncated": False,
        "missing": {},
        "item_counter": Counter(),
        "user_counter": Counter(),
        "request_counter": Counter(),
    }


def _resolve_tfrecord_filenames(path: str, tf) -> List[str]:
    path = os.path.expanduser(os.path.expandvars(path))
    if os.path.isdir(path) or tf.io.gfile.isdir(path):
        filenames = []
        for pattern in ("*.tfrecord", "*.tfrecord.gz"):
            filenames.extend(sorted(tf.io.gfile.glob(os.path.join(path, pattern))))
        if filenames:
            return filenames
        for name in sorted(tf.io.gfile.listdir(path)):
            if name.startswith(("_", ".")):
                continue
            candidate = os.path.join(path, name)
            if not tf.io.gfile.isdir(candidate):
                filenames.append(candidate)
        return filenames
    matched = sorted(tf.io.gfile.glob(path))
    return matched or [path]


def _resolve_compression_type(compression_type: Any, filenames: List[str], tf) -> str:
    if compression_type is None:
        return ""
    compression = str(compression_type).strip()
    if not compression:
        return ""
    upper = compression.upper()
    if upper in {"GZIP", "ZLIB"}:
        return upper
    if upper != "AUTO":
        return compression
    if not filenames:
        return ""
    first_file = filenames[0]
    if first_file.endswith(".gz"):
        return "GZIP"
    try:
        with tf.io.gfile.GFile(first_file, "rb") as fp:
            if fp.read(2) == b"\x1f\x8b":
                return "GZIP"
    except Exception:
        return ""
    return ""


def _label_name(dataset_config: Dict[str, Any]) -> str:
    label_col = dataset_config.get("label_col", "label")
    if isinstance(label_col, dict):
        return _config_name(label_col, "label")
    if isinstance(label_col, list) and label_col:
        return _config_name(label_col[0], "label")
    return str(label_col or "label")


def _config_name(value: Any, default: Optional[str] = None) -> Optional[str]:
    if isinstance(value, dict):
        return value.get("name", default)
    if value in (None, ""):
        return default
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _json_value(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _format_top(counter: Counter, top_k: int) -> str:
    return ", ".join(f"{_json_value(key)}:{value}" for key, value in counter.most_common(top_k)) or "(none)"


def _log_summary(logger, label: str, stats: Dict[str, Any]) -> None:
    logger.info(
        "[%s] dataset stats summary: rows=%d positives=%d unique_items=%d unique_users=%d unique_requests=%d",
        label,
        stats["rows"],
        stats["positive_rows"],
        stats["item_side"]["unique"],
        stats["user_side"]["unique"],
        stats["request_side"]["unique"],
    )
    label_distribution = stats.get("label_distribution")
    if label_distribution:
        _log_label_distribution(logger, label, label_distribution)
