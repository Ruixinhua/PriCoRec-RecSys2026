import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional


def ensure_fuxictr_feature_map(
    feature_map_path: str,
    dataset_config: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    data_dir: str,
    logger,
    output_dir: Optional[str] = None,
) -> str:
    """Return a FuxiCTR-compatible feature_map.json path for encoded TFRecord data."""
    dataset_id = pipeline_config["dataset_id"]
    data_format = str(dataset_config.get("processed_data_format", dataset_config.get("data_format", ""))).lower()
    feature_defs = _expanded_feature_defs(dataset_config)

    raw_map = _load_json(feature_map_path) if feature_map_path and os.path.exists(feature_map_path) else {}
    existing_specs = _extract_feature_specs(raw_map)

    if _is_complete_fuxictr_map(raw_map, existing_specs, feature_defs):
        fuxictr_map = _normalise_fuxictr_map(raw_map, dataset_id, dataset_config)
        return _write_feature_map(
            fuxictr_map,
            feature_map_path,
            data_dir,
            output_dir,
            logger,
            reason="normalised existing FuxiCTR feature map",
        )

    if data_format not in {"tfrecord", "tf_record"} or not feature_defs:
        fuxictr_map = _normalise_fuxictr_map(raw_map, dataset_id, dataset_config)
        if not fuxictr_map.get("features"):
            raise ValueError(
                "feature_map.json is not FuxiCTR-compatible and dataset config has no feature_cols "
                "to generate one"
            )
        return _write_feature_map(
            fuxictr_map,
            feature_map_path,
            data_dir,
            output_dir,
            logger,
            reason="best-effort normalisation",
        )

    scan_conf = dict(dataset_config.get("feature_map_infer_conf", {}) or {})
    default_vocab_size = int(scan_conf.get("default_vocab_size", 2))
    max_records = int(scan_conf.get("max_records", 0) or 0)
    logger.warning(
        "Input feature map is not a complete FuxiCTR feature map; generating one from "
        "dataset feature_cols and TFRecord statistics."
    )
    if max_records > 0:
        logger.info("Feature map inference will scan at most %d TFRecord records", max_records)
    else:
        logger.info("Feature map inference will scan all configured TFRecord splits")

    stats = _scan_tfrecord_stats(
        dataset_config,
        feature_defs,
        max_records=max_records,
        logger=logger,
        cache_path=_scan_cache_path(
            scan_conf,
            output_dir,
            data_dir,
            dataset_id=dataset_config.get("dataset_id"),
        ),
    )
    fuxictr_map = _build_feature_map_from_defs(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        feature_defs=feature_defs,
        existing_specs=existing_specs,
        stats=stats,
        default_vocab_size=default_vocab_size,
        logger=logger,
    )
    return _write_feature_map(
        fuxictr_map,
        feature_map_path,
        data_dir,
        output_dir,
        logger,
        reason="generated from TFRecord statistics",
    )


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(os.path.expanduser(os.path.expandvars(path)), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _label_names(dataset_config: Dict[str, Any]) -> List[str]:
    label_col = dataset_config.get("label_col", "label")
    if isinstance(label_col, dict):
        return [label_col.get("name", "label")]
    if isinstance(label_col, list):
        return [col.get("name") for col in label_col if isinstance(col, dict) and col.get("name")]
    return [str(label_col or "label")]


def _expanded_feature_defs(dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    feature_cols = dataset_config.get("feature_cols_expanded") or dataset_config.get("feature_cols") or []
    expanded = []
    for feat_def in feature_cols:
        names = feat_def.get("name")
        if isinstance(names, list):
            for name in names:
                item = dict(feat_def)
                item["name"] = name
                expanded.append(item)
        elif names:
            expanded.append(dict(feat_def))
    return expanded


def _extract_feature_specs(raw_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = {}
    features = raw_map.get("features")
    if isinstance(features, dict):
        for name, spec in features.items():
            specs[str(name)] = dict(spec) if isinstance(spec, dict) else {}
    elif isinstance(features, list):
        for item in features:
            if isinstance(item, dict):
                for name, spec in item.items():
                    specs[str(name)] = dict(spec) if isinstance(spec, dict) else {}
    return specs


def _is_complete_fuxictr_map(
    raw_map: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
    feature_defs: List[Dict[str, Any]],
) -> bool:
    if not raw_map.get("features") or not specs:
        return False
    required_defs = feature_defs or [{"name": name, **spec} for name, spec in specs.items()]
    for feat_def in required_defs:
        name = feat_def.get("name")
        if name not in specs:
            return False
        feat_type = (specs[name].get("type") or feat_def.get("type") or "").lower()
        if feat_type in {"categorical", "sequence"} and not _as_positive_int(specs[name].get("vocab_size")):
            return False
        if feat_type == "sequence" and not _as_positive_int(specs[name].get("max_len")):
            return False
    return True


def _normalise_fuxictr_map(
    raw_map: Dict[str, Any],
    dataset_id: str,
    dataset_config: Dict[str, Any],
) -> Dict[str, Any]:
    specs = _extract_feature_specs(raw_map)
    labels = raw_map.get("labels") or _label_names(dataset_config)
    features = [{name: spec} for name, spec in specs.items()]
    feature_map = {
        "dataset_id": raw_map.get("dataset_id") or dataset_id,
        "num_fields": raw_map.get("num_fields") or len(features),
        "total_features": raw_map.get("total_features") or _total_features(specs),
        "input_length": raw_map.get("input_length") or _input_length(specs),
        "labels": labels,
        "features": features,
    }
    if feature_map["dataset_id"] != dataset_id:
        feature_map["dataset_id"] = dataset_id
    return feature_map


def _build_feature_map_from_defs(
    dataset_id: str,
    dataset_config: Dict[str, Any],
    feature_defs: List[Dict[str, Any]],
    existing_specs: Dict[str, Dict[str, Any]],
    stats: Dict[str, Dict[str, int]],
    default_vocab_size: int,
    logger=None,
) -> Dict[str, Any]:
    specs = {}
    for feat_def in feature_defs:
        name = feat_def["name"]
        feat_type = _normalise_type(feat_def.get("type"))
        spec = dict(existing_specs.get(name, {}))
        spec["source"] = spec.get("source", "")
        spec["type"] = feat_type
        for key, value in feat_def.items():
            if key in {"name", "type", "dtype", "feature_group"}:
                continue
            if value is not None and key not in spec:
                spec[key] = value
        if feat_type in {"categorical", "sequence"}:
            observed_vocab = _as_int(stats.get(name, {}).get("max_value"), -1) + 1
            configured_vocab = _as_positive_int(spec.get("vocab_size"))
            vocab_size = max(configured_vocab or 0, observed_vocab, default_vocab_size)
            spec["vocab_size"] = int(vocab_size)
            spec["padding_idx"] = int(spec.get("padding_idx", 0))
        if feat_type == "sequence":
            observed_len = stats.get(name, {}).get("max_len", 0)
            configured_len = (
                _as_positive_int(spec.get("max_len"))
                or _as_positive_int(feat_def.get("max_len"))
                or observed_len
                or 1
            )
            spec["max_len"] = int(max(configured_len, observed_len, 1))
            for key in ["share_embedding", "feature_encoder"]:
                if feat_def.get(key) and not spec.get(key):
                    spec[key] = feat_def[key]
            if not spec.get("feature_encoder"):
                spec["feature_encoder"] = _sequence_feature_encoder(dataset_config, feat_def)
        specs[name] = spec

    _drop_unsafe_share_embeddings(specs, logger)

    return {
        "dataset_id": dataset_id,
        "num_fields": len(specs),
        "total_features": _total_features(specs),
        "input_length": _input_length(specs),
        "labels": _label_names(dataset_config),
        "features": [{name: spec} for name, spec in specs.items()],
    }


def _drop_unsafe_share_embeddings(specs: Dict[str, Dict[str, Any]], logger=None) -> None:
    """Remove share_embedding links when per-feature ID spaces cannot fit the shared table."""
    for name, spec in specs.items():
        share_target = spec.get("share_embedding")
        if not share_target:
            continue
        share_target = str(share_target)
        target_spec = specs.get(share_target)
        if target_spec is None:
            spec.pop("share_embedding", None)
            if logger:
                logger.warning(
                    "Removed share_embedding for feature %s -> %s because target feature is missing.",
                    name,
                    share_target,
                )
            continue

        own_vocab_size = _as_positive_int(spec.get("vocab_size")) or 0
        target_vocab_size = _as_positive_int(target_spec.get("vocab_size")) or 0
        if own_vocab_size > target_vocab_size > 0:
            spec.pop("share_embedding", None)
            if logger:
                logger.warning(
                    "Removed unsafe share_embedding for feature %s -> %s because "
                    "own vocab_size=%d exceeds shared vocab_size=%d.",
                    name,
                    share_target,
                    own_vocab_size,
                    target_vocab_size,
                )


def _normalise_type(value: Any) -> str:
    value = str(value or "categorical").lower()
    if value in {"float", "dense", "number", "numeric"}:
        return "numeric"
    if value in {"seq", "sparse_seq", "sequence"}:
        return "sequence"
    return "categorical"


def _sequence_feature_encoder(dataset_config: Dict[str, Any], feat_def: Optional[Dict[str, Any]] = None) -> str:
    pooling = None
    if isinstance(feat_def, dict):
        pooling = feat_def.get("sequence_pooling")
    if not pooling:
        pooling = dataset_config.get("sequence_pooling")
    if not pooling:
        infer_conf = dataset_config.get("feature_map_infer_conf", {}) or {}
        pooling = infer_conf.get("sequence_pooling")
    pooling = str(pooling or "sum").strip().lower()
    if pooling in {"mean", "avg", "average"}:
        return "layers.MaskedAveragePooling()"
    return "layers.MaskedSumPooling()"


def _scan_tfrecord_stats(
    dataset_config: Dict[str, Any],
    feature_defs: List[Dict[str, Any]],
    max_records: int,
    logger,
    cache_path: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    import tensorflow as tf
    from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only

    configure_tensorflow_cpu_only(tf, logger)

    scan_names = {
        feat_def["name"]: _normalise_type(feat_def.get("type"))
        for feat_def in feature_defs
        if _normalise_type(feat_def.get("type")) in {"categorical", "sequence"}
    }
    stats = {name: {"max_value": None, "min_value": None, "max_len": 0} for name in scan_names}
    if not scan_names:
        return stats

    paths = [
        dataset_config.get("train_data"),
        dataset_config.get("valid_data"),
        dataset_config.get("test_data"),
    ]
    filenames = []
    for path in paths:
        if path:
            filenames.extend(_resolve_tfrecord_filenames(path, tf))
    filenames = sorted(dict.fromkeys(filenames))
    if not filenames:
        logger.warning("No TFRecord files found while inferring feature map statistics")
        return stats

    loader_conf = dict(dataset_config.get("tfrecord_load_conf", {}) or {})
    scan_conf = dict(dataset_config.get("feature_map_infer_conf", {}) or {})
    cache_policy = str(scan_conf.get("cache_policy", "prefer") or "prefer").lower()
    cache_require_all_features = _as_bool(scan_conf.get("cache_require_all_features", True), True)
    compression_type = _resolve_compression_type(
        loader_conf.get("compression_type", "AUTO"),
        filenames,
        tf,
    )
    batch_size = int(loader_conf.get("scan_batch_size", 8192) or 8192)
    progress_interval = int(loader_conf.get("scan_progress_interval", 0) or 0)
    cache_key = _scan_cache_key(
        dataset_id=dataset_config.get("dataset_id"),
        filenames=filenames,
        scan_names=scan_names,
        max_records=max_records,
        compression_type=compression_type,
        batch_size=batch_size,
    )
    cached_stats = _load_scan_cache(
        cache_path,
        cache_key,
        logger,
        cache_policy=cache_policy,
        required_feature_names=scan_names.keys() if cache_require_all_features else None,
        expected_dataset_id=dataset_config.get("dataset_id"),
    )
    if cached_stats is not None:
        return cached_stats

    logger.info(
        "TFRecord feature stats scan: files=%d, features=%d, batch_size=%d, max_records=%s, cache=%s",
        len(filenames),
        len(scan_names),
        batch_size,
        max_records if max_records > 0 else "all",
        cache_path or "disabled",
    )

    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type=compression_type,
        buffer_size=loader_conf.get("buffer_size"),
        num_parallel_reads=loader_conf.get("num_parallel_reads"),
    )
    if max_records > 0:
        dataset = dataset.take(max_records)
    feature_kinds = _infer_tfrecord_feature_kinds(
        filenames,
        compression_type,
        scan_names.keys(),
        tf,
    )
    schema = {
        name: tf.io.VarLenFeature(_tf_dtype_for_feature_kind(feature_kinds.get(name), tf))
        for name in scan_names
    }
    dataset = dataset.batch(batch_size)
    prefetch_size = loader_conf.get("scan_prefetch_size", loader_conf.get("prefetch_size"))
    if prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)

    start_time = time.time()
    next_progress = progress_interval if progress_interval > 0 else None
    scanned = 0
    for batch_records in dataset:
        batch_rows = int(batch_records.shape[0] or tf.shape(batch_records)[0].numpy())
        parsed = tf.io.parse_example(batch_records, features=schema)
        for name, feat_type in scan_names.items():
            sparse_tensor = parsed.get(name)
            if sparse_tensor is None:
                continue
            values = sparse_tensor.values
            if int(tf.size(values).numpy()) == 0:
                continue
            max_value = int(tf.reduce_max(values).numpy())
            current_max = stats[name].get("max_value")
            if current_max is None or max_value > current_max:
                stats[name]["max_value"] = max_value
            min_value = int(tf.reduce_min(values).numpy())
            current_min = stats[name].get("min_value")
            if current_min is None or min_value < current_min:
                stats[name]["min_value"] = min_value
            if feat_type == "sequence":
                row_indices = sparse_tensor.indices[:, 0]
                counts = tf.math.bincount(
                    tf.cast(row_indices, tf.int32),
                    minlength=batch_rows,
                    maxlength=batch_rows,
                )
                max_len = int(tf.reduce_max(counts).numpy())
                if max_len > stats[name]["max_len"]:
                    stats[name]["max_len"] = max_len
        scanned += batch_rows
        if next_progress is not None and scanned >= next_progress:
            elapsed = max(time.time() - start_time, 1e-9)
            logger.info(
                "Scanned %d TFRecord examples for feature map inference (%.1f records/s)",
                scanned,
                scanned / elapsed,
            )
            while scanned >= next_progress:
                next_progress += progress_interval
    elapsed = max(time.time() - start_time, 1e-9)
    logger.info(
        "Scanned %d TFRecord examples for feature map inference in %.1fs (%.1f records/s)",
        scanned,
        elapsed,
        scanned / elapsed,
    )
    for values in stats.values():
        if values.get("max_value") is None:
            values["max_value"] = -1
        if values.get("min_value") is None:
            values["min_value"] = 0
    _write_scan_cache(
        cache_path,
        cache_key,
        stats,
        scanned,
        filenames,
        logger,
        dataset_id=dataset_config.get("dataset_id"),
    )
    return stats


def _infer_tfrecord_feature_kinds(
    filenames: List[str],
    compression_type: str,
    feature_names: Iterable[str],
    tf,
    max_examples: int = 1000,
) -> Dict[str, str]:
    targets = {str(name) for name in feature_names if name}
    inferred: Dict[str, str] = {}
    if not filenames or not targets:
        return inferred
    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression_type)
    for record in dataset.take(max_examples):
        example = tf.train.Example.FromString(record.numpy())
        features = example.features.feature
        for name in list(targets - set(inferred)):
            feature = features.get(name)
            if feature is None:
                continue
            kind = feature.WhichOneof("kind")
            if kind in {"int64_list", "float_list"}:
                inferred[name] = kind
        if targets.issubset(inferred):
            break
    return inferred


def _tf_dtype_for_feature_kind(kind: Optional[str], tf):
    if kind == "float_list":
        return tf.float32
    return tf.int64


def _scan_cache_path(
    scan_conf: Dict[str, Any],
    output_dir: Optional[str],
    data_dir: Optional[str],
    dataset_id: Optional[str] = None,
) -> Optional[str]:
    if not _as_bool(scan_conf.get("cache_enabled", True), True):
        return None
    configured_path = scan_conf.get("cache_path")
    if configured_path:
        return os.path.expanduser(os.path.expandvars(str(configured_path)))
    filename = f"feature_map_tfrecord_stats_cache_{_cache_slug(dataset_id)}.json"
    configured_dir = scan_conf.get("cache_dir")
    if configured_dir:
        target_dir = os.path.expanduser(os.path.expandvars(str(configured_dir)))
        return os.path.join(target_dir, filename)
    target_dir = output_dir or data_dir
    if not target_dir:
        return None
    return os.path.join(target_dir, filename)


def _scan_cache_key(
    dataset_id: Optional[str],
    filenames: List[str],
    scan_names: Dict[str, str],
    max_records: int,
    compression_type: str,
    batch_size: int,
) -> str:
    file_fingerprints = []
    for filename in filenames:
        try:
            stat = os.stat(filename)
            file_fingerprints.append({
                "path": filename,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            })
        except OSError:
            file_fingerprints.append({"path": filename, "missing": True})
    payload = {
        "dataset_id": str(dataset_id or ""),
        "files": file_fingerprints,
        "scan_names": sorted(scan_names.items()),
        "max_records": int(max_records or 0),
        "compression_type": compression_type or "",
        "batch_size": int(batch_size or 0),
        "stats_version": 3,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_scan_cache(
    cache_path: Optional[str],
    cache_key: str,
    logger,
    cache_policy: str = "prefer",
    required_feature_names: Optional[Iterable[str]] = None,
    expected_dataset_id: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    if not cache_path or not os.path.isfile(cache_path):
        return None
    if cache_policy in {"refresh", "rebuild", "ignore"}:
        logger.info("Ignoring TFRecord stats cache due to cache_policy=%s: %s", cache_policy, cache_path)
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read TFRecord stats cache %s: %s", cache_path, exc)
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    if expected_dataset_id is not None:
        cached_dataset_id = payload.get("dataset_id")
        if cached_dataset_id != str(expected_dataset_id):
            logger.info(
                "TFRecord stats cache dataset_id mismatch; rebuilding. expected=%s cached=%s path=%s",
                expected_dataset_id,
                cached_dataset_id if cached_dataset_id is not None else "<missing>",
                cache_path,
            )
            return None
    if required_feature_names is not None:
        missing = sorted(str(name) for name in required_feature_names if str(name) not in stats)
        if missing:
            logger.info(
                "TFRecord stats cache does not cover %d required feature(s); rebuilding. Missing examples: %s",
                len(missing),
                ", ".join(missing[:20]),
            )
            return None
    cache_key_matches = payload.get("cache_key") == cache_key
    if not cache_key_matches and cache_policy in {"strict", "validate", "validated"}:
        logger.info("TFRecord stats cache miss: %s", cache_path)
        return None
    if not cache_key_matches:
        logger.info(
            "Loaded TFRecord feature stats from cache despite cache key mismatch "
            "(cache_policy=%s): %s",
            cache_policy,
            cache_path,
        )
    else:
        logger.info(
            "Loaded TFRecord feature stats from cache: %s (scanned_records=%s)",
            cache_path,
            payload.get("scanned_records", "unknown"),
        )
    return {
        str(name): {
            "max_value": _as_int(values.get("max_value"), -1),
            "min_value": _as_int(values.get("min_value"), 0),
            "max_len": _as_int(values.get("max_len"), 0),
        }
        for name, values in stats.items()
        if isinstance(values, dict)
    }


def _write_scan_cache(
    cache_path: Optional[str],
    cache_key: str,
    stats: Dict[str, Dict[str, int]],
    scanned: int,
    filenames: List[str],
    logger,
    dataset_id: Optional[str] = None,
) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "dataset_id": str(dataset_id or ""),
            "scanned_records": int(scanned),
            "files": filenames,
            "stats": stats,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved TFRecord feature stats cache to %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to write TFRecord stats cache %s: %s", cache_path, exc)


def _cache_slug(value: Optional[Any]) -> str:
    text = str(value or "dataset").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", ".", "-"} else "_" for ch in text)
    safe = safe.strip("._-")
    return safe or "dataset"


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


def _write_feature_map(
    feature_map: Dict[str, Any],
    original_path: str,
    data_dir: str,
    output_dir: Optional[str],
    logger,
    reason: str,
) -> str:
    target_dir = output_dir or data_dir or os.path.dirname(original_path)
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "feature_map_fuxictr.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_map, f, indent=2)
    logger.info("Saved %s to %s", reason, output_path)
    return output_path


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _total_features(specs: Dict[str, Dict[str, Any]]) -> int:
    total = 0
    for spec in specs.values():
        if spec.get("type") in {"categorical", "sequence"}:
            total += int(spec.get("vocab_size", 0) or 0)
    return total


def _input_length(specs: Dict[str, Dict[str, Any]]) -> int:
    length = 0
    for spec in specs.values():
        if spec.get("type") == "sequence":
            length += int(spec.get("max_len", 1) or 1)
        else:
            length += 1
    return length
