# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Item Pool Generation Utilities

This module provides utilities for generating and managing item pools
for the recommendation pipeline. The item pool is essential for:
- Building item embeddings index (Retrieval stage)
- Negative sampling during training
- Item feature lookup during inference (Preranking/Reranking)
"""

import hashlib
import json
import logging
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence
import pandas as pd


_ITEM_POOL_CACHE_VERSION = 1


def _atomic_write_parquet(item_df: pd.DataFrame, output_path: str) -> None:
    """Write a pool atomically so a failed build cannot become a valid cache."""
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".item_pool_",
        suffix=".parquet",
        dir=output_dir,
    )
    os.close(fd)
    try:
        item_df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _iter_tabular_item_frames(
    input_path: str,
    columns: Sequence[str],
    batch_size: int = 65536,
) -> Iterable[pd.DataFrame]:
    """Yield only required item columns without loading every source row at once."""
    if input_path.endswith('.parquet'):
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(input_path)
            available_columns = set(parquet_file.schema_arrow.names)
            missing_columns = [column for column in columns if column not in available_columns]
            if missing_columns:
                raise KeyError(
                    f"Item pool source {input_path} is missing required columns: {missing_columns}"
                )
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
                yield batch.to_pandas()
            return
        except ImportError:
            # pyarrow is a project dependency, but retain a functional fallback
            # for environments that use a different pandas parquet engine.
            yield pd.read_parquet(input_path, columns=list(columns))
            return

    if input_path.endswith('.csv'):
        try:
            for frame in pd.read_csv(input_path, usecols=list(columns), chunksize=batch_size):
                yield frame
            return
        except ValueError as exc:
            raise KeyError(
                f"Item pool source {input_path} is missing required columns: {list(columns)}"
            ) from exc

    raise ValueError(f"Unsupported item pool source format: {input_path}")


def _take_unseen_items(
    frame: pd.DataFrame,
    item_id_col: str,
    seen_item_ids: set,
) -> pd.DataFrame:
    """Keep the first occurrence of each item while retaining only O(unique-items)."""
    if item_id_col not in frame.columns:
        raise KeyError(f"Item pool source is missing item id column '{item_id_col}'")
    if frame[item_id_col].isna().any():
        raise ValueError(f"Item pool source contains null values in '{item_id_col}'")

    frame = frame.drop_duplicates(subset=[item_id_col], keep='first')
    if seen_item_ids:
        frame = frame.loc[~frame[item_id_col].isin(seen_item_ids)]
    if frame.empty:
        return frame

    seen_item_ids.update(frame[item_id_col].tolist())
    return frame


def extract_item_corpus(
    input_paths: List[str],
    output_path: str,
    item_id_col: str,
    item_feature_cols: List[str],
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Extract a unique item corpus from one or more dataset files.

    This function reads item columns in bounded batches, extracts unique items
    based on item_id, and saves the result. It retains only unique rows across
    batches rather than materialising every source row before de-duplication.

    Args:
        input_paths: List of paths to input datasets (parquet or csv)
        output_path: Path to save the output item pool (parquet)
        item_id_col: Column name for the unique item ID
        item_feature_cols: List of item feature column names to include
        logger: Optional logger instance

    Returns:
        DataFrame containing the unique item pool
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    cols_to_keep = [item_id_col] + [
        column for column in item_feature_cols if column != item_id_col
    ]
    unique_frames = []
    seen_item_ids = set()
    original_count = 0

    for input_path in input_paths:
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found: {input_path}, skipping...")
            continue

        logger.info(f"Loading data from {input_path}...")

        source_rows = 0
        for frame in _iter_tabular_item_frames(input_path, cols_to_keep):
            source_rows += len(frame)
            unseen = _take_unseen_items(frame, item_id_col, seen_item_ids)
            if not unseen.empty:
                unique_frames.append(unseen)
        original_count += source_rows
        logger.info(f"  Scanned {source_rows} rows from {os.path.basename(input_path)}")

    if not unique_frames:
        raise ValueError("No valid input files found")

    item_df = pd.concat(unique_frames, ignore_index=True)
    logger.info(f"Total rows: {original_count} -> Unique items: {len(item_df)}")
    _atomic_write_parquet(item_df, output_path)
    logger.info(f"Saved item pool to {output_path}")

    return item_df


def extract_item_corpus_from_tfrecord(
    input_paths: List[str],
    output_path: str,
    item_id_col: str,
    item_feature_cols: List[str],
    feature_map,
    tfrecord_load_conf: Optional[dict] = None,
    batch_size: int = 8192,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Extract a unique item corpus from encoded TFRecord files and save as parquet.

    The output remains parquet because downstream retrieval, negative sampling,
    and ranking lookup code already consume item pools as DataFrames.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if feature_map is None:
        raise ValueError("feature_map is required to generate item pool from TFRecord data")

    item_cols = [item_id_col] + [c for c in item_feature_cols if c != item_id_col]
    missing_item_cols = [c for c in item_cols if c not in feature_map.features]
    if missing_item_cols:
        raise ValueError(
            "Configured item pool fields are missing from feature_map: "
            f"{missing_item_cols}"
        )
    available_item_cols = item_cols

    import copy
    import numpy as np
    from fuxictr.pytorch.dataloaders.tfrecord_dataloader import TFRecordDataLoader

    item_feature_map = copy.deepcopy(feature_map)
    item_feature_map.features = {
        name: spec
        for name, spec in item_feature_map.features.items()
        if name in available_item_cols
    }
    item_feature_map.labels = []
    item_feature_map.set_column_index()

    loader_conf = dict(tfrecord_load_conf or {})
    loader_conf["count_samples"] = False
    unique_frames = []
    seen_item_ids = set()
    original_count = 0

    for input_path in input_paths:
        logger.info(f"Scanning TFRecord item features from {input_path}...")
        loader = TFRecordDataLoader(
            feature_map=item_feature_map,
            data_path=input_path,
            split="item_pool",
            batch_size=batch_size,
            shuffle=False,
            tfrecord_load_conf=loader_conf,
        )
        for batch in loader:
            data = {}
            for col in available_item_cols:
                tensor = batch[col].cpu()
                arr = tensor.numpy()
                if col == item_id_col:
                    if arr.ndim > 1:
                        if arr.shape[1:] != (1,):
                            raise ValueError(
                                f"Item id column '{item_id_col}' must be scalar, got shape {arr.shape}"
                            )
                        arr = arr.reshape(-1)
                    data[col] = arr
                elif arr.ndim > 1:
                    data[col] = [np.asarray(row) for row in arr]
                else:
                    data[col] = arr
            frame = pd.DataFrame(data)
            original_count += len(frame)
            unseen = _take_unseen_items(frame, item_id_col, seen_item_ids)
            if not unseen.empty:
                unique_frames.append(unseen)

    if not unique_frames:
        raise ValueError("No TFRecord rows found for item pool generation")

    item_df = pd.concat(unique_frames, ignore_index=True)
    logger.info(f"Total rows: {original_count} -> Unique items: {len(item_df)}")

    _atomic_write_parquet(item_df, output_path)
    logger.info(f"Saved item pool to {output_path}")
    return item_df


def _item_pool_column_diffs(pool_path: str, required_cols: List[str], logger: logging.Logger):
    """Return missing and stale columns for an existing parquet item pool."""
    try:
        import pyarrow.parquet as pq
        existing_cols = list(pq.ParquetFile(pool_path).schema_arrow.names)
    except Exception as exc:
        logger.debug("Falling back to pandas item pool schema read for %s: %s", pool_path, exc)
        existing_cols = list(pd.read_parquet(pool_path).columns)
    existing_set = set(existing_cols)
    required_set = set(required_cols)
    missing = [col for col in required_cols if col not in existing_set]
    extra = [col for col in existing_cols if col not in required_set]
    return missing, extra


def _item_feature_columns(dataset_config: dict, feature_group_manager) -> List[str]:
    """Resolve the item ID plus FG1/configured item fields in a stable order."""
    item_id_col = dataset_config.get('item_id_col', 'cand_item_id')
    item_feature_cols = []

    if feature_group_manager is not None:
        from ..config.feature_groups import FeatureGroup

        for feat_name, group in feature_group_manager.feature_assignments.items():
            if group == FeatureGroup.FG1 and feat_name != item_id_col:
                item_feature_cols.append(feat_name)

    for feat in dataset_config.get('item_features', []):
        if feat not in item_feature_cols and feat != item_id_col:
            item_feature_cols.append(feat)
    return [item_id_col] + item_feature_cols


def _processed_data_format(data_paths: dict, dataset_config: dict) -> str:
    return str(dataset_config.get(
        'processed_data_format',
        dataset_config.get('data_format', data_paths.get('data_format', 'parquet')),
    )).lower()


def _collect_item_pool_inputs(
    data_paths: dict,
    source_keys: Sequence[str],
    data_format: str,
    logger: logging.Logger,
) -> List[str]:
    input_paths = []
    for key in source_keys:
        path = data_paths.get(key)
        if not path:
            continue
        if data_format in {'tfrecord', 'tf_record'} or os.path.exists(path):
            input_paths.append(path)
        else:
            logger.warning("Item pool source does not exist and will be skipped: %s", path)
    if not input_paths:
        raise ValueError(
            f"No usable item pool source paths for {list(source_keys)}; "
            "do not fall back to another split."
        )
    return input_paths


def _feature_map_signature(feature_map, required_cols: Sequence[str]) -> Dict[str, object]:
    if feature_map is None:
        return {}
    features = getattr(feature_map, 'features', {}) or {}
    return {column: features.get(column) for column in required_cols if column in features}


def _item_pool_cache_payload(
    input_paths: Sequence[str],
    required_cols: Sequence[str],
    data_format: str,
    source_kind: str,
    feature_map,
) -> dict:
    source_stats = []
    for path in input_paths:
        descriptor = {'path': os.path.abspath(path)}
        try:
            stat = os.stat(path)
            descriptor.update({'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        except OSError:
            # TFRecord loaders may receive a glob/path resolved by TensorFlow.
            # Preserve the literal path in the cache key rather than silently
            # treating it as equivalent to another source.
            descriptor['unstatable'] = True
        source_stats.append(descriptor)

    return {
        'version': _ITEM_POOL_CACHE_VERSION,
        'source_kind': source_kind,
        'data_format': data_format,
        'required_columns': list(required_cols),
        'sources': source_stats,
        'feature_map': _feature_map_signature(feature_map, required_cols),
    }


def _item_pool_cache_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _item_pool_metadata_path(pool_path: str) -> str:
    return f"{pool_path}.metadata.json"


def _item_pool_cache_matches(pool_path: str, fingerprint: str, logger: logging.Logger) -> bool:
    metadata_path = _item_pool_metadata_path(pool_path)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as file_handle:
            metadata = json.load(file_handle)
    except FileNotFoundError:
        logger.info("Item pool cache metadata is missing for %s; regenerating.", pool_path)
        return False
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Item pool cache metadata is unreadable for %s: %s", pool_path, exc)
        return False

    if metadata.get('fingerprint') != fingerprint:
        logger.info("Item pool cache fingerprint changed for %s; regenerating.", pool_path)
        return False
    return True


def _write_item_pool_metadata(pool_path: str, payload: dict) -> None:
    metadata_path = _item_pool_metadata_path(pool_path)
    metadata = {
        'fingerprint': _item_pool_cache_fingerprint(payload),
        'payload': payload,
    }
    output_dir = os.path.dirname(metadata_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix='.item_pool_metadata_',
        suffix='.json',
        dir=output_dir,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file_handle:
            json.dump(metadata, file_handle, sort_keys=True)
            file_handle.write('\n')
        os.replace(temporary_path, metadata_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _ensure_item_pool_from_sources(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    output_path: str,
    source_keys: Sequence[str],
    source_kind: str,
    logger: logging.Logger,
    force_regenerate: bool,
    feature_map,
) -> str:
    required_cols = _item_feature_columns(dataset_config, feature_group_manager)
    item_id_col, *item_feature_cols = required_cols
    data_format = _processed_data_format(data_paths, dataset_config)
    input_paths = _collect_item_pool_inputs(data_paths, source_keys, data_format, logger)
    cache_payload = _item_pool_cache_payload(
        input_paths=input_paths,
        required_cols=required_cols,
        data_format=data_format,
        source_kind=source_kind,
        feature_map=feature_map,
    )
    fingerprint = _item_pool_cache_fingerprint(cache_payload)

    if os.path.exists(output_path) and not force_regenerate:
        missing_cols, extra_cols = _item_pool_column_diffs(output_path, required_cols, logger)
        if not missing_cols and not extra_cols and _item_pool_cache_matches(output_path, fingerprint, logger):
            logger.info("%s item pool already exists at %s", source_kind, output_path)
            return output_path
        logger.info(
            "Regenerating %s item pool at %s: missing=%d extra=%d",
            source_kind,
            output_path,
            len(missing_cols),
            len(extra_cols),
        )

    logger.info(
        "Generating %s item pool from %s with features: %s",
        source_kind,
        list(source_keys),
        item_feature_cols,
    )
    if data_format in {'tfrecord', 'tf_record'}:
        batch_size = dataset_config.get('item_pool', {}).get('batch_size', 8192)
        extract_item_corpus_from_tfrecord(
            input_paths=input_paths,
            output_path=output_path,
            item_id_col=item_id_col,
            item_feature_cols=item_feature_cols,
            feature_map=feature_map,
            tfrecord_load_conf=dataset_config.get('tfrecord_load_conf', {}),
            batch_size=batch_size,
            logger=logger,
        )
    else:
        extract_item_corpus(
            input_paths=input_paths,
            output_path=output_path,
            item_id_col=item_id_col,
            item_feature_cols=item_feature_cols,
            logger=logger,
        )
    _write_item_pool_metadata(output_path, cache_payload)
    return output_path


def ensure_item_pool(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    logger: logging.Logger = None,
    force_regenerate: bool = False,
    feature_map=None,
) -> str:
    """
    Ensure the evaluation item pool (valid/test only) exists.

    This pool is intentionally separate from the training negative-sampling
    pool. Do not use it to sample training negatives because it is sourced
    from holdout splits.

    Args:
        data_paths: Dict with keys 'valid_path', 'test_path', 'item_pool_path'
        dataset_config: Dataset configuration dict with 'item_id_col' and optionally 'item_features'
        feature_group_manager: FeatureGroupManager to get FG1 (item) features
        logger: Optional logger instance
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Path to the item pool file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    item_pool_path = data_paths.get('item_pool_path')
    if not item_pool_path:
        raise ValueError("item_pool_path not found in data_paths")
    return _ensure_item_pool_from_sources(
        data_paths=data_paths,
        dataset_config=dataset_config,
        feature_group_manager=feature_group_manager,
        output_path=item_pool_path,
        source_keys=('valid_path', 'test_path'),
        source_kind='evaluation',
        logger=logger,
        force_regenerate=force_regenerate,
        feature_map=feature_map,
    )


def ensure_train_item_pool(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    logger: logging.Logger = None,
    force_regenerate: bool = False,
    feature_map=None,
) -> str:
    """Ensure a train-only item pool for negative sampling.

    The output path may be passed as ``train_item_pool_path``. If omitted, a
    deterministic ``cand_items_train.parquet`` is placed next to the training
    split. This is deliberately a different filename from evaluation and full
    candidate pools so call sites cannot silently reuse a holdout-derived
    cache.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    train_path = data_paths.get('train_path')
    if not train_path:
        raise ValueError("train_path not found in data_paths for train-only item pool")

    train_pool_path = data_paths.get('train_item_pool_path')
    if not train_pool_path:
        train_pool_file = dataset_config.get('item_pool', {}).get(
            'train_file', 'cand_items_train.parquet'
        )
        if not str(train_pool_file).endswith('.parquet'):
            train_pool_file = f"{train_pool_file}.parquet"
        train_pool_path = os.path.join(os.path.dirname(train_path), train_pool_file)

    return _ensure_item_pool_from_sources(
        data_paths=data_paths,
        dataset_config=dataset_config,
        feature_group_manager=feature_group_manager,
        output_path=train_pool_path,
        source_keys=('train_path',),
        source_kind='train',
        logger=logger,
        force_regenerate=force_regenerate,
        feature_map=feature_map,
    )


def ensure_full_item_pool(
    data_paths: dict,
    dataset_config: dict,
    feature_group_manager,
    logger: logging.Logger = None,
    force_regenerate: bool = False,
    feature_map=None,
) -> str:
    """
    Ensure a full item pool (train + valid + test) for all-items evaluation.

    This is not safe for training negative sampling because it includes holdout
    splits. Use :func:`ensure_train_item_pool` for training instead.

    Args:
        data_paths: Dict with keys 'train_path', 'valid_path', 'test_path', 'full_item_pool_path'
        dataset_config: Dataset configuration dict
        feature_group_manager: FeatureGroupManager to get FG1 (item) features
        logger: Optional logger instance
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Path to the full item pool file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    full_item_pool_path = data_paths.get('full_item_pool_path')
    if not full_item_pool_path:
        raise ValueError("full_item_pool_path not found in data_paths")
    return _ensure_item_pool_from_sources(
        data_paths=data_paths,
        dataset_config=dataset_config,
        feature_group_manager=feature_group_manager,
        output_path=full_item_pool_path,
        source_keys=('valid_path', 'test_path', 'train_path'),
        source_kind='full_evaluation',
        logger=logger,
        force_regenerate=force_regenerate,
        feature_map=feature_map,
    )


def validate_item_pool_coverage(
    item_pool_path: str,
    data_path: str,
    item_id_col: str,
    logger: logging.Logger = None
) -> dict:
    """
    Validate that the item pool covers all items in a dataset.

    Args:
        item_pool_path: Path to item pool parquet file
        data_path: Path to data file to validate against
        item_id_col: Column name for item ID
        logger: Optional logger instance

    Returns:
        Dict with validation results:
            - total_data_items: Number of unique items in data
            - total_pool_items: Number of items in pool
            - covered_items: Number of data items covered by pool
            - missing_items: Set of item IDs not in pool
            - coverage_rate: Percentage of data items covered
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load item pool
    item_pool = pd.read_parquet(item_pool_path)
    pool_item_ids = set(item_pool[item_id_col].unique())

    # Load data file
    if data_path.endswith('.parquet'):
        data_df = pd.read_parquet(data_path, columns=[item_id_col])
    else:
        data_df = pd.read_csv(data_path, usecols=[item_id_col])

    data_item_ids = set(data_df[item_id_col].unique())

    # Calculate coverage
    covered_items = data_item_ids & pool_item_ids
    missing_items = data_item_ids - pool_item_ids

    result = {
        'total_data_items': len(data_item_ids),
        'total_pool_items': len(pool_item_ids),
        'covered_items': len(covered_items),
        'missing_items': missing_items,
        'coverage_rate': len(covered_items) / len(data_item_ids) * 100 if data_item_ids else 100.0
    }

    if missing_items:
        logger.warning(f"Item pool coverage: {result['coverage_rate']:.2f}% "
                      f"({len(missing_items)} items missing from pool)")
    else:
        logger.info(f"Item pool coverage: 100% ({len(covered_items)} items)")

    return result
