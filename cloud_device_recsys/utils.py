from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .pipeline.stage_output import StageOutput


_SAFE_IDENTIFIER_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_experiment_id(experiment_id: str) -> str:
    """Validate an output-directory component supplied by the CLI."""
    if not isinstance(experiment_id, str):
        raise ValueError("experiment_id must be a string")
    normalized = experiment_id.strip()
    if (
        not _SAFE_IDENTIFIER_SEGMENT.fullmatch(normalized)
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError(
            "experiment_id must be a single safe identifier containing only "
            "letters, digits, '.', '_' or '-'; path separators and '..' are not allowed."
        )
    return normalized


def validate_pipeline_id(pipeline_id: str) -> str:
    """Validate a config-relative pipeline ID while allowing safe nested config paths."""
    if not isinstance(pipeline_id, str):
        raise ValueError("pipeline_id must be a string")
    normalized = pipeline_id.strip()
    if not normalized or os.path.isabs(normalized) or "\\" in normalized:
        raise ValueError(
            "pipeline_id must be a relative config path with safe identifier segments."
        )
    segments = normalized.split("/")
    if any(segment in {"", ".", ".."} or not _SAFE_IDENTIFIER_SEGMENT.fullmatch(segment)
           for segment in segments):
        raise ValueError(
            "pipeline_id must contain only safe identifier segments separated by '/'; "
            "'.', '..', and path separators other than '/' are not allowed."
        )
    return normalized


def resolve_pipeline_config_path(config_dir: str, pipeline_id: str) -> str:
    """Resolve a validated pipeline ID and enforce containment in ``config_dir``."""
    safe_pipeline_id = validate_pipeline_id(pipeline_id)
    config_root = Path(config_dir).expanduser().resolve()
    pipeline_path = (config_root / f"{safe_pipeline_id}.yaml").resolve()
    try:
        pipeline_path.relative_to(config_root)
    except ValueError as exc:
        raise ValueError(
            f"pipeline_id={pipeline_id!r} resolves outside config directory {config_root}"
        ) from exc
    return str(pipeline_path)


def load_remap_dicts_json(path: str) -> Dict[str, Dict[int, int]]:
    """Load a data-only vocabulary remap artifact with a strict schema."""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid remap JSON artifact {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Remap JSON must contain an object keyed by feature name")

    result: Dict[str, Dict[int, int]] = {}
    for feature_name, mapping in payload.items():
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("Remap JSON contains an invalid feature name")
        if not isinstance(mapping, dict):
            raise ValueError(f"Remap for feature {feature_name!r} must be an object")
        parsed_mapping: Dict[int, int] = {}
        for old_id, new_id in mapping.items():
            if isinstance(new_id, bool):
                raise ValueError(f"Remap for feature {feature_name!r} contains a boolean ID")
            try:
                old_id_int = int(old_id)
                new_id_int = int(new_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Remap for feature {feature_name!r} must use integer IDs"
                ) from exc
            if old_id_int < 0 or new_id_int < 0:
                raise ValueError(f"Remap for feature {feature_name!r} contains a negative ID")
            parsed_mapping[old_id_int] = new_id_int
        result[feature_name] = parsed_mapping
    return result


def filter_feature_map(feature_map, fg_manager, allowed_feature_groups, use_feature_encoder=False):
    """
    Create a new FeatureMap with only features belonging to allowed feature groups.

    This function filters features based on their assigned feature group, keeping only
    those that belong to the specified allowed groups. Labels and special columns
    are always preserved.

    Args:
        feature_map: FuxiCTR FeatureMap object to filter
        fg_manager: FeatureGroupManager with feature assignments
        allowed_feature_groups: List of allowed FeatureGroup enums (e.g., [FeatureGroup.FG1, FeatureGroup.FG2])
        use_feature_encoder: If true, use feature encoder
    Returns:
        A new FeatureMap containing only the allowed features (deep copy of original)
    """
    import copy
    from collections import OrderedDict

    new_fm = copy.deepcopy(feature_map)
    new_features = OrderedDict()
    use_features = []
    user_features = fg_manager.get_user_features()
    item_col = feature_map.dataset_config.get('item_id_col')
    for name, spec in feature_map.features.items():
        # Check if feature belongs to allowed groups
        group = fg_manager.feature_assignments.get(name)
        is_allowed = False
        for allowed_grp in allowed_feature_groups:
            # Compare enum members directly if possible, or string representation
            if group == allowed_grp or str(group) == str(allowed_grp):
                is_allowed = True
                if name in user_features:
                    spec['source'] = 'user'
                else:
                    spec['source'] = 'item'
                break
        impression_col = feature_map.dataset_config.get('impression_id_col', 'impression_id')
        # Always keep label, score, and special columns (needed for training/indexing)
        if name in [impression_col, 'group_id', 'click', 'clk', 'label'] + feature_map.labels:
            is_allowed = True
        if item_col and name == item_col:
            is_allowed = True
            spec['source'] = 'item'

        if is_allowed:
            if not use_feature_encoder:
                spec['feature_encoder'] = None
            new_features[name] = spec
            use_features.append(name)

    new_fm.features = new_features
    new_fm.use_features = use_features
    # Re-set column indices after filtering
    new_fm.set_column_index()
    new_fm.num_fields = len(new_fm.features)
    return new_fm


def setup_logging(output_dir) -> None:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to ensure we control output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    for noisy_logger in ("h5py", "h5py._conv"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Saving to {log_file}")


def get_data_dir(dataset_config, dataset_id=None):
    """Get the data directory from dataset configuration."""
    if 'processed_data_root' in dataset_config:
        data_dir = dataset_config['processed_data_root']
    elif dataset_config.get('feature_map_path'):
        data_dir = os.path.dirname(os.path.expanduser(os.path.expandvars(dataset_config['feature_map_path'])))
    else:
        data_dir = os.path.join(dataset_config.get('data_root', './data'), dataset_id)
    return data_dir


def resolve_config_path(path: str) -> str:
    """Expand environment variables and user markers in configured paths."""
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(path)))


def create_debug_dataset(input_path: str, n_rows: int, output_path: str, logger) -> str:
    """
    Create a small debug dataset from the original parquet file (memory efficient).

    Args:
        input_path: Path to the original parquet file
        n_rows: Number of rows to sample
        output_path: Path to save the debug dataset
        logger: Logger instance

    Returns:
        Path to the debug dataset (or original if creation failed)
    """
    import pyarrow.parquet as pq

    if not os.path.exists(output_path):
        logger.info(f"Creating debug dataset (n={n_rows}) from {input_path}...")
        try:
            # Read efficiently using pyarrow
            pf = pq.ParquetFile(input_path)
            # Read first row group (usually enough for debug)
            table = pf.read_row_group(0)
            df = table.to_pandas()

            # Ensure n_rows doesn't exceed dataframe length
            sample_n = min(n_rows, len(df))
            df_sample = df.head(sample_n)
            df_sample.to_parquet(output_path)
            logger.info(f"Saved debug dataset to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to create debug dataset: {e}. Using original.")
            return input_path
    else:
        logger.info(f"Using existing debug dataset: {output_path}")
    return output_path


def get_data_paths(dataset_config: dict, pipeline_config: dict, logger):
    """
    Get common data paths for all stages.

    Args:
        dataset_config: Dataset configuration dict
        pipeline_config: Pipeline configuration dict
        logger: Logger instance

    Returns:
        dict with keys: data_dir, data_format, debug_n_rows, debug_dir,
                        processed_data_root, train_path, valid_path, test_path, item_pool_path
    """
    data_dir = get_data_dir(dataset_config, pipeline_config['dataset_id'])
    data_format = dataset_config.get(
        'processed_data_format',
        dataset_config.get('data_format', 'parquet')
    )

    # Check for debug mode
    debug_n_rows = pipeline_config.get('debug', {}).get('n_rows')
    if debug_n_rows is not None:
        logger.info(f"Debug mode enabled: using {debug_n_rows} rows.")
        debug_dir = os.path.join(data_dir, f"debug_{debug_n_rows}")
        os.makedirs(debug_dir, exist_ok=True)
        processed_data_root = debug_dir
    else:
        debug_dir = None
        processed_data_root = dataset_config.get('processed_data_root', data_dir)

    # Build standard paths
    train_path = resolve_config_path(
        dataset_config.get('train_data')
        or os.path.join(processed_data_root, f'train.{data_format}')
    )
    valid_path = resolve_config_path(
        dataset_config.get('valid_data')
        or os.path.join(processed_data_root, f'valid.{data_format}')
    )
    test_path = resolve_config_path(
        dataset_config.get('test_data')
        or os.path.join(processed_data_root, f'test.{data_format}')
    )

    # Candidate item pool path (actual source splits are controlled by dataset_config.item_pool.source)
    item_pool_config = dataset_config.get('item_pool', {})
    item_pool_file = item_pool_config.get('file', 'cand_item_list')
    if item_pool_config.get('path'):
        item_pool_path = resolve_config_path(item_pool_config['path'])
    elif str(item_pool_file).endswith('.parquet'):
        item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), item_pool_file)
    else:
        item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), f'{item_pool_file}.parquet')

    # Full item pool path (train+valid+test items - for all-items evaluation).
    # Explicit training negatives use the distinct train-only pool.
    full_item_pool_path = os.path.join(dataset_config.get('processed_data_root', data_dir), 'cand_items_all.parquet')

    return {
        'data_dir': data_dir,
        'data_format': data_format,
        'debug_n_rows': debug_n_rows,
        'debug_dir': debug_dir,
        'processed_data_root': processed_data_root,
        'train_path': train_path,
        'valid_path': valid_path,
        'test_path': test_path,
        'item_pool_path': item_pool_path,
        'item_pool_format': 'parquet',
        'full_item_pool_path': full_item_pool_path,
        'item_pool_file': item_pool_file,
        'tfrecord_load_conf': dataset_config.get('tfrecord_load_conf', {}),
    }


def prepare_debug_paths(paths: dict, dataset_config: dict, logger) -> dict:
    """
    Create debug datasets if in debug mode and update paths accordingly.

    Args:
        paths: Dict from get_data_paths()
        dataset_config: Dataset configuration dict
        logger: Logger instance

    Returns:
        Updated paths dict with debug dataset paths
    """
    debug_n_rows = paths['debug_n_rows']
    debug_dir = paths['debug_dir']
    data_format = paths['data_format']

    if debug_n_rows is not None and data_format == 'parquet':
        original_data_root = dataset_config.get('processed_data_root', paths['data_dir'])
        original_train = os.path.join(original_data_root, f'train.{data_format}')
        original_valid = os.path.join(original_data_root, f'valid.{data_format}')
        original_test = os.path.join(original_data_root, f'test.{data_format}')
        # original_item_pool = os.path.join(original_data_root, f"{paths['item_pool_file']}.parquet")

        # Create debug datasets and update paths
        paths['train_path'] = create_debug_dataset(original_train, debug_n_rows, os.path.join(debug_dir, 'train.parquet'), logger)
        paths['valid_path'] = create_debug_dataset(original_valid, debug_n_rows, os.path.join(debug_dir, 'valid.parquet'), logger)
        paths['test_path'] = create_debug_dataset(original_test, debug_n_rows, os.path.join(debug_dir, 'test.parquet'), logger)
        # paths['item_pool_path'] = create_debug_dataset(original_item_pool, debug_n_rows, os.path.join(debug_dir, f"{paths['item_pool_file']}.parquet"), logger)

    return paths


# =========================================================================
# Shared Inference Utilities for Preranking/Reranking Stages
# =========================================================================

def build_inference_batch_for_candidates(
    user_features: Dict[str, Any],
    candidate_item_ids: List[Any],
    item_features_df: pd.DataFrame,
    feature_map,
    item_id_col: str = 'cand_item_id',
    fallback_item_features: Dict[str, Any] = None,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Build inference batch for a single query's candidate items.

    For each candidate item, replicates user features and looks up item features
    from the item pool DataFrame.

    Args:
        user_features: User feature dict {feature_name: value} for this query
        candidate_item_ids: List of candidate item IDs to score
        item_features_df: DataFrame with item features, indexed by item_id
        feature_map: FuxiCTR FeatureMap for dtype info
        item_id_col: Column name for item ID
        fallback_item_features: Fallback features for first item (GT) if missing from pool

    Returns:
        Tuple of:
            - batch_dict: {feature_name: np.ndarray} ready for model inference
            - valid_indices: Indices of items that were successfully built
    """
    num_items = len(candidate_item_ids)

    # 1. Replicate user features for all items
    user_feature_batch = {}
    for k, val in user_features.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
            user_feature_batch[k] = np.tile(val, (num_items, 1))
        else:
            user_feature_batch[k] = np.full((num_items,), val)

    # 2. Look up item features
    item_feature_cols = list(item_features_df.columns)
    item_feature_batch = {k: [] for k in item_feature_cols}
    if item_id_col not in item_feature_batch and item_id_col in feature_map.features:
        item_feature_batch[item_id_col] = []

    valid_indices = []

    for idx, item_id in enumerate(candidate_item_ids):
        if item_id in item_features_df.index:
            valid_indices.append(idx)
            row = item_features_df.loc[item_id]
            for col in item_feature_cols:
                item_feature_batch[col].append(row[col])
            if item_id_col in item_feature_batch:
                item_feature_batch[item_id_col].append(item_id)
        elif idx == 0 and fallback_item_features is not None:
            # Use fallback for GT item (first item)
            valid_indices.append(idx)
            for col in item_feature_cols:
                if col in fallback_item_features:
                    item_feature_batch[col].append(fallback_item_features[col])
                else:
                    item_feature_batch[col].append(0)
            if item_id_col in item_feature_batch:
                item_feature_batch[item_id_col].append(item_id)
        # else: skip this item

    # 3. Convert to numpy arrays
    for k in item_feature_batch:
        item_feature_batch[k] = np.array(item_feature_batch[k])

    # 4. Slice user features to match valid items
    num_valid = len(valid_indices)
    for k in user_feature_batch:
        user_feature_batch[k] = user_feature_batch[k][:num_valid]

    # 5. Merge user and item features
    batch_dict = {**user_feature_batch, **item_feature_batch}

    return batch_dict, valid_indices


def batch_to_tensors(
    batch_dict: Dict[str, np.ndarray],
    feature_map,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy batch to PyTorch tensors based on feature types.

    Args:
        batch_dict: {feature_name: np.ndarray}
        feature_map: FuxiCTR FeatureMap for dtype info
        device: Target device for tensors

    Returns:
        {feature_name: torch.Tensor}
    """
    tensor_batch = {}
    for k, v in batch_dict.items():
        if k not in feature_map.features:
            continue
        ftype = feature_map.features[k]['type']
        if ftype == 'sequence':
            tensor_batch[k] = torch.tensor(v, dtype=torch.long).to(device)
        elif ftype == 'categorical':
            tensor_batch[k] = torch.tensor(v, dtype=torch.long).to(device)
        else:
            tensor_batch[k] = torch.tensor(v, dtype=torch.float).to(device)
    return tensor_batch

def save_stage_output(stage_output: StageOutput, output_dir: str, prefix: str,
                      logger: logging.Logger = None) -> str:
    """
    Save a StageOutput object to disk using Parquet format for efficiency.

    Args:
        stage_output: StageOutput object to save
        output_dir: Directory to save the output file
        prefix: Prefix for the output filename (e.g., 'retrieval_valid', 'preranking_test')
        logger: Optional logger instance

    Returns:
        Path to the saved directory (Parquet format)
    """
    import time
    if stage_output is None:
        raise ValueError('StageOutput object must not be None')

    os.makedirs(output_dir, exist_ok=True)

    # Use Parquet format (directory-based)
    dirname = f"{prefix}_stage_output"
    dirpath = os.path.join(output_dir, dirname)

    start_time = time.time()
    stage_output.save_parquet(dirpath)
    elapsed = time.time() - start_time

    if logger:
        logger.info(f"Saved {stage_output.stage_name} output ({stage_output.get_num_requests()} requests, "
                    f"{stage_output.get_total_candidates()} total candidates) to {dirpath} in {elapsed:.2f}s")

    return dirpath

def load_stage_output(filepath: str, logger: logging.Logger = None) -> Optional[StageOutput]:
    """
    Load a StageOutput object from a Parquet directory.

    Args:
        filepath: Path to the Parquet directory
        logger: Optional logger instance

    Returns:
        StageOutput object, or None if loading fails
    """
    import time
    from .pipeline.stage_output import StageOutput  # Runtime import to avoid circular dependency
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Path {filepath} not found")
    if not os.path.isdir(filepath) or not os.path.isfile(os.path.join(filepath, 'candidates.parquet')):
        raise ValueError(
            "Only Parquet stage-output directories are supported. Refusing legacy pickle "
            f"artifact at {filepath!r}; regenerate it with save_stage_output()."
        )

    try:
        start_time = time.time()
        stage_output = StageOutput.load_parquet(filepath)
        fmt = "Parquet"

        elapsed = time.time() - start_time

        if logger:
            logger.info(f"Loaded {stage_output.stage_name} output ({fmt}) from {filepath}: "
                        f"{stage_output.get_num_requests()} requests, "
                        f"{stage_output.get_total_candidates()} total candidates in {elapsed:.2f}s")
        return stage_output
    except Exception as e:
        if logger:
            logger.error(f"Failed to load stage output from {filepath}: {e}")
        return None

def load_stage_outputs_from_dir(
        output_dir: str,
        prev_stage_name: str,
        logger: logging.Logger = None,
        load_test: bool = True
) -> Tuple[Optional[StageOutput], Optional[StageOutput]]:
    """
    Load valid and test stage outputs from a directory in Parquet format.

    Args:
        output_dir: Directory containing stage output files (e.g., './outputs/exp_xxx/stage_outputs')
        prev_stage_name: Name of the previous stage (e.g., 'retrieval' or 'preranking')
        logger: Optional logger instance
        load_test: If True, load test stage outputs

    Returns:
        Tuple of (valid_output, test_output), both can be None if loading fails
    """
    test_output = None
    if load_test:
        test_parquet = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output")
        test_legacy = os.path.join(output_dir, f"{prev_stage_name}_test_stage_output.pkl")
        if not os.path.isdir(test_parquet) and os.path.exists(test_legacy):
            raise ValueError(
                f"Refusing legacy pickle stage output at {test_legacy!r}; regenerate it in Parquet format."
            )
        test_output = load_stage_output(test_parquet, logger)
    valid_parquet = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output")
    valid_legacy = os.path.join(output_dir, f"{prev_stage_name}_valid_stage_output.pkl")
    if not os.path.isdir(valid_parquet) and os.path.exists(valid_legacy):
        raise ValueError(
            f"Refusing legacy pickle stage output at {valid_legacy!r}; regenerate it in Parquet format."
        )
    valid_output = load_stage_output(valid_parquet, logger)
    return valid_output, test_output


def enrich_stage_output_user_features(
        stage_output: StageOutput,
        data_path: str,
        fg_manager,
        impression_id_col: str = 'impression_id',
        logger: logging.Logger = None
) -> StageOutput:
    """
    Enrich StageOutput with missing FG3 user features from original data files.

    This provides backward compatibility for stage_output files that were saved
    without FG3 features (due to filtered feature_map in DataLoader).

    Args:
        stage_output: StageOutput to enrich
        data_path: Path to original data file (parquet/csv) with complete features
        fg_manager: FeatureGroupManager with feature assignments
        impression_id_col: Column name for request IDs (default: 'impression_id')
        logger: Optional logger instance

    Returns:
        StageOutput with enriched user features (modified in-place)
    """
    if stage_output is None:
        return None

    if logger is None:
        logger = logging.getLogger(__name__)

    user_features_df = stage_output.user_features_df
    if user_features_df is None or len(user_features_df) == 0:
        logger.warning("No user features to enrich")
        return stage_output

    # Get all user features (FG2 + FG3) that should be present
    all_user_features = fg_manager.get_user_features()
    existing_columns = set(user_features_df.columns)
    missing_features = all_user_features - existing_columns

    if not missing_features:
        logger.info("All expected user features already present, no enrichment needed")
        return stage_output

    logger.info(f"Found {len(missing_features)} missing user features: {sorted(missing_features)}")

    # Load source data
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}, cannot enrich features")
        return stage_output

    try:
        if data_path.endswith('.parquet'):
            source_df = pd.read_parquet(data_path)
            if impression_id_col not in source_df.columns:
                source_df[impression_id_col] = source_df.index  # Assume index is impression_id if column missing
        elif data_path.endswith('.csv'):
            source_df = pd.read_csv(data_path)
        elif data_path.endswith('.tfrecord') or data_path.endswith('.tfrecord.gz'):
            logger.warning(
                "Cannot enrich missing user features directly from TFRecord path %s. "
                "Regenerate retrieval stage outputs with the full feature map.",
                data_path,
            )
            return stage_output
        else:
            logger.warning("Unsupported enrichment data format: %s", data_path)
            return stage_output

        # Check which columns actually exist in source
        available_missing = [f for f in missing_features if f in source_df.columns]
        if not available_missing:
            logger.warning(f"None of the missing features found in source data: {data_path}")
            return stage_output

        logger.info(f"Loading {len(available_missing)} features from source: {sorted(available_missing)}")

        # Deduplicate source data by impression_id (keep first occurrence)
        cols_to_keep = [impression_id_col] + available_missing
        source_df = source_df[cols_to_keep].drop_duplicates(subset=[impression_id_col], keep='first')

        # Merge with existing user features
        merge_key = 'request_id' if 'request_id' in user_features_df.columns else impression_id_col
        source_merge_key = impression_id_col

        # Rename source key if needed
        if merge_key != source_merge_key and merge_key in user_features_df.columns:
            source_df = source_df.rename(columns={source_merge_key: merge_key})

        # Perform left merge to add missing features
        enriched_df = user_features_df.merge(
            source_df,
            on=merge_key,
            how='left'
        )

        # Replace the user features DataFrame
        stage_output._user_features_df = enriched_df

        logger.info(f"Successfully enriched user features: {len(user_features_df)} -> {len(enriched_df)} rows, "
                   f"added {len(available_missing)} columns")

    except Exception as e:
        logger.error(f"Failed to enrich user features: {e}")

    return stage_output

def parse_optional_bool(value):
    """Parse CLI bools while still allowing an omitted value via nargs='?'."""
    if value is None or isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_pipeline_args():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description='Cloud-Device Recommendation Pipeline')
    parser.add_argument('--config', type=str, default='./config',
                       help='Configuration directory')
    parser.add_argument('--pipeline_id', type=str, default='default',
                       help='Pipeline configuration ID (default uses ConfigParser fallback)')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Dataset ID from dataset_config.yaml')
    parser.add_argument('--mode', type=str, default='full',
                       choices=[
                           'full',
                           'retrieval',
                           'preranking',
                           'reranking',
                           'save_retrieval_outputs',
                           'save_preranking_outputs',
                           'save_reranking_outputs',
                       ],
                       help='Execution mode')
    parser.add_argument('--stage', type=str, default=None,
                       choices=['retrieval', 'preranking', 'reranking'],
                       help='Stage name for train/evaluate mode')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--prev_output_path', type=str, default=None,
                       help='Path to directory containing previous stage outputs (stage_outputs/). '
                            'Will load Parquet {stage}_valid_stage_output and {stage}_test_stage_output directories')
    parser.add_argument('--retrieval_output_path', type=str, default=None,
                       help='Path to directory containing retrieval stage outputs (stage_outputs/). '
                            'Used by reranking mode for evaluation scope alignment.')
    parser.add_argument('--experiment_id', type=str, default=None,
                       help='Unique identifier for the current experiment run')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')

    # Validation control arguments
    parser.add_argument('--run_retrieval_test', type=int, default=0,
                       help='Whether to run retrieval test evaluation (1=yes, 0=no)')
    parser.add_argument('--run_preranking_test', type=int, default=0,
                       help='Whether to run preranking test evaluation (1=yes, 0=no)')
    parser.add_argument('--run_reranking_test', type=int, default=0,
                       help='Whether to run reranking test evaluation (1=yes, 0=no)')

    parser.add_argument('--n_rows', type=int, default=None,
                       help='Override debug n_rows (set to small number for quick testing)')
    parser.add_argument('--save_stage_outputs', nargs='?', const='1', default=None,
                        type=parse_optional_bool,
                        help='Save intermediate stage outputs (StageOutput) to disk for later reuse. '
                             'If omitted, defaults to output.save_intermediate from the pipeline config.')
    parser.add_argument('--model_weights_path', type=str, default=None,
                        help='Path to pre-trained model weights (.model file). '
                             'Used with save_*_outputs modes to skip training and export outputs from the best checkpoint.')
    return parser.parse_args()
