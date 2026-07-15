#!/usr/bin/env python
# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Main Pipeline Runner

Entry point for running the cloud-device recommendation pipeline.
Supports both full pipeline execution and individual stage runs.

Usage:
    # Full pipeline
    python run_pipeline.py --config ./config --mode full --gpu 0

    # Individual stages
    python run_pipeline.py --config ./config --mode retrieval --gpu 0
    python run_pipeline.py --config ./config --mode preranking --gpu 0
    python run_pipeline.py --config ./config --mode reranking --gpu 0

"""

import os
import copy
import sys
import logging
import json
import math
import re
import yaml
import numpy as np
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything

from cloud_device_recsys.config.feature_groups import FeatureGroupManager, FeatureGroup
from cloud_device_recsys.pipeline import RetrievalStage, PrerankingStage, RerankingStage
from cloud_device_recsys.pipeline.stage_output import StageOutput
from cloud_device_recsys.utils import (
    setup_logging, get_data_dir, get_data_paths,
    prepare_debug_paths, parse_pipeline_args,
    save_stage_output, load_stage_outputs_from_dir,
    enrich_stage_output_user_features, load_remap_dicts_json,
    resolve_pipeline_config_path, validate_experiment_id, validate_pipeline_id,
)
from cloud_device_recsys.data.item_pool import (
    ensure_item_pool,
    ensure_full_item_pool,
    ensure_train_item_pool,
)
from cloud_device_recsys.data.positive_data import get_train_path_for_mode
from cloud_device_recsys.data.tfrecord_feature_map import ensure_fuxictr_feature_map
from cloud_device_recsys.data.dataset_statistics import write_dataset_statistics
from cloud_device_recsys.config.config_parser import ConfigParser
import pandas as pd

POPULARITY_BASELINE_COLUMN = "__popularity__"

TARGET_CONDITIONED_USER_COLUMN_EXACT = set()

TARGET_CONDITIONED_USER_COLUMN_PATTERNS = ()


def load_offline_remap_dicts(data_dir, logger):
    """Load only validated JSON remaps; legacy pickle artifacts are executable."""
    remap_json_path = os.path.join(data_dir, 'remap_dict.json')
    legacy_remap_path = os.path.join(data_dir, 'remap_dict.pkl')
    if os.path.isfile(remap_json_path):
        remap_dicts = load_remap_dicts_json(remap_json_path)
        logger.info(
            "[VocabPruner] Offline Mode: Loaded validated remap JSON from %s.",
            remap_json_path,
        )
        return remap_dicts
    if os.path.exists(legacy_remap_path):
        raise RuntimeError(
            "Refusing unsafe legacy pickle remap artifact at "
            f"{legacy_remap_path}. Regenerate it as schema-validated remap_dict.json."
        )
    logger.warning(
        "[VocabPruner] Offline mode enabled but remap_dict.json was not found in %s. "
        "Evaluation metrics may break.",
        data_dir,
    )
    return None

def build_feature_group_manager(config: dict) -> FeatureGroupManager:
    """Build and configure feature group manager"""
    manager = FeatureGroupManager()

    # Load custom assignments if provided
    if 'feature_groups' in config:
        for fg_name, features in config['feature_groups'].items():
            fg = FeatureGroup.from_string(fg_name)
            for feature in features:
                manager.assign_feature(feature, fg)

    return manager

def create_stages(
        feature_map: FeatureMap,
        feature_group_manager: FeatureGroupManager,
        config: dict,
        output_dir,
        gpu: int = -1
) -> dict:
    """Create pipeline stages based on configuration"""
    stages = {}
    stages_config = config.get('stages', config)

    # Map stage names to their classes and special parameters
    STAGE_REGISTRY = {
        'retrieval': {
            'class': RetrievalStage,
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k'
            }
        },
        'preranking': {
            'class': PrerankingStage,
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k',
                'use_diversity': 'use_diversity'
            }
        },
        'reranking': {
            'class': RerankingStage,
            'kwargs_map': {
                'features': 'allowed_feature_groups',
                'top_k': 'top_k',
            }
        },
    }

    for stage_name, stage_info in STAGE_REGISTRY.items():
        if stage_name in stages_config and stages_config[stage_name].get('enabled', False):
            stage_config = stages_config[stage_name]

            # Prepare model_params
            model_params = stage_config.get('model_params', {}).copy()  # Use .copy() to avoid modifying original config
            model_params['gpu'] = gpu
            model_params['model'] = stage_config.get('model')  # Pass model name from config
            if 'metrics' in stage_config:
                model_params['metrics'] = stage_config['metrics']
            # Explicitly pass save_fp16 down to the stage through model_params
            vocab_pruning_config = config.get('vocab_pruning', {})
            model_params['save_fp16'] = vocab_pruning_config.get('save_fp16', False)

            # Prepare stage-specific keyword arguments for the constructor
            stage_kwargs = {
                'feature_map': copy.deepcopy(feature_map),
                'feature_group_manager': feature_group_manager,
                'model_params': model_params,
                'output_dir': os.path.join(output_dir, stage_name),
            }

            # Add extra parameters dynamically
            for config_key, class_param in stage_info['kwargs_map'].items():
                if config_key in stage_config:
                    if config_key == 'features':  # Special handling for 'features' to map to enum
                        stage_kwargs[class_param] = [getattr(FeatureGroup, f) for f in stage_config['features']]
                    elif config_key == 'distillation':  # Special handling for 'distillation' to extract 'enabled'
                        stage_kwargs[class_param] = stage_config[config_key].get('enabled', False)
                    else:
                        stage_kwargs[class_param] = stage_config[config_key]

            # Lazy Instantiation: Return a factory method
            # Use default arguments to capture loop variables (cls, kwargs) correctly!
            stages[stage_name] = lambda cls=stage_info['class'], kwargs=stage_kwargs: cls(**kwargs)

    return stages

def _prepare_stage_data_loaders(feature_map, stage_config: dict, paths: dict,
                                create_train=True, create_test=True, shuffle_train=True,
                                create_item_loader=False, item_feature_map=None,
                                num_negatives: int = 0, label_col: str = "label",
                                force_positive_train_data: bool = False,
                                logger=None):
    """
    Create data loaders for a pipeline stage.

    Args:
        feature_map: FeatureMap instance
        stage_config: Stage-specific config (e.g., pipeline_config['stages']['retrieval'])
        paths: Dict from get_data_paths()
        create_train: Whether to create train/valid loader
        create_test: Whether to create test loader
        shuffle_train: Whether to shuffle training data
        create_item_loader: Whether to create item pool loader (for retrieval)
        item_feature_map: Feature map for item pool (required if create_item_loader=True)
        num_negatives: Number of negatives per positive (0 = pointwise, >0 = pairwise)
        label_col: Label column name for filtering positive samples
        force_positive_train_data: Whether to force positive-only train data even when num_negatives == 0
        logger: Logger instance for positive data creation

    Returns:
        dict with keys: train_loader, test_loader, item_loader (based on flags)
    """
    batch_size = stage_config.get('training', stage_config).get('batch_size', 4096)
    data_format = paths['data_format']
    loader_kwargs = {}
    if data_format in {'tfrecord', 'tf_record'}:
        loader_kwargs['tfrecord_load_conf'] = paths.get('tfrecord_load_conf', {})

    result = {}

    if create_train:
        # Select training data path based on training mode
        train_path = get_train_path_for_mode(
            train_path=paths['train_path'],
            num_negatives=num_negatives,
            label_col=label_col,
            logger=logger,
            force_positive_only=force_positive_train_data,
            data_format=data_format,
            tfrecord_load_conf=paths.get('tfrecord_load_conf', {}),
        )

        # train_fm = copy.deepcopy(feature_map)
        # if "impression_id" in train_fm.labels:
        #     train_fm.labels.remove("impression_id")
        result['train_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='train',
            train_data=train_path,
            batch_size=batch_size,
            shuffle=shuffle_train,
            data_format=data_format,
            **loader_kwargs
        )
        result['valid_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='test',
            test_data=paths['valid_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=data_format,
            **loader_kwargs
        )

    if create_test:
        result['test_loader'] = RankDataLoader(
            feature_map=feature_map,
            stage='test',
            test_data=paths['test_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=data_format,
            **loader_kwargs
        )

    if create_item_loader and item_feature_map is not None:
        result['item_loader'] = RankDataLoader(
            feature_map=item_feature_map,
            stage='train',  # Acts as feed generator
            train_data=paths['item_pool_path'],
            batch_size=batch_size,
            shuffle=False,
            data_format=paths.get('item_pool_format', 'parquet')
        )

    return result


def _annotate_loader_sample_count(data_gen, split_name: str, dataset_config: dict, logger=None):
    """Use cached dataset statistics to expose TFRecord sample/batch counts in logs."""
    if data_gen is None:
        return
    if getattr(data_gen, "num_samples", -1) is not None and getattr(data_gen, "num_samples", -1) >= 0:
        return
    stats_path = dataset_config.get("_dataset_statistics_path")
    if not stats_path or not os.path.isfile(stats_path):
        return
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    split_stats = (stats.get("splits") or {}).get(split_name) or {}
    if split_stats.get("truncated"):
        return
    rows = split_stats.get("rows")
    try:
        rows = int(rows)
    except (TypeError, ValueError):
        return
    batch_size = int(getattr(data_gen, "batch_size", 0) or 0)
    if rows < 0 or batch_size <= 0:
        return
    data_gen.num_samples = rows
    data_gen.num_batches = int(math.ceil(rows / batch_size))
    if logger is not None:
        logger.info(
            "Annotated %s loader sample count from dataset statistics: samples=%d, batch_size=%d, batches=%d",
            split_name,
            data_gen.num_samples,
            batch_size,
            data_gen.num_batches,
        )


def _config_name(value, default=None):
    if isinstance(value, dict):
        return value.get('name', default)
    if value in (None, ''):
        return default
    return str(value)


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _to_non_negative_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _training_negative_config(model_params):
    model_params = model_params or {}
    num_negatives = _to_non_negative_int(model_params.get('num_negatives', 0), 0)
    use_in_batch_negatives = _to_bool(model_params.get('use_in_batch_negatives'), False)
    use_diversity_loss = _to_bool(model_params.get('use_diversity_loss'), False)
    training_mode = str(model_params.get('training_mode', 'auto') or 'auto').strip().lower()
    use_hybrid_loss = (
        training_mode == 'hybrid'
        or _to_bool(model_params.get('hybrid_loss'), False)
    )
    diversity_num_negatives = _to_non_negative_int(
        model_params.get('diversity_num_negatives', num_negatives),
        num_negatives,
    )
    training_num_negatives = max(
        num_negatives,
        diversity_num_negatives if use_diversity_loss else 0,
    )
    force_positive_train_data = (use_in_batch_negatives or training_num_negatives > 0) and not use_hybrid_loss
    loader_num_negatives = 0 if use_hybrid_loss else training_num_negatives
    return {
        'num_negatives': num_negatives,
        'use_in_batch_negatives': use_in_batch_negatives,
        'use_diversity_loss': use_diversity_loss,
        'use_hybrid_loss': use_hybrid_loss,
        'diversity_num_negatives': diversity_num_negatives,
        'training_num_negatives': training_num_negatives,
        'loader_num_negatives': loader_num_negatives,
        'force_positive_train_data': force_positive_train_data,
    }


def _ensure_train_negative_pool(
        stage_name: str,
        stage_config: dict,
        paths: dict,
        dataset_config: dict,
        feature_group_manager,
        feature_map,
        logger,
) -> str:
    """Return the training-only item corpus used for explicit negatives.

    Older configs exposed ``candidate`` and ``full`` options.  Both pools are
    derived from validation/test splits and therefore cannot be used to sample
    training negatives.  Keep accepting those values so existing experiment
    configs fail safe instead of silently reverting to a holdout pool.
    """
    requested_pool = str((stage_config or {}).get('neg_sampling_pool', 'train')).strip().lower()
    if requested_pool not in {'train', 'training', 'train_only'}:
        logger.warning(
            "[%s] Ignoring neg_sampling_pool=%r: explicit training negatives "
            "must use the train-only item pool.",
            stage_name,
            requested_pool,
        )
    return ensure_train_item_pool(
        data_paths=paths,
        dataset_config=dataset_config,
        feature_group_manager=feature_group_manager,
        logger=logger,
        feature_map=feature_map,
    )


def _get_train_vocab_scan_paths(data_dir: str) -> list:
    """Choose the only split allowed to define runtime-pruned vocabularies."""
    train_positive_path = os.path.join(data_dir, 'train_positive.parquet')
    train_full_path = os.path.join(data_dir, 'train.parquet')
    train_scan_path = train_positive_path if os.path.exists(train_positive_path) else train_full_path
    if not os.path.exists(train_scan_path):
        raise FileNotFoundError(
            "Runtime vocabulary pruning requires a training parquet split; "
            f"checked {train_positive_path!r} and {train_full_path!r}."
        )
    return [train_scan_path]


def _as_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.copy()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _row_value(array, row_idx, feature_type=None):
    value = array[row_idx]
    if feature_type == 'sequence':
        return np.asarray(value).copy()
    return _as_scalar(value)


def _safe_json_value(value):
    value = _as_scalar(value)
    if isinstance(value, np.ndarray):
        return [_safe_json_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _normalize_id_for_compare(value):
    value = _as_scalar(value)
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _sequence_values_for_compare(value):
    value = _as_scalar(value)
    if isinstance(value, np.ndarray):
        raw_values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                raw_values = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                raw_values = [part for part in re.split(r"[,;\s]+", stripped.strip("[]")) if part]
        else:
            raw_values = [part for part in re.split(r"[,;\s]+", stripped) if part]
    else:
        raw_values = [value]
    return {_normalize_id_for_compare(item) for item in raw_values if _normalize_id_for_compare(item)}


def _sequence_feature_columns(feature_map, columns):
    sequence_cols = []
    for col in columns:
        spec = (getattr(feature_map, 'features', {}) or {}).get(col, {}) or {}
        if spec.get('type') == 'sequence' or spec.get('max_len'):
            sequence_cols.append(col)
    return sequence_cols


def _configured_name_list(value):
    if value in (None, ''):
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).replace(';', ',').split(',')
    return [str(item).strip() for item in values if str(item).strip()]


def _is_target_conditioned_user_column(col):
    col = str(col)
    if col in TARGET_CONDITIONED_USER_COLUMN_EXACT:
        return True
    return any(re.search(pattern, col) for pattern in TARGET_CONDITIONED_USER_COLUMN_PATTERNS)


def detect_candidate_conditioned_user_columns(
        user_columns,
        item_features_df,
        feature_map,
        dataset_config,
        fg_manager=None,
        extra_columns=None):
    item_id_col = _config_name(dataset_config.get('item_id_col'), 'cand_item_id')
    label_col = _config_name(dataset_config.get('label_col'), 'label')
    impression_id_col = _config_name(dataset_config.get('impression_id_col'), 'impression_id')
    user_columns = [str(col) for col in user_columns]
    item_columns = {str(col) for col in item_features_df.columns} if item_features_df is not None else set()
    extra_columns = set(_configured_name_list(extra_columns))
    special_columns = {
        item_id_col,
        'cand_item_id',
        'app_id',
        label_col,
        'label',
        impression_id_col,
        'column_id',
    }
    safe_columns = {'request_id', 'user_id'}
    conditioned = []
    for col in user_columns:
        if col in safe_columns:
            continue
        group = getattr(fg_manager, 'feature_assignments', {}).get(col) if fg_manager is not None else None
        if (
            col in extra_columns
            or col in special_columns
            or col in item_columns
            or group == FeatureGroup.FG1
            or _is_target_conditioned_user_column(col)
        ):
            conditioned.append(col)
    return conditioned


def _neutral_feature_value(feature_map, feature_name):
    spec = (getattr(feature_map, 'features', {}) or {}).get(feature_name, {}) or {}
    feature_type = spec.get('type')
    if feature_type == 'sequence' or spec.get('max_len'):
        length = int(spec.get('max_len') or 1)
        padding_idx = spec.get('padding_idx', 0)
        try:
            padding_idx = int(padding_idx)
        except (TypeError, ValueError):
            padding_idx = 0
        return [padding_idx] * max(1, length)
    return 0.0 if feature_type == 'numeric' else 0


def apply_strict_all_items_user_features(
        stage_output,
        item_features_df,
        feature_map,
        dataset_config,
        fg_manager=None,
        extra_columns=None,
        logger=None,
        split_name='unknown'):
    if stage_output is None:
        return stage_output
    metadata = dict(stage_output.metadata or {})
    if not metadata.get('lazy_all_items'):
        return stage_output
    user_df = stage_output.user_features_df
    if user_df is None or user_df.empty:
        return stage_output
    drop_columns = detect_candidate_conditioned_user_columns(
        user_columns=user_df.columns,
        item_features_df=item_features_df,
        feature_map=feature_map,
        dataset_config=dataset_config,
        fg_manager=fg_manager,
        extra_columns=extra_columns,
    )
    if not drop_columns:
        metadata['strict_all_items_eval'] = True
        metadata['strict_all_items_neutralized_user_columns'] = []
        return StageOutput.from_dataframes(
            stage_name=stage_output.stage_name,
            candidates_df=stage_output.candidates_df,
            user_features_df=user_df,
            metrics=stage_output.metrics,
            metadata=metadata,
        )

    strict_user_df = user_df.copy()
    for col in drop_columns:
        if col in strict_user_df.columns:
            strict_user_df[col] = [_neutral_feature_value(feature_map, col)] * len(strict_user_df)
    metadata['strict_all_items_eval'] = True
    metadata['strict_all_items_neutralized_user_columns'] = drop_columns
    metadata['strict_all_items_active_user_columns'] = [str(col) for col in strict_user_df.columns]
    if logger is not None:
        logger.info(
            "[Preranking][StrictAllItems:%s] Neutralized %d candidate-conditioned user columns: %s",
            split_name,
            len(drop_columns),
            drop_columns,
        )
    return StageOutput.from_dataframes(
        stage_name=stage_output.stage_name,
        candidates_df=stage_output.candidates_df,
        user_features_df=strict_user_df,
        metrics=stage_output.metrics,
        metadata=metadata,
    )


def run_preranking_feature_leakage_check(
        stage_output,
        item_features_df,
        feature_map,
        dataset_config,
        output_dir,
        split_name,
        max_sequence_checks=0,
        fg_manager=None,
        extra_candidate_conditioned_columns=None,
        logger=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    if stage_output is None or stage_output.user_features_df is None:
        logger.warning("[Preranking][LeakageCheck:%s] No user features available.", split_name)
        return {}
    if item_features_df is None:
        logger.warning("[Preranking][LeakageCheck:%s] No item features available.", split_name)
        return {}

    user_df = stage_output.user_features_df
    candidates_df = stage_output.candidates_df
    item_id_col = _config_name(dataset_config.get('item_id_col'), 'cand_item_id')
    label_col = _config_name(dataset_config.get('label_col'), 'label')
    impression_id_col = _config_name(dataset_config.get('impression_id_col'), 'impression_id')
    user_columns = [str(col) for col in user_df.columns]
    item_columns = [str(col) for col in item_features_df.columns]
    risky_user_names = {
        item_id_col,
        'cand_item_id',
        'app_id',
        label_col,
        'label',
        impression_id_col,
        'column_id',
    }
    risky_item_names = {label_col, 'label', impression_id_col, 'column_id'}
    risky_user_columns = [col for col in user_columns if col in risky_user_names]
    risky_item_columns = [col for col in item_columns if col in risky_item_names]
    shared_feature_exclusions = {'request_id', impression_id_col, 'impression_id'}
    item_column_set = set(item_columns)
    candidate_feature_columns_in_user_features = [
        col for col in user_columns
        if col in item_column_set and col not in shared_feature_exclusions
    ]
    detected_candidate_conditioned_user_columns = detect_candidate_conditioned_user_columns(
        user_columns=user_columns,
        item_features_df=item_features_df,
        feature_map=feature_map,
        dataset_config=dataset_config,
        fg_manager=fg_manager,
        extra_columns=extra_candidate_conditioned_columns,
    )
    neutralized_candidate_conditioned_user_columns = list(
        (stage_output.metadata or {}).get('strict_all_items_neutralized_user_columns') or []
    )
    neutralized_set = set(neutralized_candidate_conditioned_user_columns)
    active_candidate_conditioned_user_columns = [
        col for col in detected_candidate_conditioned_user_columns
        if col not in neutralized_set
    ]
    candidate_conditioned_user_columns = sorted(set(
        detected_candidate_conditioned_user_columns
        + neutralized_candidate_conditioned_user_columns
    ))
    user_sequence_columns = _sequence_feature_columns(feature_map, user_columns)
    item_sequence_columns = _sequence_feature_columns(feature_map, item_columns)

    positive_by_request = {}
    if candidates_df is not None and not candidates_df.empty:
        positive_df = candidates_df
        if 'label' in positive_df.columns:
            positive_df = positive_df[positive_df['label'].fillna(0).astype(int) > 0]
        for req_id, group in positive_df.groupby('request_id', sort=False):
            positive_by_request[_normalize_id_for_compare(req_id)] = [
                _normalize_id_for_compare(item_id)
                for item_id in group['item_id'].to_numpy()
            ]

    sequence_checks = 0
    sequence_hits = 0
    sequence_hit_examples = []
    max_checks = max(0, int(max_sequence_checks or 0))
    if user_sequence_columns and positive_by_request:
        for _, row in user_df.iterrows():
            req_key = _normalize_id_for_compare(row.get('request_id'))
            positive_items = set(positive_by_request.get(req_key, []))
            if not positive_items:
                continue
            for seq_col in user_sequence_columns:
                if max_checks > 0 and sequence_checks >= max_checks:
                    break
                seq_values = _sequence_values_for_compare(row.get(seq_col))
                sequence_checks += 1
                matched = sorted(positive_items.intersection(seq_values))
                if matched:
                    sequence_hits += 1
                    if len(sequence_hit_examples) < 20:
                        sequence_hit_examples.append({
                            'request_id': _safe_json_value(row.get('request_id')),
                            'sequence_column': seq_col,
                            'positive_item': matched[0],
                            'positive_items': sorted(positive_items),
                            'sequence_values_sample': sorted(seq_values)[:50],
                        })
            if max_checks > 0 and sequence_checks >= max_checks:
                break

    report = {
        'split': split_name,
        'lazy_all_items': bool((stage_output.metadata or {}).get('lazy_all_items')),
        'strict_all_items_eval': bool((stage_output.metadata or {}).get('strict_all_items_eval')),
        'user_feature_columns': user_columns,
        'item_feature_columns': item_columns,
        'user_sequence_columns': user_sequence_columns,
        'item_sequence_columns': item_sequence_columns,
        'risky_user_columns': risky_user_columns,
        'risky_item_columns': risky_item_columns,
        'candidate_feature_columns_in_user_features': candidate_feature_columns_in_user_features,
        'candidate_conditioned_user_columns': candidate_conditioned_user_columns,
        'active_candidate_conditioned_user_columns': active_candidate_conditioned_user_columns,
        'neutralized_candidate_conditioned_user_columns': neutralized_candidate_conditioned_user_columns,
        'positive_request_count': len(positive_by_request),
        'sequence_positive_item_checks': sequence_checks,
        'sequence_positive_item_hits': sequence_hits,
        'sequence_positive_item_hit_rate': (
            float(sequence_hits / sequence_checks) if sequence_checks > 0 else 0.0
        ),
        'sequence_positive_item_hit_examples': sequence_hit_examples,
    }
    report_dir = os.path.join(output_dir, 'ranking_diagnostics')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'{split_name}_feature_leakage_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(
        "[Preranking][LeakageCheck:%s] user_features=%d item_features=%d user_sequences=%d "
        "sequence_hits=%d/%d report=%s",
        split_name,
        len(user_columns),
        len(item_columns),
        len(user_sequence_columns),
        sequence_hits,
        sequence_checks,
        report_path,
    )
    if (
        risky_user_columns
        or risky_item_columns
        or candidate_feature_columns_in_user_features
        or active_candidate_conditioned_user_columns
        or sequence_hits > 0
    ):
        logger.warning(
            "[Preranking][LeakageCheck:%s] Potential leakage: risky_user_columns=%s "
            "risky_item_columns=%s candidate_feature_columns_in_user_features=%s "
            "active_candidate_conditioned_user_columns=%s sequence_hits=%d/%d",
            split_name,
            risky_user_columns,
            risky_item_columns,
            candidate_feature_columns_in_user_features,
            active_candidate_conditioned_user_columns,
            sequence_hits,
            sequence_checks,
        )
    return report


def _batch_to_numpy(batch_data):
    arrays = {}
    for key, value in batch_data.items():
        if hasattr(value, 'detach'):
            arrays[key] = value.detach().cpu().numpy()
        elif hasattr(value, 'cpu') and hasattr(value, 'numpy'):
            arrays[key] = value.cpu().numpy()
        else:
            arrays[key] = np.asarray(value)
    return arrays


def build_observed_candidate_stage_output(
        feature_map,
        dataset_config,
        pipeline_config,
        fg_manager,
        split_name,
        data_path,
        logger=None):
    """Build a retrieval-like StageOutput when retrieval outputs are unavailable."""
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    if not data_path:
        logger.warning("[%s] Cannot bootstrap preranking candidates: data path is empty.", split_name)
        return None

    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)
    data_format = paths['data_format']
    preranking_config = pipeline_config.get('stages', {}).get('preranking', {})
    bootstrap_conf = dict(
        pipeline_config.get('preranking_bootstrap', {})
        or preranking_config.get('bootstrap_candidates', {})
        or {}
    )
    batch_size = int(
        bootstrap_conf.get(
            'batch_size',
            preranking_config.get('training', {}).get('batch_size', 4096),
        )
    )
    candidate_strategy = str(bootstrap_conf.get('candidate_strategy', 'all_items')).lower()
    if candidate_strategy in {'all', 'full', 'full_item_pool', 'item_pool'}:
        candidate_strategy = 'all_items'
    if candidate_strategy not in {'all_items', 'observed'}:
        logger.warning(
            "[Preranking] Unknown candidate_strategy=%r; falling back to all_items.",
            candidate_strategy,
        )
        candidate_strategy = 'all_items'

    max_records = int(bootstrap_conf.get('max_records', 0) or 0)
    max_requests = int(bootstrap_conf.get('max_requests', 0) or 0)
    max_candidate_rows = int(bootstrap_conf.get('max_candidate_rows', 0) or 0)
    positive_requests_only = _to_bool(
        bootstrap_conf.get('positive_requests_only'),
        candidate_strategy == 'all_items',
    )
    loader_kwargs = {}
    if data_format in {'tfrecord', 'tf_record'}:
        loader_kwargs['tfrecord_load_conf'] = paths.get('tfrecord_load_conf', {})
    loader_kwargs['num_workers'] = int(bootstrap_conf.get('num_workers', 0) or 0)

    logger.info(
        "[Preranking] Bootstrapping %s candidates directly from %s "
        "(strategy=%s, format=%s, batch_size=%d, max_records=%s, "
        "max_requests=%s, max_candidate_rows=%s, positive_requests_only=%s, num_workers=%d).",
        split_name,
        data_path,
        candidate_strategy,
        data_format,
        batch_size,
        max_records if max_records > 0 else "all",
        max_requests if max_requests > 0 else "all",
        max_candidate_rows if max_candidate_rows > 0 else "all",
        positive_requests_only,
        loader_kwargs['num_workers'],
    )
    loader = RankDataLoader(
        feature_map=feature_map,
        stage='test',
        test_data=data_path,
        batch_size=batch_size,
        shuffle=False,
        data_format=data_format,
        **loader_kwargs,
    )
    data_iter = loader.make_iterator()

    item_id_col = _config_name(dataset_config.get('item_id_col'), 'cand_item_id')
    user_id_col = _config_name(dataset_config.get('user_id_col'), 'user_id')
    impression_id_col = _config_name(dataset_config.get('impression_id_col'), 'impression_id')
    label_col = _config_name(dataset_config.get('label_col'), 'label')
    user_feature_names = (
        fg_manager.get_user_features()
        if fg_manager is not None
        else set(feature_map.features.keys()) - {item_id_col, impression_id_col, label_col}
    )

    candidate_rows = []
    user_row_by_request = {}
    request_order = []
    positives_by_request = {}
    seen_requests = set()
    total_rows = 0

    for batch_data in data_iter:
        arrays = _batch_to_numpy(batch_data)
        if item_id_col not in arrays:
            raise KeyError(
                f"Cannot bootstrap preranking candidates because item_id_col={item_id_col!r} "
                f"is missing from {split_name} batch. Available columns: {sorted(arrays.keys())}"
            )
        batch_size_actual = len(arrays[item_id_col])
        for row_idx in range(batch_size_actual):
            if max_records > 0 and total_rows >= max_records:
                break
            request_id = (
                _as_scalar(arrays[impression_id_col][row_idx])
                if impression_id_col in arrays
                else total_rows
            )
            item_id = _as_scalar(arrays[item_id_col][row_idx])
            raw_label = _as_scalar(arrays[label_col][row_idx]) if label_col in arrays else 0
            label = int(float(raw_label) > 0.0)
            if label > 0:
                positives_by_request.setdefault(request_id, set()).add(item_id)
            if candidate_strategy == 'observed':
                candidate_rows.append({
                    'request_id': request_id,
                    'item_id': item_id,
                    'score': 0.0,
                    'label': label,
                })

            if request_id not in seen_requests:
                seen_requests.add(request_id)
                request_order.append(request_id)
                user_id = (
                    _as_scalar(arrays[user_id_col][row_idx])
                    if user_id_col and user_id_col in arrays
                    else request_id
                )
                user_row = {'request_id': request_id, 'user_id': user_id}
                for feat_name in sorted(user_feature_names):
                    if feat_name in {item_id_col, impression_id_col, label_col}:
                        continue
                    if feat_name not in arrays or feat_name not in feature_map.features:
                        continue
                    feat_type = feature_map.features[feat_name].get('type')
                    user_row[feat_name] = _row_value(arrays[feat_name], row_idx, feat_type)
                user_row_by_request[request_id] = user_row
            total_rows += 1
        if max_records > 0 and total_rows >= max_records:
            break

    if candidate_strategy == 'all_items' and positive_requests_only:
        before_count = len(request_order)
        request_order = [request_id for request_id in request_order if positives_by_request.get(request_id)]
        logger.info(
            "[Preranking] Positive-request filter for %s all-items evaluation: requests=%d -> %d.",
            split_name,
            before_count,
            len(request_order),
        )

    if max_requests > 0 and len(request_order) > max_requests:
        logger.warning(
            "[Preranking] Limiting %s bootstrap requests from %d to %d.",
            split_name,
            len(request_order),
            max_requests,
        )
        request_order = request_order[:max_requests]

    if candidate_strategy == 'all_items':
        full_item_pool_path = ensure_full_item_pool(
            data_paths=paths,
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=feature_map,
        )
        try:
            item_pool_df = pd.read_parquet(full_item_pool_path, columns=[item_id_col])
        except Exception:
            item_pool_df = pd.read_parquet(full_item_pool_path)
            if item_id_col not in item_pool_df.columns:
                raise KeyError(
                    f"Full item pool {full_item_pool_path} does not contain item_id_col={item_id_col!r}. "
                    f"Available columns: {list(item_pool_df.columns)}"
                )
            item_pool_df = item_pool_df[[item_id_col]]
        item_ids = item_pool_df[item_id_col].drop_duplicates().to_numpy()
        if len(item_ids) == 0:
            raise ValueError(f"Full item pool {full_item_pool_path} is empty.")

        estimated_rows = len(request_order) * len(item_ids)
        if max_candidate_rows > 0 and estimated_rows > max_candidate_rows:
            max_allowed_requests = max(1, max_candidate_rows // len(item_ids))
            logger.warning(
                "[Preranking] Limiting %s all-items bootstrap from %d to %d requests "
                "because estimated candidates=%d exceeds max_candidate_rows=%d.",
                split_name,
                len(request_order),
                max_allowed_requests,
                estimated_rows,
                max_candidate_rows,
            )
            request_order = request_order[:max_allowed_requests]
            estimated_rows = len(request_order) * len(item_ids)
        positive_rows = []
        item_set = set(item_ids.tolist())
        for request_id in request_order:
            for positive_item in positives_by_request.get(request_id, ()):
                if positive_item in item_set:
                    positive_rows.append({
                        'request_id': request_id,
                        'item_id': positive_item,
                        'score': 0.0,
                        'label': 1,
                    })
        candidates_df = pd.DataFrame(
            positive_rows,
            columns=['request_id', 'item_id', 'score', 'label'],
        )
        logger.info(
            "[Preranking] %s all-items bootstrap will be evaluated lazily "
            "(requests=%d * items=%d = %d estimated candidates, positives=%d).",
            split_name,
            len(request_order),
            len(item_ids),
            estimated_rows,
            len(positive_rows),
        )
        source = 'full_item_pool'
        source_path = full_item_pool_path
        lazy_all_items = True
    else:
        candidates_df = pd.DataFrame(candidate_rows, columns=['request_id', 'item_id', 'score', 'label'])
        source = 'observed_split'
        source_path = data_path
        lazy_all_items = False

    user_rows = [user_row_by_request[request_id] for request_id in request_order if request_id in user_row_by_request]
    user_features_df = pd.DataFrame(user_rows)
    if user_features_df.empty:
        user_features_df = pd.DataFrame(columns=['request_id', 'user_id'])
    output = StageOutput.from_dataframes(
        stage_name='retrieval',
        candidates_df=candidates_df,
        user_features_df=user_features_df,
        metrics={},
        metadata={
            'source': source,
            'split': split_name,
            'data_path': data_path,
            'candidate_strategy': candidate_strategy,
            'candidate_source_path': source_path,
            'lazy_all_items': lazy_all_items,
            'estimated_candidates': int(estimated_rows) if candidate_strategy == 'all_items' else len(candidates_df),
            'positive_requests_only': positive_requests_only,
        },
    )
    logger.info(
        "[Preranking] Bootstrapped %s candidates: strategy=%s, requests=%d, stored_candidates=%d, positives=%d, lazy_all_items=%s.",
        split_name,
        candidate_strategy,
        len(user_features_df),
        output.get_total_candidates(),
        int(candidates_df['label'].sum()) if 'label' in candidates_df else 0,
        lazy_all_items,
    )
    return output


def _create_item_feature_map(feature_map, fg_manager):
    """
    Create a lean feature map for item pool (FG1 features only).

    Args:
        feature_map: Full FeatureMap instance
        fg_manager: FeatureGroupManager instance
        dataset_config: Dataset configuration dict

    Returns:
        FeatureMap with only item (FG1) features
    """
    item_feature_names = [name for name, spec in feature_map.features.items()
                          if fg_manager.feature_assignments.get(name) == FeatureGroup.FG1]

    item_fm = copy.deepcopy(feature_map)
    item_fm.features = {k: v for k, v in item_fm.features.items() if k in item_feature_names}
    item_fm.labels = []

    item_fm.set_column_index()
    item_fm.column_index = {k: v for k, v in item_fm.column_index.items() if k in item_feature_names}
    return item_fm


def _resolve_train_path_for_popularity(dataset_config):
    train_path = dataset_config.get('train_data')
    if not train_path:
        processed_paths = dataset_config.get('processed_paths', {}) or {}
        train_path = processed_paths.get('train')
    if not train_path:
        processed_root = dataset_config.get('processed_data_root')
        data_format = dataset_config.get(
            'processed_data_format',
            dataset_config.get('data_format', 'parquet')
        )
        if processed_root:
            train_path = os.path.join(processed_root, f'train.{data_format}')
    if train_path:
        train_path = os.path.expanduser(os.path.expandvars(str(train_path)))
    return train_path


def _build_popularity_feature_map(feature_map, dataset_config, item_id_col, label_col, positive_only, logger):
    if item_id_col not in feature_map.features:
        logger.warning(
            "[Preranking] Popularity baseline disabled: item id column %r is missing from feature_map.",
            item_id_col,
        )
        return None

    pop_fm = copy.deepcopy(feature_map)
    pop_fm.features = OrderedDict([(item_id_col, pop_fm.features[item_id_col])])
    if positive_only and label_col:
        pop_fm.labels = [label_col]
    else:
        pop_fm.labels = []
    pop_fm.dataset_config = dataset_config
    pop_fm.set_column_index()
    return pop_fm


def _count_train_item_popularity(
        feature_map,
        dataset_config,
        paths,
        item_id_col,
        label_col,
        positive_only=True,
        batch_size=8192,
        progress_interval=0,
        logger=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    train_path = _resolve_train_path_for_popularity(dataset_config) or paths.get('train_path')
    data_format = str(paths.get('data_format') or dataset_config.get(
        'processed_data_format',
        dataset_config.get('data_format', 'parquet')
    )).lower()
    is_tfrecord = data_format in {'tfrecord', 'tf_record'} or str(train_path).endswith(('.tfrecord', '.tfrecord.gz'))
    if not train_path:
        logger.warning("[Preranking] Popularity baseline disabled: train split path is empty.")
        return None
    if not is_tfrecord and not os.path.exists(train_path):
        logger.warning(
            "[Preranking] Popularity baseline disabled: train split not found at %s.",
            train_path,
        )
        return None

    pop_feature_map = _build_popularity_feature_map(
        feature_map=feature_map,
        dataset_config=dataset_config,
        item_id_col=item_id_col,
        label_col=label_col,
        positive_only=positive_only,
        logger=logger,
    )
    if pop_feature_map is None:
        return None

    loader_kwargs = {}
    if data_format in {'tfrecord', 'tf_record'}:
        loader_conf = dict(paths.get('tfrecord_load_conf') or dataset_config.get('tfrecord_load_conf') or {})
        loader_conf['count_samples'] = False
        loader_kwargs['tfrecord_load_conf'] = loader_conf

    try:
        loader = RankDataLoader(
            feature_map=pop_feature_map,
            stage='train',
            train_data=train_path,
            batch_size=max(1, int(batch_size or 8192)),
            shuffle=False,
            data_format=data_format,
            **loader_kwargs,
        )
        train_iter, _ = loader.make_iterator()
    except Exception as exc:
        logger.warning(
            "[Preranking] Popularity baseline disabled: failed to open train split %s (%s).",
            train_path,
            exc,
        )
        return None

    counts = {}
    rows_seen = 0
    rows_counted = 0
    positive_filter_missing = False
    for batch_idx, batch in enumerate(train_iter, 1):
        if item_id_col not in batch:
            logger.warning(
                "[Preranking] Popularity baseline disabled: %r missing from train batch.",
                item_id_col,
            )
            return None
        values = batch[item_id_col].detach().cpu().numpy()
        values = np.asarray(values).reshape(-1)
        rows_seen += int(values.size)
        if positive_only:
            if label_col in batch:
                labels = batch[label_col].detach().cpu().numpy().reshape(-1)
                values = values[labels > 0]
            else:
                positive_filter_missing = True
        if values.size:
            uniq, freq = np.unique(values, return_counts=True)
            for value, count in zip(uniq.tolist(), freq.tolist()):
                key = _as_scalar(value)
                counts[key] = counts.get(key, 0) + int(count)
            rows_counted += int(values.size)
        if progress_interval and batch_idx % int(progress_interval) == 0:
            logger.info(
                "[Preranking] Popularity baseline scan progress: batches=%d, rows_seen=%d, rows_counted=%d.",
                batch_idx,
                rows_seen,
                rows_counted,
            )

    if positive_filter_missing:
        logger.warning(
            "[Preranking] Popularity baseline requested positive_only=true but label column %r was not available; "
            "counted all train rows.",
            label_col,
        )
    logger.info(
        "[Preranking] Popularity baseline loaded from %s: unique_items=%d, rows_seen=%d, rows_counted=%d, positive_only=%s.",
        train_path,
        len(counts),
        rows_seen,
        rows_counted,
        positive_only and not positive_filter_missing,
    )
    return counts


def attach_preranking_popularity_baseline(
        preranking_stage,
        dataset_config,
        paths,
        pipeline_config,
        logger=None):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    model_params = (
        ((pipeline_config.get('stages') or {}).get('preranking') or {}).get('model_params')
        or {}
    )
    if not _to_bool(model_params.get('popularity_baseline'), False):
        return
    item_features_df = getattr(preranking_stage, 'item_features_df', None)
    if item_features_df is None or item_features_df.empty:
        logger.warning("[Preranking] Popularity baseline enabled but item feature pool is empty.")
        return
    if POPULARITY_BASELINE_COLUMN in item_features_df.columns:
        return

    item_id_col = _config_name(dataset_config.get('item_id_col'), 'cand_item_id')
    label_col = _config_name(dataset_config.get('label_col'), 'label')
    counts = _count_train_item_popularity(
        feature_map=preranking_stage.feature_map,
        dataset_config=dataset_config,
        paths=paths,
        item_id_col=item_id_col,
        label_col=label_col,
        positive_only=_to_bool(model_params.get('popularity_positive_only'), True),
        batch_size=int(model_params.get('popularity_batch_size', 8192) or 8192),
        progress_interval=int(model_params.get('popularity_progress_interval', 0) or 0),
        logger=logger,
    )
    if counts is None:
        return

    if hasattr(preranking_stage, 'set_popularity_prior'):
        preranking_stage.set_popularity_prior(
            counts,
            item_id_col=item_id_col,
            transform=str(model_params.get('popularity_blend_transform', 'log1p') or 'log1p'),
            normalize=str(model_params.get('popularity_blend_normalize', 'zscore') or 'zscore'),
        )

    popularity_values = np.fromiter(
        (float(counts.get(_as_scalar(item_id), 0.0)) for item_id in item_features_df.index.to_numpy()),
        dtype=np.float32,
        count=len(item_features_df),
    )
    preranking_stage.item_features_df = item_features_df.assign(
        **{POPULARITY_BASELINE_COLUMN: popularity_values}
    )
    nonzero_items = int(np.count_nonzero(popularity_values))
    logger.info(
        "[Preranking] Attached popularity baseline column to item pool: items=%d, nonzero_items=%d.",
        len(popularity_values),
        nonzero_items,
    )


def run_retrieval_stage(retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=None, run_test=True):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    retrieval_config = pipeline_config['stages']['retrieval']
    metrics = {}

    # 1. Prepare data loaders for retrieval stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Ensure item pool exists (generate if missing)
    ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'test_path': paths['test_path'],
                    'valid_path': paths['valid_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger,
        feature_map=getattr(retrieval_stage, 'full_feature_map', retrieval_stage.feature_map),
    )

    # Create item feature map for item pool loader
    item_fm = _create_item_feature_map(retrieval_stage.feature_map, fg_manager)
    logger.info(f"Loading item pool from {paths['item_pool_path']}")

    # Get training mode config
    num_negatives = retrieval_config.get('model_params', {}).get('num_negatives', 0)
    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # Debug: Verify feature map size
    logger.info(f"[Retrieval] Using feature map with {len(retrieval_stage.feature_map.features)} features.")
    debug_features = sorted(list(retrieval_stage.feature_map.features.keys()))
    logger.info(f"[Retrieval] Features: {debug_features[:10]} ... (Total {len(debug_features)})")

    loaders = _prepare_stage_data_loaders(
        feature_map=getattr(retrieval_stage, 'full_feature_map', retrieval_stage.feature_map),
        stage_config=retrieval_config,
        paths=paths,
        create_train=True,
        create_test=run_test,
        shuffle_train=True,
        create_item_loader=True,
        item_feature_map=item_fm,
        num_negatives=num_negatives,
        label_col=label_col,
        logger=logger
    )
    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']
    item_loader = loaders['item_loader']
    test_loader = loaders['test_loader'] if run_test else None

    # 2. Train and build item index if needed
    if retrieval_stage.item_embeddings is None:
        logger.info("[Training] Building item index for retrieval stage...")
        train_gen, _ = train_loader.make_iterator()
        valid_gen = valid_loader.make_iterator()
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_model()

        # Load train-only item features for explicit negative sampling.  Holdout
        # item pools remain reserved for validation/test candidate generation.
        item_features_df = None
        num_negatives = retrieval_stage.num_negatives if hasattr(retrieval_stage, 'num_negatives') else 0
        if num_negatives > 0:
            train_pool_path = _ensure_train_negative_pool(
                stage_name='Retrieval',
                stage_config=retrieval_config,
                paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                feature_map=getattr(retrieval_stage, 'full_feature_map', retrieval_stage.feature_map),
                logger=logger,
            )
            if train_pool_path and os.path.exists(train_pool_path):
                logger.info("Loading train-only item pool for negative sampling from %s", train_pool_path)
                item_features_df = pd.read_parquet(train_pool_path)
                logger.info("Loaded %d items for negative sampling (train-only pool)", len(item_features_df))
            else:
                logger.warning("Train-only item pool not found. Negative sampling will be disabled.")

        retrieval_stage.train(
            train_data=train_gen,
            valid_data=valid_gen,
            item_data=item_gen,
            item_features_df=item_features_df,
            epochs=retrieval_config['training'].get('epochs', 10),
            patience=retrieval_config['training'].get('patience', 2),
            monitor=retrieval_config['training'].get('monitor', 'Recall@1000'),
            mode=retrieval_config['training'].get('mode', 'max'),
            progress_interval=retrieval_config['training'].get(
                'progress_interval',
                retrieval_config.get('model_params', {}).get('progress_interval', 100),
            ),
        )

        # After training, load the best model and build item index for valid evaluation
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built successfully after training.")
    else:
        logger.info("Retrieval model already trained and item index built. Skipping training.")

    # Ensure item embeddings are available for test evaluation if not from training
    if retrieval_stage.item_embeddings is None:
        item_gen, _ = item_loader.make_iterator()
        retrieval_stage.build_item_index(item_gen)
        logger.info("Item index built for test evaluation.")
    logger.info("Evaluating on valid set and generating candidate sets")
    valid_output, valid_metrics = retrieval_stage.process(valid_loader.make_iterator(), compute_metrics=True)
    metrics.update({f"retrieval_valid_{k}": v for k, v in valid_metrics.items()})
    # 3. Evaluate on test set
    test_output = None
    if run_test:
        logger.info("Evaluating on test set and generating candidate sets")
        test_output, test_metrics = retrieval_stage.process(test_loader.make_iterator())
        metrics.update({f"retrieval_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping retrieval test evaluation as requested.")

    return metrics, valid_output, test_output


def save_retrieval_outputs(
        retrieval_stage,
        pipeline_config,
        dataset_config,
        fg_manager,
        model_weights_path,
        logger=None,
        run_test=True,
):
    """Load a trained retrieval checkpoint and export valid/test stage outputs."""
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    retrieval_config = pipeline_config['stages']['retrieval']
    metrics = {}

    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'test_path': paths['test_path'],
                    'valid_path': paths['valid_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger,
        feature_map=getattr(retrieval_stage, 'full_feature_map', retrieval_stage.feature_map),
    )

    item_fm = _create_item_feature_map(retrieval_stage.feature_map, fg_manager)
    num_negatives = retrieval_config.get('model_params', {}).get('num_negatives', 0)
    label_col = dataset_config.get('label_col', {}).get('name', 'label')
    loaders = _prepare_stage_data_loaders(
        feature_map=getattr(retrieval_stage, 'full_feature_map', retrieval_stage.feature_map),
        stage_config=retrieval_config,
        paths=paths,
        create_train=True,
        create_test=run_test,
        shuffle_train=False,
        create_item_loader=True,
        item_feature_map=item_fm,
        num_negatives=num_negatives,
        label_col=label_col,
        logger=logger
    )
    valid_loader = loaders['valid_loader']
    item_loader = loaders['item_loader']
    test_loader = loaders['test_loader'] if run_test else None

    retrieval_stage.build_model()
    retrieval_stage.model.load_weights(model_weights_path)
    retrieval_stage.best_weights_path = model_weights_path
    logger.info(f"[save_retrieval_outputs] Loaded pre-trained weights from: {model_weights_path}")

    item_gen, _ = item_loader.make_iterator()
    retrieval_stage.build_item_index(item_gen)
    logger.info("[save_retrieval_outputs] Item index built from candidate item pool.")

    logger.info("[save_retrieval_outputs] Processing valid set...")
    valid_output, valid_metrics = retrieval_stage.process(valid_loader.make_iterator(), compute_metrics=True)
    metrics.update({f"retrieval_valid_{k}": v for k, v in valid_metrics.items()})

    test_output = None
    if run_test:
        logger.info("[save_retrieval_outputs] Processing test set...")
        test_output, test_metrics = retrieval_stage.process(test_loader.make_iterator(), compute_metrics=True)
        metrics.update({f"retrieval_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("[save_retrieval_outputs] Skipping retrieval test evaluation.")

    return metrics, valid_output, test_output

def run_preranking_stage(preranking_stage, pipeline_config, dataset_config, fg_manager=None, logger=None,
                         prev_output_test=None, prev_output_valid=None, run_test=True, **kwargs):
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    preranking_config = pipeline_config['stages']['preranking']
    evaluation_config = dict(preranking_config.get('evaluation', {}) or {})
    evaluation_mode = str(evaluation_config.get('mode', 'listwise')).lower()
    pointwise_eval = evaluation_mode in {'pointwise', 'samplewise', 'observed'}
    ranking_eval_enabled = _to_bool(evaluation_config.get('ranking_metrics_enabled'), False)
    need_ranking_candidates = (not pointwise_eval) or ranking_eval_enabled
    pointwise_metrics = evaluation_config.get('metrics', ['AUC', 'logloss', 'pcoc', 'prauc'])
    pointwise_group_id_col = _config_name(
        evaluation_config.get('group_id_col'),
        _config_name(dataset_config.get('impression_id_col'), 'impression_id'),
    )
    pointwise_monitor = _config_name(evaluation_config.get('monitor'), 'AUC')
    checkpoint_eval_mode = str(evaluation_config.get('checkpoint_eval_mode') or '').lower()
    if not checkpoint_eval_mode:
        monitor_lc = str(pointwise_monitor or '').lower()
        ranking_monitor = monitor_lc.startswith((
            'recall@',
            'ndcg@',
            'diversity@',
            'mrr',
            'gauc',
            'popularity',
        ))
        checkpoint_eval_mode = 'ranking' if pointwise_eval and ranking_eval_enabled and ranking_monitor else 'pointwise'
    if checkpoint_eval_mode in {'ranking', 'listwise'}:
        checkpoint_eval_mode = 'listwise'
    else:
        checkpoint_eval_mode = 'pointwise' if pointwise_eval else 'listwise'
    save_pointwise_scores = _to_bool(evaluation_config.get('save_scores'), False)
    return_stage_output = _to_bool(
        preranking_config.get('model_params', {}).get('return_output', True),
        True,
    )
    preranking_model_params = preranking_config.get('model_params', {}) or {}
    metrics = {}

    if pointwise_eval:
        old_monitor = getattr(preranking_stage, 'monitor', None)
        if old_monitor != pointwise_monitor:
            logger.info(
                "[Preranking] Pointwise evaluation overrides monitor: %s -> %s",
                old_monitor,
                pointwise_monitor,
            )
        preranking_stage.monitor = pointwise_monitor
        preranking_stage.model_params['monitor'] = pointwise_monitor

    # 1. Prepare Data Loaders for preranking stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Create data loaders for preranking stage
    train_neg_conf = _training_negative_config(preranking_config.get('model_params', {}))
    num_negatives = train_neg_conf['num_negatives']
    use_diversity_loss = train_neg_conf['use_diversity_loss']
    diversity_num_negatives = train_neg_conf['diversity_num_negatives']
    training_num_negatives = train_neg_conf['training_num_negatives']
    loader_num_negatives = train_neg_conf['loader_num_negatives']
    if use_diversity_loss and num_negatives <= 0 and diversity_num_negatives > 0:
        logger.info(
            "[Preranking] Diversity loss requires explicit negatives; using diversity_num_negatives=%d "
            "for training data and item-pool preparation.",
            diversity_num_negatives,
        )
    if train_neg_conf['use_hybrid_loss']:
        logger.info(
            "[Preranking] Hybrid loss enabled; using original pointwise train data with ranking auxiliary negatives."
        )
    loaders = _prepare_stage_data_loaders(
        feature_map=preranking_stage.feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=pointwise_eval and run_test,
        num_negatives=loader_num_negatives,
        label_col=dataset_config.get('label_col', {}).get('name', 'label'),
        force_positive_train_data=train_neg_conf['force_positive_train_data'],
        logger=logger
    )
    train_gen, _ = loaders['train_loader'].make_iterator()
    _annotate_loader_sample_count(train_gen, 'train', dataset_config, logger)
    valid_pointwise_gen = loaders['valid_loader'].make_iterator() if pointwise_eval else None
    if pointwise_eval:
        _annotate_loader_sample_count(valid_pointwise_gen, 'valid', dataset_config, logger)
    lazy_all_items_eval = (
        need_ranking_candidates
        and (
            (getattr(prev_output_valid, 'metadata', {}) or {}).get('lazy_all_items')
            or (getattr(prev_output_test, 'metadata', {}) or {}).get('lazy_all_items')
        )
    )
    cand_item_pool_path = paths['item_pool_path']
    if pointwise_eval and training_num_negatives <= 0 and not ranking_eval_enabled:
        logger.info("[Preranking] Pointwise evaluation with num_negatives=0; skipping item pool preparation.")
    elif lazy_all_items_eval:
        cand_item_pool_path = (
            (getattr(prev_output_valid, 'metadata', {}) or {}).get('candidate_source_path')
            or (getattr(prev_output_test, 'metadata', {}) or {}).get('candidate_source_path')
        )
        if not cand_item_pool_path or not os.path.exists(cand_item_pool_path):
            cand_item_pool_path = ensure_full_item_pool(
                data_paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                logger=logger,
                feature_map=preranking_stage.feature_map,
            )
    else:
        cand_item_pool_path = ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=preranking_stage.feature_map,
        )
    # Keep the holdout-derived evaluation pool separate from the train-only
    # negative corpus.  Listwise validation during training must still be able
    # to score holdout candidates, but those candidates must never be sampled
    # as training negatives.
    if pointwise_eval and not ranking_eval_enabled and training_num_negatives <= 0:
        logger.info("[Preranking] Pointwise evaluation without explicit negatives; skipping item feature pool load.")
    elif os.path.exists(cand_item_pool_path):
        preranking_stage.load_item_features(cand_item_pool_path)
        logger.info("[Preranking] Loaded evaluation item pool: %s", cand_item_pool_path)
    else:
        logger.warning(
            "Evaluation item pool not found at %s. Listwise evaluation will be unavailable.",
            cand_item_pool_path,
        )

    if training_num_negatives > 0:
        train_negative_pool_path = _ensure_train_negative_pool(
            stage_name='Preranking',
            stage_config=preranking_config,
            paths=paths,
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            feature_map=preranking_stage.feature_map,
            logger=logger,
        )
        preranking_stage.load_negative_item_features(train_negative_pool_path)
        logger.info(
            "[Preranking] Loaded train-only negative pool: %s (%d items).",
            train_negative_pool_path,
            len(preranking_stage.train_negative_item_features_df),
        )
    # Enrich stage outputs with missing FG3 user features (backward compatibility).
    # Pointwise evaluation consumes the observed TFRecord rows directly and does not
    # need retrieval-style StageOutput user feature enrichment.
    if fg_manager is not None and need_ranking_candidates:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )
    if _to_bool(preranking_model_params.get('strict_all_items_eval'), False) and lazy_all_items_eval:
        extra_conditioned_columns = preranking_model_params.get(
            'strict_all_items_candidate_conditioned_columns',
            None,
        )
        prev_output_valid = apply_strict_all_items_user_features(
            stage_output=prev_output_valid,
            item_features_df=preranking_stage.item_features_df,
            feature_map=preranking_stage.feature_map,
            dataset_config=dataset_config,
            fg_manager=fg_manager,
            extra_columns=extra_conditioned_columns,
            logger=logger,
            split_name='valid',
        )
        if run_test and prev_output_test is not None:
            prev_output_test = apply_strict_all_items_user_features(
                stage_output=prev_output_test,
                item_features_df=preranking_stage.item_features_df,
                feature_map=preranking_stage.feature_map,
                dataset_config=dataset_config,
                fg_manager=fg_manager,
                extra_columns=extra_conditioned_columns,
                logger=logger,
                split_name='test',
            )
    if need_ranking_candidates:
        attach_preranking_popularity_baseline(
            preranking_stage=preranking_stage,
            dataset_config=dataset_config,
            paths=paths,
            pipeline_config=pipeline_config,
            logger=logger,
        )
    if _to_bool(preranking_model_params.get('feature_leakage_check'), False) and need_ranking_candidates:
        max_sequence_checks = _to_non_negative_int(
            preranking_model_params.get('feature_leakage_check_max_sequence_checks', 0),
            0,
        )
        run_preranking_feature_leakage_check(
            stage_output=prev_output_valid,
            item_features_df=preranking_stage.item_features_df,
            feature_map=preranking_stage.feature_map,
            dataset_config=dataset_config,
            output_dir=preranking_stage.output_dir,
            split_name='valid',
            max_sequence_checks=max_sequence_checks,
            fg_manager=fg_manager,
            extra_candidate_conditioned_columns=preranking_model_params.get(
                'strict_all_items_candidate_conditioned_columns',
                None,
            ),
            logger=logger,
        )
        if run_test and prev_output_test is not None:
            run_preranking_feature_leakage_check(
                stage_output=prev_output_test,
                item_features_df=preranking_stage.item_features_df,
                feature_map=preranking_stage.feature_map,
                dataset_config=dataset_config,
                output_dir=preranking_stage.output_dir,
                split_name='test',
                max_sequence_checks=max_sequence_checks,
                fg_manager=fg_manager,
                extra_candidate_conditioned_columns=preranking_model_params.get(
                    'strict_all_items_candidate_conditioned_columns',
                    None,
                ),
                logger=logger,
            )
    # 2. Train Preranking Model
    logger.info("[Preranking] Training model...")
    preranking_stage.build_model()
    train_valid_data = valid_pointwise_gen if pointwise_eval else prev_output_valid
    train_eval_mode = 'pointwise' if pointwise_eval else 'listwise'
    if pointwise_eval and checkpoint_eval_mode == 'listwise':
        if prev_output_valid is None:
            logger.warning(
                "[Preranking] checkpoint_eval_mode=ranking requested, but ranking valid candidates are unavailable; "
                "falling back to pointwise checkpoint evaluation."
            )
        else:
            train_valid_data = prev_output_valid
            train_eval_mode = 'listwise'
            logger.info(
                "[Preranking] Checkpoint monitor will use ranking validation: monitor=%s.",
                preranking_stage.monitor,
            )
    preranking_stage.train(
        train_data=train_gen,
        valid_data=train_valid_data,
        epochs=preranking_config['training'].get('epochs', 5),
        batch_size=preranking_config['training'].get('batch_size', 4096),
        evaluation_mode=train_eval_mode,
        pointwise_metrics=pointwise_metrics,
        pointwise_group_id_col=pointwise_group_id_col,
    )

    if need_ranking_candidates:
        attach_preranking_popularity_baseline(
            preranking_stage=preranking_stage,
            dataset_config=dataset_config,
            paths=paths,
            pipeline_config=pipeline_config,
            logger=logger,
        )

    if pointwise_eval:
        logger.info(
            "[Preranking] Pointwise evaluation mode enabled; evaluating observed valid/test rows directly."
        )
        score_dir = os.path.join(preranking_stage.output_dir, 'pointwise_scores')
        valid_score_path = (
            os.path.join(score_dir, 'valid_score.csv') if save_pointwise_scores else None
        )
        valid_pointwise_gen = loaders['valid_loader'].make_iterator()
        valid_metrics = preranking_stage.evaluate_pointwise(
            valid_pointwise_gen,
            metrics=pointwise_metrics,
            group_id_col=pointwise_group_id_col,
            score_path=valid_score_path,
            split_name='valid',
        )
        metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})

        test_output = None
        if run_test and loaders.get('test_loader') is not None:
            test_score_path = (
                os.path.join(score_dir, 'test_score.csv') if save_pointwise_scores else None
            )
            test_pointwise_gen = loaders['test_loader'].make_iterator()
            test_metrics = preranking_stage.evaluate_pointwise(
                test_pointwise_gen,
                metrics=pointwise_metrics,
                group_id_col=pointwise_group_id_col,
                score_path=test_score_path,
                split_name='test',
            )
            metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
        elif run_test:
            logger.warning("[Preranking] Test pointwise loader unavailable; skipping test evaluation.")
        else:
            logger.info("Skipping preranking test evaluation as requested.")
        if not ranking_eval_enabled:
            return metrics, None, test_output
        logger.info("[Preranking] Additional ranking metrics enabled; processing candidate pools for Recall/nDCG.")

    # 3. Pipeline Processing
    logger.info("[Preranking] Processing pipeline candidates...")
    ranking_top1_enabled = _to_bool(preranking_model_params.get('ranking_top1_diagnostics'), False)
    ranking_diag_dir = os.path.join(preranking_stage.output_dir, 'ranking_diagnostics')
    ranking_top1_top_k = _to_non_negative_int(preranking_model_params.get('ranking_top1_top_k', 10), 10) or 10
    ranking_top1_max_requests = _to_non_negative_int(
        preranking_model_params.get('ranking_top1_max_requests', 0),
        0,
    )

    test_output = None
    if run_test:
        if prev_output_test is None:
            logger.warning("[Preranking] No previous test output provided, cannot run test evaluation.")
        else:
            test_output, test_metrics = preranking_stage.process(
                prev_output_test,
                compute_metrics=True,
                return_output=return_stage_output,
                ranking_top1_path=(
                    os.path.join(ranking_diag_dir, 'test_ranking_top1.csv')
                    if ranking_top1_enabled else None
                ),
                ranking_top1_top_k=ranking_top1_top_k,
                ranking_top1_max_requests=ranking_top1_max_requests,
            )
            if not lazy_all_items_eval:
                logger.info("[Preranking] Ranking test metrics: %s", test_metrics)
            metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping preranking test evaluation as requested.")

    valid_output, valid_metrics = preranking_stage.process(
        prev_output_valid,
        compute_metrics=True,
        return_output=return_stage_output,
        ranking_top1_path=(
            os.path.join(ranking_diag_dir, 'valid_ranking_top1.csv')
            if ranking_top1_enabled else None
        ),
        ranking_top1_top_k=ranking_top1_top_k,
        ranking_top1_max_requests=ranking_top1_max_requests,
    )
    if not lazy_all_items_eval:
        logger.info("[Preranking] Ranking valid metrics: %s", valid_metrics)
    metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})
    return metrics, valid_output, test_output

def run_joint_training_stage(preranking_stage, reranking_stage, pipeline_config, dataset_config, fg_manager, logger=None,
                             run_test=True, prev_output_test=None, prev_output_valid=None):
    """Orchestrate joint preranking + reranking training.

    Delegates all training logic to CloudDeviceJointTrainingStage, which holds
    references to the stage objects and calls their existing train() methods.
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    import copy
    from cloud_device_recsys.pipeline import CloudDeviceJointTrainingStage

    preranking_config = pipeline_config['stages']['preranking']
    reranking_config  = pipeline_config['stages']['reranking']
    joint_params = pipeline_config.get('joint_training', {})

    # 1. Prepare data paths
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Per-stage num_negatives: read from each stage's model_params
    # (joint_params.num_negatives acts as fallback if not set per-stage)
    _jt_nn = joint_params.get('num_negatives', None)
    pre_num_negatives = preranking_config.get('model_params', {}).get('num_negatives', _jt_nn or 0)
    re_num_negatives  = reranking_config.get('model_params', {}).get('num_negatives', _jt_nn or 0)
    pre_use_in_batch_negatives = preranking_config.get('model_params', {}).get('use_in_batch_negatives', False)
    logger.info(f"[Joint Training] num_negatives: preranking={pre_num_negatives}, reranking={re_num_negatives}")

    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # 1a. Preranking train loader — uses preranking feature_map + config (correct batch_size)
    pre_loader_fm = copy.deepcopy(preranking_stage.feature_map)
    pre_loaders = _prepare_stage_data_loaders(
        feature_map=pre_loader_fm,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=pre_num_negatives,
        label_col=label_col,
        force_positive_train_data=pre_use_in_batch_negatives,
        logger=logger,
    )
    preranking_train_loader = pre_loaders['train_loader']

    # 1b. Reranking train loader — uses reranking feature_map + config (correct batch_size)
    re_loader_fm = copy.deepcopy(reranking_stage.feature_map)
    re_loaders = _prepare_stage_data_loaders(
        feature_map=re_loader_fm,
        stage_config=reranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=re_num_negatives,
        label_col=label_col,
        logger=logger,
    )
    reranking_train_loader = re_loaders['train_loader']

    # 2. Ensure item pool exists
    if fg_manager is not None:
        ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'],
                        'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=reranking_stage.feature_map,
        )
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )
    # 3. Build individual stage models
    preranking_stage.build_model()
    preranking_stage.best_weights_path = os.path.join(
        preranking_stage.model.model_dir, preranking_stage.model.model_id + ".model"
    )
    reranking_stage.build_model()
    reranking_stage.best_weights_path = os.path.join(
        reranking_stage.model.model_dir, reranking_stage.model.model_id + ".model"
    )

    # 4. Load item features for negative sampling
    # neg_sampling_pool: 'full' → cand_items_all.parquet (800K+ items, like standalone preranking)
    #                    'candidate' → item_pool_path (~28K, default)
    neg_pool_mode = joint_params.get('neg_sampling_pool', 'candidate')
    if neg_pool_mode == 'full' and os.path.exists(paths['full_item_pool_path']):
        neg_pool_path = paths['full_item_pool_path']
    else:
        neg_pool_path = paths['item_pool_path']
        if neg_pool_mode == 'full':
            logger.warning("[Joint Training] full_item_pool_path not found, falling back to candidate pool.")
    logger.info(f"[Joint Training] Negative sampling pool: '{neg_pool_mode}' → {neg_pool_path}")
    max_neg_nn = max(pre_num_negatives, re_num_negatives)
    if max_neg_nn > 0 and os.path.exists(neg_pool_path):
        preranking_stage.load_item_features(neg_pool_path)
        # Share item_features_df to reranking — avoids a redundant file load
        reranking_stage.item_features_df = preranking_stage.item_features_df
        logger.info("[Joint Training] Shared item_features_df from preranking to reranking stage.")
    neg_pool_path_for_params = neg_pool_path

    # 5. Merge per-stage num_negatives and pool info for the orchestrator
    merged_joint_params = dict(joint_params)
    merged_joint_params['pre_num_negatives'] = pre_num_negatives
    merged_joint_params['re_num_negatives']  = re_num_negatives
    merged_joint_params['_neg_pool_path']    = neg_pool_path_for_params

    # 6. Create orchestrator and train
    logger.info("[Joint Training] Initializing CloudDeviceJointTrainingStage...")
    joint = CloudDeviceJointTrainingStage(
        preranking_stage=preranking_stage,
        reranking_stage=reranking_stage,
        joint_params=merged_joint_params,
        preranking_config=preranking_config,
        reranking_config=reranking_config,
        preranking_train_loader=preranking_train_loader,
        reranking_train_loader=reranking_train_loader,
        fg_manager=fg_manager,
        dataset_config=dataset_config,
        paths=paths,
        logger=logger,
    )
    metrics = joint.train(prev_output_valid, prev_output_test, run_test=run_test)
    return metrics


def run_dtcn_preranking_stage(pipeline_config, dataset_config, fg_manager, feature_map,
                              logger=None, run_test=True,
                              prev_output_test=None, prev_output_valid=None, gpu=-1, **kwargs):
    """
    Run the DTCN preranking stage: dual-model (cloud + full) training with CL.

    Creates two DataLoaders:
      - Cloud DataLoader: FG1+FG2 features
      - Full DataLoader:  FG1+FG2+FG3 features
    Then creates a DTCNPrerankingStage and delegates training/evaluation.
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')

    from cloud_device_recsys.utils import filter_feature_map

    preranking_config = pipeline_config['stages']['preranking']
    dtcn_config = pipeline_config['stages'].get('dtcn', {})
    metrics = {}

    # 1. Prepare Data Paths
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    num_negatives = preranking_config.get('model_params', {}).get('num_negatives', 0)
    use_in_batch_negatives = preranking_config.get('model_params', {}).get('use_in_batch_negatives', False)
    label_col = dataset_config.get('label_col', {}).get('name', 'label')

    # 2. Build feature maps
    #    Cloud feature map: FG1 + FG2 (same as normal preranking)
    cloud_features = [FeatureGroup.from_string(f) for f in preranking_config.get('features', ['FG1', 'FG2'])]
    cloud_feature_map = filter_feature_map(
        feature_map, fg_manager, cloud_features,
        use_feature_encoder=preranking_config.get('model_params', {}).get('use_feature_encoder', False)
    )
    cloud_feature_map.default_emb_dim = preranking_config.get('model_params', {}).get('embedding_dim', 16)

    #    Full feature map: FG1 + FG2 + FG3
    full_features_list = [FeatureGroup.from_string(f) for f in dtcn_config.get('full_features', ['FG1', 'FG2', 'FG3'])]
    full_feature_map = filter_feature_map(
        feature_map, fg_manager, full_features_list,
        use_feature_encoder=dtcn_config.get('full_model_params', {}).get('use_feature_encoder',
                                            preranking_config.get('model_params', {}).get('use_feature_encoder', False))
    )
    full_feature_map.default_emb_dim = dtcn_config.get('full_model_params', {}).get(
        'embedding_dim', preranking_config.get('model_params', {}).get('embedding_dim', 16)
    )

    logger.info(f"[DTCN] Cloud feature map: {len(cloud_feature_map.features)} features")
    logger.info(f"[DTCN] Full feature map: {len(full_feature_map.features)} features")

    # 3. Create DataLoaders
    #    Cloud DataLoader (FG1+FG2)
    cloud_loaders = _prepare_stage_data_loaders(
        feature_map=cloud_feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=label_col,
        force_positive_train_data=use_in_batch_negatives,
        logger=logger
    )

    #    Full DataLoader (FG1+FG2+FG3)
    full_loaders = _prepare_stage_data_loaders(
        feature_map=full_feature_map,
        stage_config=preranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=label_col,
        force_positive_train_data=use_in_batch_negatives,
        logger=logger
    )

    # 4. Ensure item pool exists and load it
    cand_item_pool_path = ensure_item_pool(
        data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                    'test_path': paths['test_path']},
        dataset_config=dataset_config,
        feature_group_manager=fg_manager,
        logger=logger,
        feature_map=full_feature_map,
    )

    # 5. Build model params
    model_params = preranking_config.get('model_params', {}).copy()
    model_params['gpu'] = gpu
    model_params['model'] = preranking_config.get('model', 'PNN')
    if 'metrics' in preranking_config:
        model_params['metrics'] = preranking_config['metrics']

    # 6. Create DTCNPrerankingStage
    output_dir = kwargs.get('output_dir', './outputs/dtcn_preranking')
    dtcn_stage = DTCNPrerankingStage(
        feature_map=cloud_feature_map,
        full_feature_map=full_feature_map,
        feature_group_manager=fg_manager,
        model_params=model_params,
        dtcn_params=dtcn_config,
        allowed_feature_groups=cloud_features,
        output_dir=os.path.join(output_dir, 'preranking'),
        top_k=preranking_config.get('top_k', 100),
    )

    # 7. Load item features for negative sampling
    if num_negatives > 0 and fg_manager is not None:
        neg_pool_mode = preranking_config.get('neg_sampling_pool', 'candidate')
        if neg_pool_mode == 'full':
            neg_pool_path = ensure_full_item_pool(
                data_paths=paths,
                dataset_config=dataset_config,
                feature_group_manager=fg_manager,
                logger=logger,
                feature_map=full_feature_map,
            )
        else:
            neg_pool_path = cand_item_pool_path
        dtcn_stage.load_item_features(neg_pool_path)
        logger.info(f"[DTCN] Negative sampling pool: '{neg_pool_mode}' -> {neg_pool_path} ({len(dtcn_stage.item_features_df)} items)")
    else:
        if os.path.exists(paths['item_pool_path']):
            dtcn_stage.load_item_features(paths['item_pool_path'])
        else:
            logger.warning(f"Item pool not found at {paths['item_pool_path']}.")

    # 8. Enrich stage outputs with FG3 user features
    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )

    # 9. Train DTCN models
    logger.info("[DTCN] Building models...")
    dtcn_stage.build_model()

    cloud_train_gen, _ = cloud_loaders['train_loader'].make_iterator()
    full_train_gen, _ = full_loaders['train_loader'].make_iterator()

    dtcn_stage.train(
        train_data=cloud_train_gen,
        full_train_data=full_train_gen,
        valid_data=prev_output_valid,
        epochs=preranking_config['training'].get('epochs', 10),
        batch_size=preranking_config['training'].get('batch_size', 4096),
    )

    # 10. After training, switch to candidate item pool for evaluation
    if num_negatives > 0 and os.path.exists(cand_item_pool_path):
        dtcn_stage.load_item_features(cand_item_pool_path)
        logger.info(f"[DTCN] Switched to eval item pool ({len(dtcn_stage.item_features_df)} items)")

    # 11. Evaluate and process
    logger.info("[DTCN] Processing pipeline candidates (cloud model only)...")
    valid_output, valid_metrics = dtcn_stage.process(prev_output_valid, compute_metrics=True)
    metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})

    test_output = None
    if run_test:
        if prev_output_test is None:
            logger.warning("[DTCN] No previous test output provided, cannot run test evaluation.")
        else:
            test_output, test_metrics = dtcn_stage.process(prev_output_test, compute_metrics=True)
            metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping DTCN preranking test evaluation.")

    return metrics, valid_output, test_output


def run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=None, run_test=True, logger=None,
                        prev_output_path=None, preranking_model=None, prev_output_test=None, prev_output_valid=None,
                        retrieval_output_valid=None, retrieval_output_test=None,
                        feature_map=None, original_feature_map=None):
    """Run reranking stage with optional cloud teacher and evaluation scope alignment.

    Args:
        reranking_stage: RerankingStage instance (lazily instantiated)
        pipeline_config: Full pipeline configuration dict
        dataset_config: Dataset configuration dict
        fg_manager: FeatureGroupManager instance
        run_test: Whether to evaluate on test set
        logger: Logger instance
        prev_output_path: Path to load previous stage outputs from disk
        preranking_model: Pre-ranking model for legacy cloud_score injection
        prev_output_test: Preranking test output (StageOutput)
        prev_output_valid: Preranking valid output (StageOutput)
        retrieval_output_valid: Optional retrieval valid output for eval scope alignment.
            When provided, AUC/gAUC/MRR are computed on this larger pool (~1000 candidates)
            instead of the preranking subset (~100), aligning with preranking evaluation.
        retrieval_output_test: Optional retrieval test output for eval scope alignment.
        feature_map: Base FeatureMap (needed for cloud_teacher feature map construction)
        original_feature_map: The original, unpruned FeatureMap before any filtering for stage-specific features.
    """
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    if prev_output_path is not None:
        prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
            prev_output_path,
            "preranking",
            logger,
            load_test=bool(run_test),
        )
    reranking_config = pipeline_config['stages']['reranking']
    metrics = {}

    # 1. Parse cloud_teacher config and build teacher feature map
    cloud_teacher_config = pipeline_config['stages'].get('cloud_teacher', {})
    cloud_teacher_feature_map = None

    if cloud_teacher_config.get('mode') and feature_map is not None and fg_manager is not None:
        from cloud_device_recsys.utils import filter_feature_map
        teacher_features_str = cloud_teacher_config.get('cloud_teacher_features', ['FG1', 'FG2'])
        teacher_feature_groups = [FeatureGroup.from_string(f) for f in teacher_features_str]
        # Use original (unpruned) feature_map for teacher model construction.
        # The teacher checkpoint was saved with original vocab sizes, so building
        # the teacher from a pruned feature_map causes size mismatches on load.
        teacher_base_fm = original_feature_map if original_feature_map is not None else feature_map
        cloud_teacher_feature_map = filter_feature_map(
            teacher_base_fm, fg_manager, teacher_feature_groups,
            use_feature_encoder=cloud_teacher_config.get('cloud_teacher_model_params', {}).get(
                'use_feature_encoder',
                reranking_config.get('model_params', {}).get('use_feature_encoder', False)
            )
        )
        cloud_teacher_feature_map.default_emb_dim = cloud_teacher_config.get(
            'cloud_teacher_model_params', {}).get(
            'embedding_dim',
            reranking_config.get('model_params', {}).get('embedding_dim', 16)
        )
        logger.info(f"[Reranking] Cloud teacher feature map: {len(cloud_teacher_feature_map.features)} features "
                     f"({teacher_features_str})")

        # Inject into reranking_stage
        reranking_stage.cloud_teacher_params = cloud_teacher_config
        reranking_stage.cloud_teacher_feature_map = cloud_teacher_feature_map
        reranking_stage.cloud_teacher_mode = cloud_teacher_config.get('mode')

        # Update cloud_score injection flag based on mode
        if reranking_stage.cloud_teacher_mode == 'inject':
            reranking_stage.use_cloud_score = True
        elif reranking_stage.cloud_teacher_mode == 'distill':
            reranking_stage.use_cloud_score = False
            reranking_stage.kd_loss_weight = cloud_teacher_config.get('kd_loss_weight', 0.1)
            reranking_stage.kd_loss_type = cloud_teacher_config.get('kd_loss_type', 'mse')
            reranking_stage.kd_temperature = cloud_teacher_config.get('kd_temperature', 1.0)
        elif reranking_stage.cloud_teacher_mode == 'residual_inject':
            reranking_stage.use_cloud_score = False  # cloud_score NOT registered as feature
            reranking_stage.residual_weight = cloud_teacher_config.get('residual_weight', 1.0)
        elif reranking_stage.cloud_teacher_mode == 'hybrid_inject':
            reranking_stage.use_cloud_score = True   # acts as both feature and residual
            reranking_stage.residual_weight = cloud_teacher_config.get('residual_weight', 1.0)

    # 2. Prepare Data Loaders for reranking stage
    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    # Ensure the valid/test item pool exists for evaluation. It is not a
    # source of training negatives.
    eval_item_pool_path = paths['item_pool_path']
    if fg_manager is not None:
        eval_item_pool_path = ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=reranking_stage.feature_map,
        )

    # Create data loaders for reranking stage
    # Use deepcopy of feature_map so DataLoader doesn't see cloud_score
    # (cloud_score is registered in build_model() but doesn't exist in parquet)
    import copy
    loader_feature_map = copy.deepcopy(reranking_stage.feature_map)
    num_negatives = reranking_config.get('model_params', {}).get('num_negatives', 0)
    loaders = _prepare_stage_data_loaders(
        feature_map=loader_feature_map,
        stage_config=reranking_config,
        paths=paths,
        create_train=True,
        create_test=False,
        num_negatives=num_negatives,
        label_col=dataset_config.get('label_col', {}).get('name', 'label'),
        logger=logger
    )
    train_gen, _ = loaders['train_loader'].make_iterator()

    if os.path.exists(eval_item_pool_path):
        reranking_stage.load_item_features(eval_item_pool_path)
        logger.info("[Reranking] Loaded evaluation item pool: %s", eval_item_pool_path)
    else:
        logger.warning(
            "Evaluation item pool not found at %s. Reranking evaluation will be unavailable.",
            eval_item_pool_path,
        )

    if num_negatives > 0:
        train_negative_pool_path = _ensure_train_negative_pool(
            stage_name='Reranking',
            stage_config=reranking_config,
            paths=paths,
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            feature_map=reranking_stage.feature_map,
            logger=logger,
        )
        reranking_stage.load_negative_item_features(train_negative_pool_path)
        logger.info(
            "[Reranking] Loaded train-only negative pool: %s (%d items).",
            train_negative_pool_path,
            len(reranking_stage.train_negative_item_features_df),
        )

    # Enrich stage outputs with missing FG3 user features (backward compatibility)
    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )

        if retrieval_output_valid is not None:
            retrieval_output_valid = enrich_stage_output_user_features(
                retrieval_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
            )
        if retrieval_output_test is not None:
            retrieval_output_test = enrich_stage_output_user_features(
                retrieval_output_test, paths['test_path'], fg_manager, impression_id_col, logger
            )

    # 2.5 Offline Vocab Pruning Mismatch Fix
    # If vocab pruning is in offline mode, the preranking/retrieval StageOutputs contain
    # original unmapped item IDs. The reranking model expects mapped IDs. We must translate them here.
    vocab_pruning_config = pipeline_config.get('vocab_pruning', {})
    if vocab_pruning_config.get('enabled', False) and vocab_pruning_config.get('mode', 'runtime') == 'offline':
        data_dir = get_data_dir(dataset_config)
        remap_dicts = load_offline_remap_dicts(data_dir, logger)
        if remap_dicts is not None:
            from cloud_device_recsys.data.remap_vocab_data import remap_stage_output

            # Attach to reranking_stage so it can unmap IDs for the cloud teacher model
            reranking_stage.remap_dicts = remap_dicts

            # Apply remapping only if data exists
            if prev_output_valid is not None:
                prev_output_valid = remap_stage_output(prev_output_valid, remap_dicts, feature_map)
            if prev_output_test is not None:
                prev_output_test = remap_stage_output(prev_output_test, remap_dicts, feature_map)
            if retrieval_output_valid is not None:
                retrieval_output_valid = remap_stage_output(retrieval_output_valid, remap_dicts, feature_map)
            if retrieval_output_test is not None:
                retrieval_output_test = remap_stage_output(retrieval_output_test, remap_dicts, feature_map)
    logger.info("[Reranking] Training model...")

    # Set cloud score teacher if enabled (legacy path — only if no cloud_teacher config)
    if not cloud_teacher_config.get('mode'):
        use_cloud_score = reranking_config.get('model_params', {}).get('use_cloud_score', False)
        if use_cloud_score and preranking_model is not None:
            reranking_stage.set_cloud_score_teacher(preranking_model)
        elif use_cloud_score and preranking_model is None:
            logger.warning("[Reranking] use_cloud_score=True but no preranking model is loaded. Cloud score disabled.")

    reranking_stage.build_model()
    reranking_stage.train(
        train_data=train_gen,
        valid_data=prev_output_valid,
        epochs=reranking_config['training'].get('epochs', 5),
        batch_size=reranking_config['training'].get('batch_size', 4096)
    )

    # 4. Evaluate with optional evaluation scope alignment
    valid_metrics = reranking_stage.evaluate(
        prev_output_valid,
        retrieval_output=retrieval_output_valid,
    )
    metrics.update({f"reranking_valid_{k}": v for k, v in valid_metrics.items()})

    if run_test:
        if prev_output_test is None:
             logger.warning("[Reranking] No previous test output provided, cannot run test evaluation.")
        else:
            logger.info("[Reranking] Evaluating on test set...")
            test_metrics = reranking_stage.evaluate(
                prev_output_test,
                retrieval_output=retrieval_output_test,
            )
            logger.info(f"Test (Ranking): {test_metrics}")
            metrics.update({f"reranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("Skipping reranking test evaluation as requested.")

    return metrics


def save_reranking_outputs(
        reranking_stage,
        pipeline_config,
        dataset_config,
        fg_manager=None,
        logger=None,
        model_weights_path=None,
        prev_output_path=None,
        retrieval_output_path=None,
        run_test=True,
        feature_map=None,
        original_feature_map=None,
):
    """Load a trained reranking checkpoint and export valid/test reranked outputs."""
    if logger is None:
        logger = logging.getLogger('PipelineRunner')
    if not model_weights_path:
        raise RuntimeError("--model_weights_path is required for save_reranking_outputs mode.")
    if not prev_output_path:
        raise RuntimeError("--prev_output_path (preranking stage outputs) is required for save_reranking_outputs mode.")

    metrics = {}
    prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
        prev_output_path, 'preranking', logger, load_test=bool(run_test)
    )

    retrieval_output_valid, retrieval_output_test = None, None
    if retrieval_output_path:
        retrieval_output_valid, retrieval_output_test = load_stage_outputs_from_dir(
            retrieval_output_path, 'retrieval', logger, load_test=bool(run_test)
        )

    reranking_config = pipeline_config['stages']['reranking']
    cloud_teacher_config = pipeline_config['stages'].get('cloud_teacher', {})

    if cloud_teacher_config.get('mode') and feature_map is not None and fg_manager is not None:
        from cloud_device_recsys.utils import filter_feature_map

        teacher_features_str = cloud_teacher_config.get('cloud_teacher_features', ['FG1', 'FG2'])
        teacher_feature_groups = [FeatureGroup.from_string(f) for f in teacher_features_str]
        teacher_base_fm = original_feature_map if original_feature_map is not None else feature_map
        cloud_teacher_feature_map = filter_feature_map(
            teacher_base_fm, fg_manager, teacher_feature_groups,
            use_feature_encoder=cloud_teacher_config.get('cloud_teacher_model_params', {}).get(
                'use_feature_encoder',
                reranking_config.get('model_params', {}).get('use_feature_encoder', False)
            )
        )
        cloud_teacher_feature_map.default_emb_dim = cloud_teacher_config.get(
            'cloud_teacher_model_params', {}
        ).get(
            'embedding_dim',
            reranking_config.get('model_params', {}).get('embedding_dim', 16)
        )

        reranking_stage.cloud_teacher_params = cloud_teacher_config
        reranking_stage.cloud_teacher_feature_map = cloud_teacher_feature_map
        reranking_stage.cloud_teacher_mode = cloud_teacher_config.get('mode')

        if reranking_stage.cloud_teacher_mode == 'inject':
            reranking_stage.use_cloud_score = True
        elif reranking_stage.cloud_teacher_mode == 'distill':
            reranking_stage.use_cloud_score = False
            reranking_stage.kd_loss_weight = cloud_teacher_config.get('kd_loss_weight', 0.1)
            reranking_stage.kd_loss_type = cloud_teacher_config.get('kd_loss_type', 'mse')
            reranking_stage.kd_temperature = cloud_teacher_config.get('kd_temperature', 1.0)
        elif reranking_stage.cloud_teacher_mode == 'residual_inject':
            reranking_stage.use_cloud_score = False
            reranking_stage.residual_weight = cloud_teacher_config.get('residual_weight', 1.0)
        elif reranking_stage.cloud_teacher_mode == 'hybrid_inject':
            reranking_stage.use_cloud_score = True
            reranking_stage.residual_weight = cloud_teacher_config.get('residual_weight', 1.0)

    paths = get_data_paths(dataset_config, pipeline_config, logger)
    paths = prepare_debug_paths(paths, dataset_config, logger)

    if fg_manager is not None:
        ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=reranking_stage.feature_map,
        )

    reranking_stage.build_model()
    reranking_stage.model.load_weights(model_weights_path)
    reranking_stage.best_weights_path = model_weights_path
    logger.info(f"[save_reranking_outputs] Loaded pre-trained weights from: {model_weights_path}")

    if os.path.exists(paths['item_pool_path']):
        reranking_stage.load_item_features(paths['item_pool_path'])
    else:
        raise RuntimeError(f"Item pool not found at {paths['item_pool_path']}")

    if fg_manager is not None:
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        prev_output_valid = enrich_stage_output_user_features(
            prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
        )
        prev_output_test = enrich_stage_output_user_features(
            prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
        )
        if retrieval_output_valid is not None:
            retrieval_output_valid = enrich_stage_output_user_features(
                retrieval_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
            )
        if retrieval_output_test is not None:
            retrieval_output_test = enrich_stage_output_user_features(
                retrieval_output_test, paths['test_path'], fg_manager, impression_id_col, logger
            )

    vocab_pruning_config = pipeline_config.get('vocab_pruning', {})
    if vocab_pruning_config.get('enabled', False) and vocab_pruning_config.get('mode', 'runtime') == 'offline':
        data_dir = get_data_dir(dataset_config)
        remap_dicts = load_offline_remap_dicts(data_dir, logger)
        if remap_dicts is not None:
            from cloud_device_recsys.data.remap_vocab_data import remap_stage_output
            reranking_stage.remap_dicts = remap_dicts
            if prev_output_valid is not None:
                prev_output_valid = remap_stage_output(prev_output_valid, remap_dicts, feature_map)
            if prev_output_test is not None:
                prev_output_test = remap_stage_output(prev_output_test, remap_dicts, feature_map)
            if retrieval_output_valid is not None:
                retrieval_output_valid = remap_stage_output(retrieval_output_valid, remap_dicts, feature_map)
            if retrieval_output_test is not None:
                retrieval_output_test = remap_stage_output(retrieval_output_test, remap_dicts, feature_map)

    logger.info("[save_reranking_outputs] Processing valid set...")
    valid_output, _ = reranking_stage.process(prev_output_valid, compute_metrics=False)
    valid_metrics = reranking_stage.evaluate(
        prev_output_valid,
        retrieval_output=retrieval_output_valid,
    )
    metrics.update({f"reranking_valid_{k}": v for k, v in valid_metrics.items()})

    test_output = None
    if run_test:
        if prev_output_test is None:
            logger.warning("[save_reranking_outputs] No preranking test output provided, skipping test export.")
        else:
            logger.info("[save_reranking_outputs] Processing test set...")
            test_output, _ = reranking_stage.process(prev_output_test, compute_metrics=False)
            test_metrics = reranking_stage.evaluate(
                prev_output_test,
                retrieval_output=retrieval_output_test,
            )
            metrics.update({f"reranking_test_{k}": v for k, v in test_metrics.items()})
    else:
        logger.info("[save_reranking_outputs] Skipping reranking test evaluation.")

    return metrics, valid_output, test_output


def main():
    """Main entry point"""
    args = parse_pipeline_args()
    # Both IDs are used to construct filesystem paths. Validate before any
    # output directory or configuration path is created.
    args.pipeline_id = validate_pipeline_id(args.pipeline_id)
    if args.experiment_id:
        args.experiment_id = validate_experiment_id(args.experiment_id)

    # Construct unique output directory for this run
    run_output_base = args.output_dir  # e.g. ./outputs
    logger = logging.getLogger('PipelineRunner')
    if args.experiment_id:
        logger.info(f"Experiment ID: {args.experiment_id}")
        run_output_dir = os.path.join(run_output_base, args.experiment_id)
    else:
        pipeline_slug = args.pipeline_id.replace("/", "__")
        run_output_dir = os.path.join(
            run_output_base,
            f"{pipeline_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(run_output_dir, exist_ok=True)
    stage_output_dir = f"{run_output_dir}/stage_outputs"

    # Setup logging to the unique output directory
    setup_logging(run_output_dir)

    logger.info(f"Starting pipeline with mode: {args.mode}")
    logger.info(f"Configuration: {args.config}, Pipeline: {args.pipeline_id}")
    logger.info(f"Output Directory: {run_output_dir}")

    seed_everything(args.seed)

    # Load configurations using ConfigParser
    config_parser = ConfigParser(config_dir=args.config)
    full_config = config_parser.get_full_config(
        pipeline_path=resolve_pipeline_config_path(args.config, args.pipeline_id)
        if args.pipeline_id != 'default' else None,
        dataset_path=os.path.join(args.config, 'dataset_config.yaml'),
        dataset_id=args.dataset_id
    )

    pipeline_config = full_config['pipeline']
    dataset_config = full_config['dataset']

    # Ensure dataset_id in pipeline_config is consistent
    pipeline_config['dataset_id'] = dataset_config['dataset_id']

    config_save_stage_outputs = bool(
        pipeline_config.get('output', {}).get('save_intermediate', False)
    )
    if args.save_stage_outputs is None:
        args.save_stage_outputs = config_save_stage_outputs
        logger.info(
            "save_stage_outputs not set via CLI. Using config output.save_intermediate=%s",
            args.save_stage_outputs,
        )
    else:
        logger.info("save_stage_outputs overridden via CLI: %s", args.save_stage_outputs)

    # --- Apply Retrieval Stage Overrides ---
    if 'retrieval' in pipeline_config.get('stages', {}):
        r_stage = pipeline_config['stages']['retrieval']
        r_params = r_stage.get('model_params', {})
        r_train = r_stage.get('training', {})

        # Write back
        pipeline_config['stages']['retrieval']['model_params'] = r_params
        pipeline_config['stages']['retrieval']['training'] = r_train

    # --- Save Run Configuration ---
    run_config_path = os.path.join(run_output_dir, 'run_config.yaml')
    if args.n_rows is not None:
        if 'debug' not in pipeline_config:
            pipeline_config['debug'] = {}
        pipeline_config['debug']['n_rows'] = args.n_rows
        logger.info(f"Override debug n_rows: {args.n_rows}")

    with open(run_config_path, 'w') as f:
        yaml.dump(full_config, f, indent=2)
    logger.info(f"Saved run configuration to {run_config_path}")

    dataset_config['gpu'] = args.gpu
    logger.info(f"Used GPU: {args.gpu if args.gpu >= 0 else 'CPU'}")

    # Build feature map
    # Use processed_data_root if available, otherwise data_root + dataset_id
    data_dir = get_data_dir(dataset_config, pipeline_config['dataset_id'])

    feature_map_json = dataset_config.get('feature_map_path') or os.path.join(data_dir, "feature_map.json")
    feature_map_json = os.path.expanduser(os.path.expandvars(feature_map_json))
    data_format = str(
        dataset_config.get('processed_data_format', dataset_config.get('data_format', ''))
    ).lower()
    if data_format in {'tfrecord', 'tf_record'}:
        feature_map_json = ensure_fuxictr_feature_map(
            feature_map_path=feature_map_json,
            dataset_config=dataset_config,
            pipeline_config=pipeline_config,
            data_dir=data_dir,
            logger=logger,
            output_dir=run_output_dir,
        )

    if os.path.exists(feature_map_json):
        # Convert dict-format feature_map to FuxiCTR's list format if needed
        with open(feature_map_json, 'r') as f:
            fm_data = json.load(f)

        # Check if features is a dict (our format) vs list (FuxiCTR format)
        if isinstance(fm_data.get('features'), dict):
            logger.info("Converting dict-format feature_map to FuxiCTR list format...")
            # Convert {"name": {...}, ...} to [{"name": {...}}, ...]
            fm_data['features'] = [{k: v} for k, v in fm_data['features'].items()]

            # Write converted format back
            converted_json = os.path.join(data_dir, "feature_map_fuxictr.json")
            with open(converted_json, 'w') as f:
                json.dump(fm_data, f, indent=2)
            feature_map_json = converted_json
            logger.info(f"Saved converted feature_map to {converted_json}")

        feature_map = FeatureMap(pipeline_config['dataset_id'], data_dir)
        feature_map.load(feature_map_json, dataset_config)
        feature_map.dataset_config = dataset_config  # Attach config for access in stages
        # --- Critical Fix: Ensure impression_id is loaded by DataLoaders ---
        # Add impression_id_col to labels so it gets loaded but not treated as a feature
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        if impression_id_col not in feature_map.labels:
            logger.info(f"Adding '{impression_id_col}' to feature_map.labels to ensure loading.")
            feature_map.labels.append(impression_id_col)

        logger.info(f"Loaded feature map with {len(feature_map.features)} features")

        # --- Vocabulary Pruning (optional) ---
        # Reduces embedding size from values observed in training only.  Holdout
        # values must remain unseen/OOV instead of influencing the feature map.
        # Preserve the original feature_map before pruning may replace it.
        # The cloud teacher model needs unpruned vocab sizes to match its checkpoint.
        original_feature_map = copy.deepcopy(feature_map)
        vocab_pruning_config = pipeline_config.get('vocab_pruning', {})
        vocab_pruning_mode = vocab_pruning_config.get('mode', 'runtime')  # 'runtime' or 'offline'
        save_fp16 = vocab_pruning_config.get('save_fp16', False)

        if vocab_pruning_config.get('enabled', False) and vocab_pruning_mode == 'offline':
            # Offline mode: use pre-mapped data from mapped/ subdirectory
            mapped_dir = os.path.join(data_dir, 'mapped')
            mapped_feature_map_json = os.path.join(mapped_dir, 'feature_map.json')
            if os.path.exists(mapped_feature_map_json):
                logger.info(f"[VocabPruner] Offline mode: using pre-mapped data from {mapped_dir}")
                # Reload feature_map from mapped directory with compact vocab sizes
                data_dir = mapped_dir
                feature_map = FeatureMap(pipeline_config['dataset_id'], mapped_dir)
                feature_map.load(mapped_feature_map_json, dataset_config)
                feature_map.dataset_config = dataset_config
                impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
                if impression_id_col not in feature_map.labels:
                    feature_map.labels.append(impression_id_col)
                # No runtime remapping needed — data is already compact
                feature_map._vocab_prune_info = None
                # Override processed_data_root so all downstream get_data_paths() use mapped dir
                dataset_config['processed_data_root'] = mapped_dir
                logger.info(f"[VocabPruner] Offline mode: loaded mapped feature_map with {len(feature_map.features)} features")
            else:
                logger.warning(f"[VocabPruner] Offline mode requested but {mapped_feature_map_json} not found. "
                               f"Run `python -m cloud_device_recsys.data.remap_vocab_data --data_dir {data_dir}` first.")
                feature_map._vocab_prune_info = None

        elif vocab_pruning_config.get('enabled', False):
            # Runtime mode: current behavior with RemappedEmbedding
            from cloud_device_recsys.data.vocab_pruner import compute_vocab_pruning
            scan_paths = _get_train_vocab_scan_paths(data_dir)
            logger.info(
                "[VocabPruner] Will scan training split only for vocabulary usage: %s",
                scan_paths[0],
            )
            prune_info = compute_vocab_pruning(
                feature_map,
                data_paths=scan_paths,
                min_vocab_size=vocab_pruning_config.get('min_vocab_size', 100),
                min_reduction_ratio=vocab_pruning_config.get('min_reduction_ratio', 0.3),
                cache_dir=data_dir,
            )
            # Attach to feature_map for use in model building (registry.py)
            feature_map._vocab_prune_info = prune_info
            logger.info(f"[VocabPruner] Pruning complete. {len(prune_info.features)} features pruned.")
        else:
            feature_map._vocab_prune_info = None

        # Propagate FP16 saving flag to feature_map for registry.py to pick up
        feature_map._save_fp16 = save_fp16
        logger.info(f"Using save_fp16={save_fp16} for embedding weights based on config")

    else:
        logger.warning(f"Feature map not found at {feature_map_json}")
        logger.info("Please run data preprocessing first")
        return

    # Build feature group manager
    fg_manager = build_feature_group_manager(pipeline_config)
    # Pass dataset_config to use explicit feature group definitions
    fg_manager.auto_assign_groups(feature_map, dataset_config)

    stats_paths = get_data_paths(dataset_config, pipeline_config, logger)
    stats_paths = prepare_debug_paths(stats_paths, dataset_config, logger)
    dataset_statistics_path = write_dataset_statistics(dataset_config, stats_paths, run_output_dir, logger)
    if dataset_statistics_path:
        dataset_config["_dataset_statistics_path"] = dataset_statistics_path

    # Create stages
    stages = create_stages(
        feature_map=feature_map,
        feature_group_manager=fg_manager,
        config=pipeline_config,
        output_dir=run_output_dir,
        gpu=args.gpu
    )

    all_metrics = {}
    metrics_path = os.path.join(run_output_dir, 'metrics.json')
    # Execute based on mode
    if args.mode == 'full':
        logger.info("Running full pipeline")
        # Instantiate Retrieval Stage lazily
        retrieval_stage = stages['retrieval']()
        # Run retrieval stage with its own data loaders
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger,
            run_test=bool(args.run_retrieval_test)
        )
        all_metrics.update(r_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save retrieval stage outputs if requested
        if args.save_stage_outputs:
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            if r_test is not None:
                save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)
            logger.info(f"Saved retrieval_valid and retrieval_test metrics to {metrics_path}")

        # Instantiate Preranking Stage lazily (after retrieval)
        preranking_stage = stages['preranking']()
        # Run preranking stage with its own data loaders
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
            logger=logger, prev_output_valid=r_valid, prev_output_test=r_test,
            run_test=bool(args.run_preranking_test),
        )
        all_metrics.update(p_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        # Save preranking stage outputs if requested
        if args.save_stage_outputs:
            if p_valid is not None:
                save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            else:
                logger.info("No preranking_valid StageOutput to save (likely pointwise evaluation mode).")
            if p_test is not None:
                save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)

        # Instantiate Reranking Stage lazily (after preranking)
        if p_valid is None:
            logger.warning("Skipping reranking because preranking returned no StageOutput.")
        else:
            reranking_stage = stages['reranking']()
            # Run reranking stage with its own data loaders
            d_metrics = run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
                                            logger=logger, run_test=bool(args.run_reranking_test),
                                            prev_output_path=args.prev_output_path, preranking_model=preranking_stage.model,
                                            prev_output_valid=p_valid, prev_output_test=p_test,
                                            retrieval_output_valid=r_valid, retrieval_output_test=r_test,
                                            feature_map=feature_map, original_feature_map=original_feature_map)
            all_metrics.update(d_metrics)

    elif args.mode == 'retrieval':
        logger.info("Running retrieval stage only")
        if 'retrieval' not in stages:
            raise RuntimeError("No retrieval stage found. Please run data preprocessing first")
        # Instantiate Retrieval Stage lazily
        retrieval_stage = stages['retrieval']()
        r_metrics, r_valid, r_test = run_retrieval_stage(
            retrieval_stage, pipeline_config, dataset_config, fg_manager, logger=logger,
            run_test=bool(args.run_retrieval_test)
        )
        all_metrics.update(r_metrics)

        # Save stage outputs if requested
        if args.save_stage_outputs:
            stage_output_dir = os.path.join(run_output_dir, 'stage_outputs')
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            if r_test is not None:
                save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)

    elif args.mode == 'save_retrieval_outputs':
        logger.info("Running save_retrieval_outputs: loading pre-trained model and generating stage outputs")
        if 'retrieval' not in stages:
            raise RuntimeError("No retrieval stage found in config.")
        if not args.model_weights_path:
            raise RuntimeError("--model_weights_path is required for save_retrieval_outputs mode.")

        retrieval_stage = stages['retrieval']()
        r_metrics, r_valid, r_test = save_retrieval_outputs(
            retrieval_stage,
            pipeline_config,
            dataset_config,
            fg_manager,
            model_weights_path=args.model_weights_path,
            logger=logger,
            run_test=bool(args.run_retrieval_test),
        )
        all_metrics.update(r_metrics)

        if args.save_stage_outputs:
            save_stage_output(r_valid, stage_output_dir, 'retrieval_valid', logger)
            if r_test is not None:
                save_stage_output(r_test, stage_output_dir, 'retrieval_test', logger)

    elif args.mode == 'preranking':
        logger.info("Running preranking stage only")
        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found. Please run retrieval first")

        preranking_config = pipeline_config['stages']['preranking']
        preranking_eval_mode = str(
            (preranking_config.get('evaluation') or {}).get('mode', 'listwise')
        ).lower()
        preranking_pointwise_eval = preranking_eval_mode in {'pointwise', 'samplewise', 'observed'}
        preranking_ranking_eval_enabled = _to_bool(
            (preranking_config.get('evaluation') or {}).get('ranking_metrics_enabled'),
            False,
        )
        preranking_needs_candidates = (not preranking_pointwise_eval) or preranking_ranking_eval_enabled

        # Load previous stage outputs if provided (from retrieval stage)
        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger, load_test=bool(args.run_preranking_test)
            )
        elif preranking_pointwise_eval and not preranking_ranking_eval_enabled:
            logger.info(
                "No retrieval stage outputs provided and preranking evaluation mode is pointwise; "
                "skipping candidate bootstrap."
            )
            prev_output_valid, prev_output_test = None, None
        else:
            logger.warning(
                "No retrieval stage outputs provided via --prev_output_path. "
                "Bootstrapping preranking candidates from dataset rows and configured item pool "
                "(ranking_metrics_enabled=%s).",
                preranking_ranking_eval_enabled,
            )
            bootstrap_paths = get_data_paths(dataset_config, pipeline_config, logger)
            bootstrap_paths = prepare_debug_paths(bootstrap_paths, dataset_config, logger)
            prev_output_valid = build_observed_candidate_stage_output(
                feature_map=feature_map,
                dataset_config=dataset_config,
                pipeline_config=pipeline_config,
                fg_manager=fg_manager,
                split_name='valid',
                data_path=bootstrap_paths.get('valid_path'),
                logger=logger,
            )
            prev_output_test = None
            if bool(args.run_preranking_test):
                prev_output_test = build_observed_candidate_stage_output(
                    feature_map=feature_map,
                    dataset_config=dataset_config,
                    pipeline_config=pipeline_config,
                    fg_manager=fg_manager,
                    split_name='test',
                    data_path=bootstrap_paths.get('test_path'),
                    logger=logger,
                )
            valid_bootstrap_requests = (
                0
                if prev_output_valid is None
                else len(prev_output_valid.user_features_df)
                if (prev_output_valid.metadata or {}).get('lazy_all_items')
                else prev_output_valid.get_total_candidates()
            )
            if preranking_needs_candidates and (prev_output_valid is None or valid_bootstrap_requests == 0):
                raise RuntimeError(
                    "Could not bootstrap preranking candidates from validation data. "
                    "Provide --prev_output_path with retrieval outputs or check valid_data."
                )

        # Instantiate Preranking Stage lazily
        preranking_stage = stages['preranking']()
        p_metrics, p_valid, p_test = run_preranking_stage(
            preranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
            logger=logger, prev_output_valid=prev_output_valid, prev_output_test=prev_output_test,
            run_test=bool(args.run_preranking_test)
        )
        all_metrics.update(p_metrics)

        # Save stage outputs if requested
        if args.save_stage_outputs:
            if p_valid is not None:
                save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
            else:
                logger.info("No preranking_valid StageOutput to save (likely pointwise evaluation mode).")
            if p_test is not None:
                save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)

    elif args.mode == 'reranking':
        logger.info("Running reranking stage only")
        if 'reranking' not in stages:
            raise RuntimeError("No reranking stage found. Please run preranking first")

        # Load retrieval stage outputs for evaluation scope alignment
        retrieval_output_valid, retrieval_output_test = None, None
        if args.retrieval_output_path:
            try:
                retrieval_output_valid, retrieval_output_test = load_stage_outputs_from_dir(
                    args.retrieval_output_path, 'retrieval', logger,
                    load_test=bool(args.run_reranking_test)
                )
            except Exception as e:
                logger.warning(f"Could not load retrieval outputs from {args.retrieval_output_path}: {e}. "
                               "Evaluation scope alignment will be disabled.")

        # Instantiate Reranking Stage lazily
        reranking_stage = stages['reranking']()
        d_metrics = run_reranking_stage(reranking_stage, pipeline_config, dataset_config, fg_manager=fg_manager,
                                        logger=logger, run_test=bool(args.run_reranking_test),
                                        prev_output_path=args.prev_output_path,
                                        feature_map=feature_map, original_feature_map=original_feature_map,
                                        retrieval_output_valid=retrieval_output_valid,
                                        retrieval_output_test=retrieval_output_test)
        all_metrics.update(d_metrics)

    elif args.mode == 'save_reranking_outputs':
        logger.info("Running save_reranking_outputs: loading pre-trained model and generating stage outputs")
        if 'reranking' not in stages:
            raise RuntimeError("No reranking stage found in config.")
        if not args.model_weights_path:
            raise RuntimeError("--model_weights_path is required for save_reranking_outputs mode.")
        if not args.prev_output_path:
            raise RuntimeError("--prev_output_path (preranking stage outputs) is required for save_reranking_outputs mode.")

        reranking_stage = stages['reranking']()
        d_metrics, d_valid, d_test = save_reranking_outputs(
            reranking_stage,
            pipeline_config,
            dataset_config,
            fg_manager=fg_manager,
            logger=logger,
            model_weights_path=args.model_weights_path,
            prev_output_path=args.prev_output_path,
            retrieval_output_path=args.retrieval_output_path,
            run_test=bool(args.run_reranking_test),
            feature_map=feature_map,
            original_feature_map=original_feature_map,
        )
        all_metrics.update(d_metrics)

        if args.save_stage_outputs:
            save_stage_output(d_valid, stage_output_dir, 'reranking_valid', logger)
            if d_test is not None:
                save_stage_output(d_test, stage_output_dir, 'reranking_test', logger)

    elif args.mode == 'joint_train':
        logger.info("Running Cloud-Device Joint Training")
        if 'preranking' not in stages or 'reranking' not in stages:
            raise RuntimeError("Both preranking and reranking stages must be defined to run joint training")

        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger
            )
        else:
            raise RuntimeError("Previous stage outputs (from retrieval) must be provided via --prev_output_path for joint training validation.")

        preranking_stage = stages['preranking']()
        reranking_stage = stages['reranking']()

        j_metrics = run_joint_training_stage(
            preranking_stage, reranking_stage, pipeline_config, dataset_config, fg_manager,
            logger=logger, run_test=bool(args.run_reranking_test),
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test
        )
        all_metrics.update(j_metrics)

    elif args.mode == 'dtcn_preranking':
        logger.info("Running DTCN Cloud-Side Preranking")
        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found in config.")

        if args.prev_output_path:
            prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
                args.prev_output_path, 'retrieval', logger, load_test=bool(args.run_preranking_test)
            )
        else:
            raise RuntimeError("Previous stage outputs (from retrieval) must be provided via --prev_output_path.")

        d_metrics, _, _ = run_dtcn_preranking_stage(
            pipeline_config, dataset_config, fg_manager, feature_map,
            logger=logger, run_test=bool(args.run_preranking_test),
            prev_output_valid=prev_output_valid, prev_output_test=prev_output_test,
            gpu=args.gpu, output_dir=run_output_dir,
        )
        all_metrics.update(d_metrics)

    elif args.mode == 'save_preranking_outputs':
        logger.info("Running save_preranking_outputs: loading pre-trained model and generating stage outputs")

        if 'preranking' not in stages:
            raise RuntimeError("No preranking stage found in config.")

        if not args.model_weights_path:
            raise RuntimeError("--model_weights_path is required for save_preranking_outputs mode.")

        if not args.prev_output_path:
            raise RuntimeError("--prev_output_path (retrieval stage outputs) is required for save_preranking_outputs mode.")

        # 1. Load retrieval stage outputs
        prev_output_valid, prev_output_test = load_stage_outputs_from_dir(
            args.prev_output_path, 'retrieval', logger, load_test=bool(args.run_preranking_test)
        )

        # 2. Instantiate preranking stage and build model architecture
        preranking_stage = stages['preranking']()
        preranking_stage.build_model()

        # 3. Load pre-trained weights (following _build_cloud_teacher pattern)
        preranking_stage.model.load_weights(args.model_weights_path)
        preranking_stage.best_weights_path = args.model_weights_path
        logger.info(f"Loaded pre-trained weights from: {args.model_weights_path}")

        # 4. Prepare data paths and load item features
        paths = get_data_paths(dataset_config, pipeline_config, logger)
        paths = prepare_debug_paths(paths, dataset_config, logger)

        cand_item_pool_path = ensure_item_pool(
            data_paths={'item_pool_path': paths['item_pool_path'], 'valid_path': paths['valid_path'],
                        'test_path': paths['test_path']},
            dataset_config=dataset_config,
            feature_group_manager=fg_manager,
            logger=logger,
            feature_map=preranking_stage.feature_map,
        )

        if os.path.exists(cand_item_pool_path):
            preranking_stage.load_item_features(cand_item_pool_path)
        elif os.path.exists(paths['item_pool_path']):
            preranking_stage.load_item_features(paths['item_pool_path'])
        else:
            raise RuntimeError(f"Item pool not found. Checked: {cand_item_pool_path}, {paths['item_pool_path']}")

        # 5. Enrich stage outputs with FG3 user features (backward compatibility)
        if fg_manager is not None:
            impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
            prev_output_valid = enrich_stage_output_user_features(
                prev_output_valid, paths['valid_path'], fg_manager, impression_id_col, logger
            )
            prev_output_test = enrich_stage_output_user_features(
                prev_output_test, paths['test_path'], fg_manager, impression_id_col, logger
            )

        # 6. Run process (inference only, no training)
        logger.info("[save_preranking_outputs] Processing valid set...")
        p_valid, valid_metrics = preranking_stage.process(prev_output_valid, compute_metrics=True, load_best_model=False)
        logger.info(f"Valid (Preranking): {valid_metrics}")
        all_metrics.update({f"preranking_valid_{k}": v for k, v in valid_metrics.items()})

        p_test = None
        if args.run_preranking_test and prev_output_test is not None:
            logger.info("[save_preranking_outputs] Processing test set...")
            p_test, test_metrics = preranking_stage.process(prev_output_test, compute_metrics=True, load_best_model=False)
            all_metrics.update({f"preranking_test_{k}": v for k, v in test_metrics.items()})
            logger.info(f"Valid (Preranking): {test_metrics}")

        # 7. Save stage outputs
        save_stage_output(p_valid, stage_output_dir, 'preranking_valid', logger)
        if p_test is not None:
            save_stage_output(p_test, stage_output_dir, 'preranking_test', logger)
        logger.info(f"[save_preranking_outputs] Stage outputs saved to {stage_output_dir}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save final metrics
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
