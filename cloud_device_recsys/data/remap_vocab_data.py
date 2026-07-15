# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Offline Vocabulary Remapping

Scans training data to build compact vocabulary mappings,
then rewrites all parquet files and feature_map.json into a `mapped/`
subdirectory. This eliminates the need for runtime remap_table buffers
in the on-device model, saving ~7MB of model size on TaobaoAd.

Usage:
    python -m cloud_device_recsys.data.remap_vocab_data \
        --data_dir /path/to/processed/TaobaoAd \
        --min_vocab_size 100000 \
        --min_reduction_ratio 0.3

Original data files are NEVER modified.
"""

import argparse
import copy
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from cloud_device_recsys.data.vocab_pruner import (
    scan_parquet_vocab,
    build_vocab_mapping,
    VocabPruneInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Files to remap (relative to data_dir)
_PARQUET_FILES = [
    'train.parquet',
    'train_positive.parquet',
    'valid.parquet',
    'test.parquet',
    'cand_items_test.parquet',
    'cand_items_all.parquet',
    # Also handle any *_full.parquet variants
    'valid_full.parquet',
    'test_full.parquet',
    'cand_items_test_full.parquet',
]


def build_remap_dicts(prune_info: VocabPruneInfo) -> Dict[str, Dict[int, int]]:
    """
    Convert remap_table tensors to plain Python dicts for fast pandas .map().

    Only stores non-zero mappings (i.e., values that are actually kept).

    Returns:
        Dict[feature_name -> Dict[old_id -> new_compact_id]]
    """
    remap_dicts = {}
    seen_bases = set()

    for feature_name, info in prune_info.features.items():
        base = info.feature_name
        if base in seen_bases:
            # Shared embedding: reuse the same dict
            remap_dicts[feature_name] = remap_dicts[base]
            continue

        table = info.remap_table.numpy()
        # Build dict only for non-zero mappings (0 = padding/OOV)
        nonzero_mask = table != 0
        old_indices = np.where(nonzero_mask)[0]
        new_indices = table[nonzero_mask]
        remap_dict = dict(zip(old_indices.tolist(), new_indices.tolist()))

        remap_dicts[base] = remap_dict
        remap_dicts[feature_name] = remap_dict
        seen_bases.add(base)

    return remap_dicts


def save_remap_dicts_json(remap_dicts: Dict[str, Dict[int, int]], output_path: str) -> None:
    """Write remaps as a data-only JSON artifact consumable by pipeline loading.

    JSON object keys are strings, so integer source IDs are encoded as decimal
    strings. ``load_remap_dicts_json`` restores both source and destination IDs
    to ``int`` and rejects malformed values before a pipeline uses them.
    """
    serialized = {}
    for feature_name, mapping in remap_dicts.items():
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("Remap feature names must be non-empty strings")
        if not isinstance(mapping, dict):
            raise ValueError(f"Remap for feature {feature_name!r} must be a dictionary")
        serialized_mapping = {}
        for old_id, new_id in mapping.items():
            if isinstance(old_id, bool) or isinstance(new_id, bool):
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
            serialized_mapping[str(old_id_int)] = new_id_int
        serialized[feature_name] = serialized_mapping

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".remap_dict.", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_path, output_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def remap_parquet_file(
    src_path: str,
    dst_path: str,
    remap_dicts: Dict[str, Dict[int, int]],
    feature_specs: Dict[str, dict],
) -> bool:
    """
    Read a parquet file, remap feature columns, write to dst_path.

    For categorical features: direct .map() replacement.
    For sequence features: element-wise remapping within each list/array.

    Returns:
        True if file was processed, False if skipped (src doesn't exist).
    """
    if not os.path.exists(src_path):
        return False

    logger.info(f"  Remapping {os.path.basename(src_path)} → {os.path.basename(dst_path)}")
    df = pd.read_parquet(src_path)

    for col_name, remap_dict in remap_dicts.items():
        if col_name not in df.columns:
            continue

        spec = feature_specs.get(col_name, {})
        feat_type = spec.get('type', 'categorical')

        if feat_type == 'sequence':
            # Sequence column: each cell is a list/array of IDs
            def remap_sequence(seq):
                if seq is None or (isinstance(seq, float) and np.isnan(seq)):
                    return seq
                return np.array([remap_dict.get(int(v), 0) for v in seq],
                                dtype=np.int32)
            df[col_name] = df[col_name].apply(remap_sequence)
        else:
            # Categorical column: scalar values
            original = df[col_name]
            df[col_name] = original.map(remap_dict).fillna(0).astype(np.int32)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_parquet(dst_path, index=False)
    logger.info(f"    Wrote {len(df)} rows to {dst_path}")
    return True


def remap_feature_map(
    feature_map_path: str,
    output_path: str,
    prune_info: VocabPruneInfo,
):
    """
    Create a new feature_map.json with compact vocab_size values.
    """
    with open(feature_map_path, 'r') as f:
        fm_data = json.load(f)

    new_fm = copy.deepcopy(fm_data)

    # Update total_features
    total_saved = 0
    for feat_entry in new_fm['features']:
        for feat_name, spec in feat_entry.items():
            if feat_name in prune_info.features:
                info = prune_info.features[feat_name]
                old_size = spec.get('vocab_size', 0)
                spec['vocab_size'] = info.compact_vocab_size
                total_saved += old_size - info.compact_vocab_size

    new_fm['total_features'] = fm_data.get('total_features', 0) - total_saved

    with open(output_path, 'w') as f:
        json.dump(new_fm, f, indent=4)

    logger.info(f"  Wrote mapped feature_map.json → {output_path}")
    logger.info(f"  total_features: {fm_data.get('total_features', 0)} → {new_fm['total_features']} "
                f"(saved {total_saved:,})")


def load_feature_map_for_scan(feature_map_path: str):
    """
    Load feature_map.json and convert to the dict format expected by
    scan_parquet_vocab / build_vocab_mapping.

    Returns a lightweight object with `.features` dict attribute.
    """
    with open(feature_map_path, 'r') as f:
        fm_data = json.load(f)

    # FuxiCTR feature_map.json has features as a list of single-key dicts
    features = {}
    for entry in fm_data['features']:
        for name, spec in entry.items():
            features[name] = spec

    class _FeatureMapProxy:
        pass

    proxy = _FeatureMapProxy()
    proxy.features = features
    proxy.data_dir = os.path.dirname(feature_map_path)
    return proxy


def run_offline_remap(
    data_dir: str,
    output_subdir: str = 'mapped',
    min_vocab_size: int = 100000,
    min_reduction_ratio: float = 0.3,
    scan_paths: Optional[List[str]] = None,
):
    """
    Main entry point: scan data, build mappings, remap all files.

    Args:
        data_dir: Path to processed dataset root (e.g. /processed/TaobaoAd)
        output_subdir: Subdirectory name for mapped output (default: 'mapped')
        min_vocab_size: Minimum vocab size to consider for remapping
        min_reduction_ratio: Minimum reduction ratio to trigger remapping
        scan_paths: Optional explicit list of paths to scan for vocabulary.
                    If None, scans train_positive when present, otherwise train.
                    Explicit paths are the caller's responsibility and may include
                    another pre-approved, time-valid catalog if required.
    """
    output_dir = os.path.join(data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    feature_map_path = os.path.join(data_dir, 'feature_map.json')
    if not os.path.exists(feature_map_path):
        raise FileNotFoundError(f"feature_map.json not found at {feature_map_path}")

    # Step 1: Load feature map
    logger.info("=== Offline Vocabulary Remapping ===")
    logger.info(f"  Source: {data_dir}")
    logger.info(f"  Output: {output_dir}")

    feature_map_proxy = load_feature_map_for_scan(feature_map_path)

    # Step 2: Determine scan paths
    if scan_paths is None:
        train_pos = os.path.join(data_dir, 'train_positive.parquet')
        train_full = os.path.join(data_dir, 'train.parquet')
        train_path = train_pos if os.path.exists(train_pos) else train_full
        scan_paths = [train_path]
        logger.info("  Using default train-only vocabulary scan: %s", train_path)
    else:
        logger.info("  Using %d caller-provided vocabulary scan path(s).", len(scan_paths))

    logger.info(f"  Scanning {len(scan_paths)} path(s) for vocabulary...")

    # Step 3: Scan vocabulary
    actual_vocab = scan_parquet_vocab(scan_paths, feature_map_proxy)

    # Step 4: Build remap tables
    prune_info = build_vocab_mapping(
        actual_vocab, feature_map_proxy,
        min_vocab_size=min_vocab_size,
        min_reduction_ratio=min_reduction_ratio,
    )

    if not prune_info.features:
        logger.info("No features qualify for remapping. Nothing to do.")
        return output_dir

    # Step 5: Convert to Python dicts for fast remapping
    remap_dicts = build_remap_dicts(prune_info)

    logger.info(f"\n--- Remapping {len(remap_dicts)} features across parquet files ---")

    # Step 6: Remap all parquet files
    processed_count = 0
    for filename in _PARQUET_FILES:
        src = os.path.join(data_dir, filename)
        dst = os.path.join(output_dir, filename)
        if remap_parquet_file(src, dst, remap_dicts, feature_map_proxy.features):
            processed_count += 1

    logger.info(f"\n--- Remapped {processed_count} parquet files ---")

    # Step 7: Write mapped feature_map.json
    remap_feature_map(
        feature_map_path,
        os.path.join(output_dir, 'feature_map.json'),
        prune_info,
    )

    # Step 8: Save a data-only remap artifact for cloud-side inference use.
    remap_dict_path = os.path.join(output_dir, 'remap_dict.json')
    save_remap_dicts_json(remap_dicts, remap_dict_path)
    logger.info(f"  Saved remap dict → {remap_dict_path}")

    # Summary
    logger.info("\n=== Offline Remapping Complete ===")
    logger.info(f"  {processed_count} parquet files remapped")
    logger.info(f"  {len(prune_info.features)} features remapped")
    for name, info in prune_info.features.items():
        if name == info.feature_name:  # Only print base features
            logger.info(f"    {name}: {info.original_vocab_size} → {info.compact_vocab_size} "
                        f"({info.reduction_ratio:.1%} reduction)")
    logger.info(f"  Output directory: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Offline vocabulary remapping for on-device models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed dataset directory (e.g. /processed/TaobaoAd)')
    parser.add_argument('--output_subdir', type=str, default='mapped',
                        help='Subdirectory name for mapped output (default: mapped)')
    parser.add_argument('--min_vocab_size', type=int, default=100000,
                        help='Minimum vocab size to consider for remapping (default: 100000)')
    parser.add_argument('--min_reduction_ratio', type=float, default=0.3,
                        help='Minimum reduction ratio to trigger remapping (default: 0.3)')
    args = parser.parse_args()

    run_offline_remap(
        data_dir=args.data_dir,
        output_subdir=args.output_subdir,
        min_vocab_size=args.min_vocab_size,
        min_reduction_ratio=args.min_reduction_ratio,
    )


def remap_stage_output(
    stage_output: Any,
    remap_dicts: Dict[str, Dict[int, int]],
    feature_map: Any
) -> Any:
    """
    Apply vocabulary remapping to the DataFrames inside a StageOutput.
    This is used when evaluating a reranking model (trained with compact mapped ids)
    on raw candidates (unmapped ids) generated by a previous unmapped stage.

    Args:
        stage_output: The incoming StageOutput containing raw unmapped IDs.
        remap_dicts: Loaded from mapped/remap_dict.json
        feature_map: FeatureMap proxy, used to know which columns hold what features.

    Returns:
        The exact same StageOutput instance modified in-place with its IDs mapped.
    """
    # 1. Remap candidates_df (item features like 'item_id')
    candidates_df = stage_output.candidates_df
    item_id_col = getattr(feature_map, 'dataset_config', {}).get('item_id_col', 'item_id')

    # In StageOutput, the primary key column for items is universally named 'item_id'.
    # But in remap_dicts (and feature_map), the vocabulary mapping is keyed by the dataset's actual ID feature name (e.g., 'adgroup_id').
    if 'item_id' in candidates_df.columns and item_id_col in remap_dicts:
        remap_dict = remap_dicts[item_id_col]
        candidates_df['item_id'] = candidates_df['item_id'].map(remap_dict).fillna(0).astype(np.int32)

    for col_name in candidates_df.columns:
        if col_name != 'item_id' and col_name in remap_dicts:
            # Usually cand_items are separate, but if candidates_df contains extra categorical cols
            # apply remapping
            spec = feature_map.features.get(col_name, {})
            feat_type = spec.get('type', 'categorical')
            remap_dict = remap_dicts[col_name]

            if feat_type == 'sequence':
                def remap_sequence(seq):
                    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
                        return seq
                    return np.array([remap_dict.get(int(v), 0) for v in seq], dtype=np.int32)
                candidates_df[col_name] = candidates_df[col_name].apply(remap_sequence)
            else:
                original = candidates_df[col_name]
                candidates_df[col_name] = original.map(remap_dict).fillna(0).astype(np.int32)

    # 2. Remap user_features_df
    user_features_df = stage_output.user_features_df
    if user_features_df is not None and not user_features_df.empty:
        for col_name in user_features_df.columns:
            if col_name in remap_dicts:
                spec = feature_map.features.get(col_name, {})
                feat_type = spec.get('type', 'categorical')
                remap_dict = remap_dicts[col_name]

                if feat_type == 'sequence':
                    def remap_sequence(seq):
                        if seq is None or (isinstance(seq, float) and np.isnan(seq)):
                            return seq
                        return np.array([remap_dict.get(int(v), 0) for v in seq], dtype=np.int32)
                    user_features_df[col_name] = user_features_df[col_name].apply(remap_sequence)
                else:
                    original = user_features_df[col_name]
                    user_features_df[col_name] = original.map(remap_dict).fillna(0).astype(np.int32)

    logger.info(f"Remapped StageOutput ({stage_output.stage_name}): {len(candidates_df)} candidates")
    return stage_output


if __name__ == '__main__':
    main()
