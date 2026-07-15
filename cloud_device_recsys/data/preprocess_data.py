#!/usr/bin/env python
# =============================================================================
# Data Preprocessing Script (v2)
# =============================================================================

"""
Preprocesses raw data for cloud-device recommendation pipeline.
Uses new config format with compact feature definitions.

Usage:
    python preprocess_data.py --config ./config
    python preprocess_data.py --config ./config --force_rebuild
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import pandas as pd

from cloud_device_recsys.config.config_parser import ConfigParser


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DataPreprocessor')


def generate_impression_id(df: pl.LazyFrame, config: dict) -> pl.LazyFrame:
    """Generate impression_id if not present"""
    impression_col = config.get('impression_id_col', 'impression_id')
    schema = df.collect_schema()

    if impression_col not in schema.names():
        # Generate unique ID from row number
        df = df.with_row_index(impression_col)

    return df


def filter_positive_samples(df: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """Filter to positive samples only"""
    return df.filter(pl.col(label_col) == 1)


def preprocess_split(
    raw_path: str,
    output_path: str,
    dataset_config: dict,
    is_eval: bool = False,
    logger: logging.Logger = None,
    n_rows: int = None
) -> str:
    """
    Preprocess a single data split.

    Args:
        raw_path: Path to raw CSV
        output_path: Path for output parquet
        dataset_config: Dataset configuration
        is_eval: If True, filter to positive samples only
        logger: Logger instance

    Returns:
        Output path
    """
    if not os.path.exists(raw_path):
        if logger:
            logger.warning(f"Raw file not found: {raw_path}")
        return None

    if logger:
        logger.info(f"Processing: {raw_path}")

    # Load data
    if n_rows:
        # Use pandas for subset reading as polars scan/read caused hangs
        import pandas as pd
        pdf = pd.read_csv(raw_path, nrows=n_rows)
        df = pl.from_pandas(pdf).lazy()
    else:
        df = pl.scan_csv(raw_path)

    # Generate impression_id if needed
    if dataset_config.get('preprocessing', {}).get('generate_impression_id', True):
        df = generate_impression_id(df, dataset_config)

    # Filter for eval data
    label_col = dataset_config.get('label_col', {}).get('name', 'label')
    if is_eval and dataset_config.get('preprocessing', {}).get('positive_only_eval', True):
        df = filter_positive_samples(df, label_col)
        if logger:
            logger.info("Filtered to positive samples only")

    # Collect and save
    result = df.collect()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.write_parquet(output_path)

    if logger:
        logger.info(f"Saved {len(result)} rows to {output_path}")

    return output_path


def build_feature_vocab(
    df: pl.DataFrame,
    feature_cols: list,
    output_path: str,
    logger: logging.Logger = None
) -> dict:
    """Build feature vocabulary from data"""
    vocab = {}

    for feat in feature_cols:
        name = feat['name']
        feat_type = feat.get('type', 'categorical')

        if feat_type == 'categorical':
            if name in df.columns:
                unique_vals = df[name].unique().to_list()
                vocab[name] = {str(v): i for i, v in enumerate(unique_vals, start=1)}
        elif feat_type == 'sequence':
            # For sequences, vocab is built from splitted values
            if name in df.columns:
                splitter = feat.get('splitter', ',')
                all_vals = set()
                for val in df[name].drop_nulls().to_list():
                    if isinstance(val, str):
                        all_vals.update(val.split(splitter))
                vocab[name] = {str(v): i for i, v in enumerate(sorted(all_vals), start=1)}

    # Save vocab
    import json
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    if logger:
        logger.info(f"Saved vocabulary to {output_path}")

    return vocab


def apply_vocab_mapping(path: str, vocab: dict, feature_cols: list, logger: logging.Logger = None) -> None:
    """
    Apply vocabulary mapping to convert raw values to indices.
    Handles categorical (int) and sequence (list of ints) features.
    Overwrite the file at path.
    """
    import pandas as pd

    if not os.path.exists(path):
        return

    if logger:
        logger.info(f"Mapping values to indices for {path}")

    df = pd.read_parquet(path)

    # Create quick lookup for feature config
    feat_config = {f['name']: f for f in feature_cols}

    for col in df.columns:
        if col not in vocab:
            continue

        mapping = vocab[col]
        conf = feat_config.get(col, {})
        ctype = conf.get('type', 'categorical')

        if ctype == 'sequence':
            if logger:
                logger.info(f"  Processing sequence column: {col}")
            splitter = conf.get('splitter', '^') # Default from config usually '^' or ','
            max_len = conf.get('max_len', 50)
            padding_idx = 0

            # Function to process sequence strings
            def process_seq(x):
                if pd.isna(x) or x == "":
                    return [padding_idx] * max_len
                # Ensure string
                x = str(x)
                tokens = x.split(splitter)
                # Map and Pad
                ids = [mapping.get(t, 0) for t in tokens]
                if len(ids) > max_len:
                    ids = ids[:max_len]
                else:
                    ids += [padding_idx] * (max_len - len(ids))
                return ids

            # Apply to column
            df[col] = df[col].apply(process_seq)

        else:
            # Categorical
            if logger:
                logger.info(f"  Processing categorical column: {col}")
            # Use map which is faster than apply
            # Must handle string conversion for lookup
            # Fillna(0) for OOV
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype('int64')

    df.to_parquet(path)
    if logger:
        logger.info(f"Saved mapped data to {path}")


def update_vocab_from_df(vocab: dict, df: pd.DataFrame, feature_cols: list, logger=None):
    """Update vocabulary with new values from a dataframe"""
    if logger:
        logger.info("Updating vocabulary from additional data...")

    feat_config = {f['name']: f for f in feature_cols}

    for col in df.columns:
        if col not in feat_config:
            continue

        conf = feat_config[col]
        ctype = conf.get('type', 'categorical')

        # Initialize if missing
        if col not in vocab:
            vocab[col] = {}

        current_map = vocab[col]
        next_idx = len(current_map) + 1 # 1-based index

        new_vals = set()
        if ctype == 'sequence':
            splitter = conf.get('splitter', '^')
            # Extract unique tokens
            # This can be slow for large DF, but ItemPool is usually manageable
            def get_tokens(x):
                if pd.isna(x) or x == "": return []
                return str(x).split(splitter)

            # Using set/union for speed
            all_tokens = set()
            for row in df[col]:
                all_tokens.update(get_tokens(row))
            new_vals = all_tokens
        else:
            new_vals = set(df[col].astype(str).unique())

        # Add new vals
        for v in new_vals:
            if v not in current_map and v != "":
                current_map[v] = next_idx
                next_idx += 1

    if logger:
        logger.info("Vocabulary update complete.")


def enforce_shared_vocab(vocab: dict, dataset_config: dict, logger=None):
    """Enforce vocabulary sharing based on config"""
    feature_cols = dataset_config.get('feature_cols_expanded', [])

    # 1. Share embedding map handling (explicit map in config)
    # Check FG configuration for share_embedding_map
    for feat_group in dataset_config.get('feature_cols', []):
        share_map = feat_group.get('share_embedding_map', {})
        for target_feat, source_feat in share_map.items():
            if source_feat in vocab:
                if logger:
                    logger.info(f"Sharing vocab: {target_feat} <- {source_feat}")
                vocab[target_feat] = vocab[source_feat]

    # 2. Check individual feature 'share_embedding' attribute
    for feat in feature_cols:
        target = feat['name']
        source = feat.get('share_embedding')
        if source and source in vocab:
             if logger:
                logger.info(f"Sharing vocab (attr): {target} <- {source}")
             vocab[target] = vocab[source]

def main():
    parser = argparse.ArgumentParser(description='Preprocess data v2')
    parser.add_argument('--config', type=str, default='./cloud_device_recsys/config',
                       help='Config directory')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Dataset ID to process (from dataset_config.yaml)')
    parser.add_argument('--force_rebuild', action='store_true',
                       help='Force rebuild')
    parser.add_argument('--n_rows', type=int, default=None,
                       help='Limit rows for testing')

    args = parser.parse_args()

    # Load config
    config_parser = ConfigParser(args.config)
    config = config_parser.get_full_config(dataset_id=args.dataset_id)

    dataset_config = config['dataset']
    # pipeline_config = config['pipeline']

    # Setup
    output_dir = dataset_config['processed_paths']['train'].rsplit('/', 1)[0]
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Cloud-Device Recommendation - Data Preprocessing v2")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_config.get('dataset_id')}")

    raw_paths = dataset_config['raw_paths']
    processed_paths = dataset_config['processed_paths']

    # Check if already preprocessed
    if os.path.exists(processed_paths['train']) and not args.force_rebuild:
        logger.info("Preprocessed data exists. Use --force_rebuild to reprocess.")
        return

    # Process train
    preprocess_split(
        raw_path=raw_paths.get('train'),
        output_path=processed_paths['train'],
        dataset_config=dataset_config,
        is_eval=False,
        logger=logger,
        n_rows=args.n_rows
    )

    # Process valid (positive only)
    preprocess_split(
        raw_path=raw_paths.get('valid'),
        output_path=processed_paths['valid'],
        dataset_config=dataset_config,
        is_eval=True,
        logger=logger
    )

    # Process test (positive only)
    preprocess_split(
        raw_path=raw_paths.get('test'),
        output_path=processed_paths['test'],
        dataset_config=dataset_config,
        is_eval=True,
        logger=logger
    )

    # Build feature vocabulary from train
    logger.info("Building feature vocabulary...")
    train_df = pl.read_parquet(processed_paths['train'])

    feature_cols = dataset_config.get('feature_cols_expanded', [])
    vocab = build_feature_vocab(
        df=train_df,
        feature_cols=feature_cols,
        output_path=processed_paths['feature_vocab'],
        logger=logger
    )

    # Update vocab from Item Pool (critical for item features coverage)
    item_pool_raw = raw_paths.get('cand_item_list')
    if item_pool_raw and os.path.exists(item_pool_raw):
        logger.info(f"Updating vocab from Item Pool: {item_pool_raw}")
        # Use pandas for vocab update
        import pandas as pd
        ip_df = pd.read_csv(item_pool_raw)

        # Rename columns to match feature_map (item_X -> cand_item_X)
        rename_map = {}
        for col in ip_df.columns:
            if col.startswith("item_"):
                new_col = "cand_" + col
                rename_map[col] = new_col
        if rename_map:
             ip_df = ip_df.rename(columns=rename_map)

        update_vocab_from_df(vocab, ip_df, feature_cols, logger)

    # Enforce shared vocabularies (sequences use ID vocab)
    enforce_shared_vocab(vocab, dataset_config, logger)

    # Save updated vocab
    import json
    with open(processed_paths['feature_vocab'], 'w') as f:
        json.dump(vocab, f, indent=4)

    # Apply mapping to train, valid, test
    logger.info("Applying vocabulary mapping to datasets...")
    apply_vocab_mapping(processed_paths['train'], vocab, feature_cols, logger)
    apply_vocab_mapping(processed_paths['valid'], vocab, feature_cols, logger)
    apply_vocab_mapping(processed_paths['test'], vocab, feature_cols, logger)

    # Also map Item Pool!
    item_pool_path = os.path.join(output_dir, "cand_item_list.parquet")
    if os.path.exists(item_pool_path):
         apply_vocab_mapping(item_pool_path, vocab, feature_cols, logger)

    # Create FuxiCTR-compatible feature_map with vocab_size
    logger.info("Creating FuxiCTR-compatible feature_map...")
    import json

    # Build features list in FuxiCTR format: [{feature_name: {type, vocab_size, ...}}, ...]
    features_list = []
    total_features = 0
    input_length = 0
    num_fields = 0

    for feat in feature_cols:
        feat_name = feat['name']
        feat_type = feat.get('type', 'categorical')
        feat_group = feat.get('feature_group', 'FG1')

        # Determine source based on feature_group:
        # FG1 (non_personalized) = item features
        # FG2 (cloud_personalized) = user behavioral features
        # FG3 (device_only) = user privacy features
        if feat_group in ['FG1', 'non_personalized']:
            source = 'item'
        else:  # FG2 or FG3
            source = 'user'

        feat_spec = {
            'source': source,
            'type': feat_type,
            'feature_group': feat_group
        }

        if feat_type in ['categorical', 'sequence']:
            # Get vocab_size from vocabulary
            if feat_name in vocab:
                feat_spec['vocab_size'] = len(vocab[feat_name]) + 1  # +1 for padding
            else:
                feat_spec['vocab_size'] = 2  # Default minimum

            feat_spec['padding_idx'] = 0
            total_features += feat_spec['vocab_size']

            if feat_type == 'sequence':
                max_len = feat.get('max_len', 50)
                feat_spec['max_len'] = max_len
                feat_spec['feature_encoder'] = 'layers.MaskedAveragePooling()'
                input_length += max_len

                # Add share_embedding if specified
                if 'share_embedding' in feat:
                    feat_spec['share_embedding'] = feat['share_embedding']
            else:
                input_length += 1
        elif feat_type == 'numeric':
            input_length += 1

        num_fields += 1
        features_list.append({feat_name: feat_spec})

    # Build complete feature_map
    feature_map = {
        'dataset_id': dataset_config.get('dataset_id'),
        'num_fields': num_fields,
        'total_features': total_features,
        'input_length': input_length,
        'labels': [dataset_config.get('label_col', {}).get('name', 'label')],
        'features': features_list
    }

    with open(processed_paths['feature_map'], 'w') as f:
        json.dump(feature_map, f, indent=4)
    logger.info(f"Saved FuxiCTR feature_map to {processed_paths['feature_map']}")

    # Extract item pool
    logger.info("Extracting item pool...")
    item_pool_config = dataset_config.get('item_pool', {})
    item_pool_raw = raw_paths.get('cand_item_list')

    if item_pool_raw and os.path.exists(item_pool_raw):
        item_pool_df = pl.read_csv(item_pool_raw)

        # Rename columns to match feature_map (item_X -> cand_item_X)
        rename_map = {}
        for col in item_pool_df.columns:
            if col.startswith("item_"):
                new_col = "cand_" + col
                rename_map[col] = new_col

        if rename_map:
            logger.info(f"Renaming item pool columns: {rename_map}")
            item_pool_df = item_pool_df.rename(rename_map)

        if vocab:
            for col_name in item_pool_df.columns:
                if col_name in vocab:
                    logger.info(f"Mapping column {col_name} using vocab")
                    mapping = vocab[col_name]
                    # Map values: Cast to Str, lookup, default to 0 (OOV)
                    # Note: vocab keys are strings.
                    item_pool_df = item_pool_df.with_columns(
                        pl.col(col_name).cast(pl.Utf8).replace(mapping, default=0).cast(pl.Int64)
                    )

        # Add 'score' column if missing (required by feature_map)
        if "score" not in item_pool_df.columns:
            logger.info("Adding default 'score' column to item pool")
            item_pool_df = item_pool_df.with_columns(pl.lit(0.0).alias("score"))

        # Add 'label' column if missing (required by RankDataLoader)
        if "label" not in item_pool_df.columns:
            logger.info("Adding default 'label' column to item pool")
            item_pool_df = item_pool_df.with_columns(pl.lit(0).alias("label"))

        # Load feature map to check for other missing columns (e.g. user features)
        # item_pool needs to conform to feature_map schema even if user features are irrelevant
        feature_map_json = os.path.join(output_dir, "feature_map.json")
        if os.path.exists(feature_map_json):
            import json
            with open(feature_map_json, 'r') as f:
                fm = json.load(f)

            features = fm.get('features', [])
            # Convert dict format to list if needed
            if isinstance(features, dict):
                features = [{k: v} for k, v in features.items()]

            for feat_entry in features:
                # feat_entry is likely {name: spec}
                fname = list(feat_entry.keys())[0]
                fspec = feat_entry[fname]

                if fname not in item_pool_df.columns:
                    ftype = fspec.get('type', 'categorical')
                    logger.info(f"Adding dummy column '{fname}' (type: {ftype}) to item pool")

                    if ftype == 'sequence':
                        max_len = fspec.get('max_len', 50)
                        # Initialize as list of 0s (padding)
                        # Polars requires specific syntax for list literals or use pl.repeat?
                        # Simplest is likely using pl.lit with series logic or a python list
                        # pl.lit([0]*max_len) creates a List type column repeated
                        item_pool_df = item_pool_df.with_columns(
                            pl.lit([0] * max_len).alias(fname)
                        )
                    elif ftype == 'numeric':
                        item_pool_df = item_pool_df.with_columns(pl.lit(0.0).alias(fname))
                    else: # categorical
                        item_pool_df = item_pool_df.with_columns(pl.lit(0).alias(fname))

        item_pool_path = os.path.join(output_dir, "cand_item_list.parquet")
        item_pool_df.write_parquet(item_pool_path)
        logger.info(f"Saved {len(item_pool_df)} items to {item_pool_path}")

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
