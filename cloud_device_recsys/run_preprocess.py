#!/usr/bin/env python
# =============================================================================
# Standalone Data Preprocessing Script
# =============================================================================
"""
Preprocesses raw data using FuxiCTR's FeatureProcessor.
Reads configuration from dataset_config.yaml and processes train/valid/test splits.

Usage:
    python cloud_device_recsys/run_preprocess.py \
        --raw_data_root /path/to/raw_data \
        --dataset_id TaobaoOpenMCC \
        --config_dir ./cloud_device_recsys/config

    # With limited rows for testing:
    python cloud_device_recsys/run_preprocess.py \
        --raw_data_root /path/to/raw_data \
        --dataset_id TaobaoOpenMCC \
        --n_rows 1000 \
        --force_rebuild
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
import gc

from fuxictr.preprocess.feature_processor import FeatureProcessor
from fuxictr.preprocess.build_dataset import transform


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir,
        f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger('FuxiCTR-Preprocess')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f"Logging initialized. Saving to {log_file}")
    return logger


# =============================================================================
# Custom Preprocessing Functions
# =============================================================================

def generate_impression_id(ddf: pl.LazyFrame, impression_col: str = 'impression_id') -> pl.LazyFrame:
    """
    Generate impression_id if not present in the data.

    Args:
        ddf: Polars LazyFrame
        impression_col: Column name for impression ID

    Returns:
        LazyFrame with impression_id column
    """
    schema = ddf.collect_schema()

    if impression_col not in schema.names():
        logging.info(f"Generating {impression_col} column...")
        ddf = ddf.with_row_index(impression_col)

    return ddf


def preserve_auxiliary_columns(
    processed_ddf: pl.LazyFrame,
    source_ddf: pl.LazyFrame,
    columns: List[str],
) -> pl.LazyFrame:
    """Append non-model columns that ``FeatureProcessor.preprocess`` projects out.

    The feature processor deliberately selects only labels and active model
    features.  Request identifiers are needed by downstream listwise
    evaluation but must not become model inputs, so keep them in the persisted
    dataset as an auxiliary column after feature preprocessing.

    Both lazy frames originate from the same, already-filtered source and
    feature preprocessing is row-preserving.  Horizontal concatenation
    therefore preserves row alignment without using request IDs as a join key
    (which would be incorrect because an impression may contain many rows).
    """
    source_columns = set(source_ddf.collect_schema().names())
    missing = [column for column in columns if column not in source_columns]
    if missing:
        raise ValueError(
            "Required auxiliary columns are missing after split preprocessing: "
            f"{missing}. Configure a valid impression_id_col or enable "
            "generate_impression_id."
        )

    processed_columns = set(processed_ddf.collect_schema().names())
    columns_to_append = [column for column in columns if column not in processed_columns]
    if not columns_to_append:
        return processed_ddf

    return pl.concat(
        [processed_ddf, source_ddf.select(columns_to_append)],
        how="horizontal",
    )


def filter_positive_samples(ddf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Filter to positive samples only (label == 1).

    Args:
        ddf: Polars LazyFrame
        label_col: Name of label column

    Returns:
        Filtered LazyFrame
    """
    return ddf.filter(pl.col(label_col) == 1)


def filter_rare_items(
    ddf: pl.LazyFrame,
    item_id_col: str,
    min_count: int
) -> pl.LazyFrame:
    """
    Drop samples whose item_id appears fewer than min_count times.

    Args:
        ddf: Polars LazyFrame
        item_id_col: Column name for item ID
        min_count: Minimum count threshold

    Returns:
        Filtered LazyFrame
    """
    if min_count <= 1:
        return ddf

    counts = ddf.group_by(item_id_col).agg(pl.len().alias("item_cnt"))
    return (
        ddf.join(
            counts.filter(pl.col("item_cnt") >= min_count),
            on=item_id_col,
            how="inner"
        )
        .drop("item_cnt")
    )


def build_item_keep_list(
    ddf: pl.LazyFrame,
    item_id_col: str,
    min_count: int,
    count_col: str = "item_cnt",
) -> pl.LazyFrame:
    """
    Build an item whitelist for rows whose item frequency is at least min_count.

    Args:
        ddf: Source LazyFrame
        item_id_col: Item ID column
        min_count: Minimum item frequency threshold
        count_col: Temporary count column name

    Returns:
        LazyFrame with one column: item_id_col
    """
    return (
        ddf.group_by(item_id_col)
        .agg(pl.len().alias(count_col))
        .filter(pl.col(count_col) >= min_count)
        .select(item_id_col)
        .unique()
    )


def build_train_head_items(
    ddf: pl.LazyFrame,
    item_id_col: str,
    label_col: str,
    min_positive_count: int
) -> pl.LazyFrame:
    """
    Build the item whitelist for train filtering from positive train samples only.

    Args:
        ddf: Train split LazyFrame
        item_id_col: Item ID column
        label_col: Label column
        min_positive_count: Keep items with at least this many positive train rows

    Returns:
        LazyFrame with one column: item_id_col
    """
    return build_item_keep_list(
        ddf.filter(pl.col(label_col) > 0),
        item_id_col=item_id_col,
        min_count=min_positive_count,
        count_col="positive_item_cnt",
    )


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_dataset_config(config_dir: str, dataset_id: str) -> Dict[str, Any]:
    """
    Load dataset configuration from dataset_config.yaml.

    Args:
        config_dir: Directory containing config files
        dataset_id: Dataset ID to load

    Returns:
        Dataset configuration dict
    """
    config_path = os.path.join(config_dir, 'dataset_config.yaml')

    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)

    if dataset_id not in all_configs:
        raise ValueError(f"Dataset '{dataset_id}' not found in {config_path}. "
                        f"Available: {list(all_configs.keys())}")

    config = all_configs[dataset_id]
    config['dataset_id'] = dataset_id
    return config


def expand_feature_cols(feature_cols: List[Dict]) -> List[Dict]:
    """
    Expand compact feature definitions (with name lists) into individual features.

    Example:
        Input:  {name: [a, b, c], type: categorical, dtype: int, feature_group: FG1}
        Output: [{name: a, type: categorical, dtype: int, feature_group: FG1},
                 {name: b, type: categorical, dtype: int, feature_group: FG1},
                 {name: c, type: categorical, dtype: int, feature_group: FG1}]

    Note: Sequence columns are always read as strings first (for splitting),
    regardless of the configured dtype.
    """
    expanded = []

    for col in feature_cols:
        name_or_list = col.get('name')

        if isinstance(name_or_list, list):
            # Handle share_embedding_map for sequences
            share_embedding_map = col.get('share_embedding_map', {})

            for name in name_or_list:
                new_col = col.copy()
                new_col['name'] = name

                # Remove list-based fields
                if 'share_embedding_map' in new_col:
                    del new_col['share_embedding_map']

                # Apply share_embedding from map if exists
                if name in share_embedding_map:
                    new_col['share_embedding'] = share_embedding_map[name]

                # Set active based on feature_group
                feature_group = new_col.get('feature_group', '')
                new_col['active'] = feature_group != 'drop'

                # Sequence columns must be read as strings for splitting
                if new_col.get('type') == 'sequence':
                    new_col['dtype'] = 'str'

                expanded.append(new_col)
        else:
            new_col = col.copy()
            new_col['active'] = col.get('feature_group', '') != 'drop'

            # Sequence columns must be read as strings for splitting
            if new_col.get('type') == 'sequence':
                new_col['dtype'] = 'str'

            expanded.append(new_col)

    return expanded


def build_label_col(config: Dict) -> List[Dict]:
    """
    Build label column specification from config.

    Args:
        config: Dataset configuration

    Returns:
        List with label column specification
    """
    label_config = config.get('label_col', {})
    return [{
        'name': label_config.get('name', 'label'),
        'dtype': label_config.get('dtype', 'float')
    }]


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

class DataPreprocessor:
    """
    Orchestrates data preprocessing using FuxiCTR's FeatureProcessor.
    """

    def __init__(
        self,
        raw_data_root: str,
        dataset_id: str,
        config_dir: str = './cloud_device_recsys/config',
        output_dir: Optional[str] = None,
        n_rows: Optional[int] = None
    ):
        self.raw_data_root = raw_data_root
        self.dataset_id = dataset_id
        self.config_dir = config_dir
        self.n_rows = n_rows

        # Load configuration
        self.config = load_dataset_config(config_dir, dataset_id)

        # Override paths if provided
        if output_dir:
            self.config['processed_data_root'] = output_dir

        # Use output dir from config
        self.output_dir = self.config.get('processed_data_root',
                                          f'./data/{dataset_id}')

        # Setup logging
        self.logger = setup_logging(self.output_dir)

        # Expand feature columns
        self.feature_cols = expand_feature_cols(self.config.get('feature_cols', []))
        self.label_cols = build_label_col(self.config)
        self.impression_id_col = self.config.get('impression_id_col', 'impression_id')
        if any(feature['name'] == self.impression_id_col for feature in self.feature_cols):
            raise ValueError(
                f"impression_id_col '{self.impression_id_col}' must be an auxiliary "
                "column, not a model feature."
            )

        # Preprocessing options
        self.preprocess_opts = self.config.get('preprocessing', {})
        self.train_filter_opts = self.preprocess_opts.get('train_item_freq_filter', {})
        self.eval_filter_opts = self.preprocess_opts.get('eval_item_freq_filter', {})
        self._train_head_items_ddf = None
        self._train_filter_initialized = False
        self._eval_keep_items_ddf = None
        self._eval_filter_initialized = False

        self.logger.info("=" * 60)
        self.logger.info("FuxiCTR Data Preprocessing")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Raw data root: {raw_data_root}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Feature columns: {len(self.feature_cols)}")
        self.logger.info(f"N rows limit: {n_rows if n_rows else 'None'}")

    def _get_train_filter_min_positive_count(self) -> int:
        if not self.train_filter_opts.get('enabled', False):
            return 1
        return int(self.train_filter_opts.get('min_positive_count', 1))

    def _get_eval_filter_min_positive_count(self) -> int:
        if not self.eval_filter_opts.get('enabled', False):
            return 1
        return int(self.eval_filter_opts.get('min_positive_count', 1))

    def _maybe_initialize_train_filter(self, train_ddf: pl.LazyFrame) -> None:
        min_positive_count = self._get_train_filter_min_positive_count()
        if min_positive_count <= 1 or self._train_filter_initialized:
            return

        item_id_col = self.config.get('item_id_col', 'item_id')
        label_col = self.label_cols[0]['name']
        self._train_head_items_ddf = build_train_head_items(
            train_ddf,
            item_id_col=item_id_col,
            label_col=label_col,
            min_positive_count=min_positive_count
        )
        kept_items = self._train_head_items_ddf.select(pl.len()).collect().item()
        self.logger.info(
            f"[train] Initialized positive item-frequency filter: "
            f"keep items with >= {min_positive_count} positive rows "
            f"-> {kept_items} unique items"
        )
        self._train_filter_initialized = True

    def _maybe_initialize_eval_filter(self, train_ddf: Optional[pl.LazyFrame]) -> None:
        """Build the optional evaluation-item filter from train labels only.

        Using valid/test positive labels to decide which evaluated rows survive
        makes holdout membership label-dependent.  The filter remains
        available, but its whitelist is now derived solely from training data.
        """
        min_positive_count = self._get_eval_filter_min_positive_count()
        if min_positive_count <= 1 or self._eval_filter_initialized:
            return

        if train_ddf is None:
            raise ValueError(
                "eval_item_freq_filter requires a training split so its whitelist "
                "can be built without using holdout labels."
            )

        label_col = self.label_cols[0]['name']
        item_id_col = self.config.get('item_id_col', 'item_id')
        self._eval_keep_items_ddf = build_item_keep_list(
            train_ddf.filter(pl.col(label_col) > 0),
            item_id_col=item_id_col,
            min_count=min_positive_count,
            count_col="train_positive_item_cnt",
        )
        kept_items = self._eval_keep_items_ddf.select(pl.len()).collect().item()
        self.logger.info(
            f"[eval] Initialized train-derived positive item-frequency filter: "
            f"keep items with >= {min_positive_count} positive train rows "
            f"-> {kept_items} unique items"
        )
        self._eval_filter_initialized = True

    def get_raw_path(self, split: str) -> str:
        """Get path to raw data file for a split."""
        data_format = self.config.get('raw_data_format', 'csv')
        return os.path.join(self.raw_data_root, f"{split}.{data_format}")

    def preprocess_split(
        self,
        ddf: pl.LazyFrame,
        split: str,
    ) -> pl.LazyFrame:
        """
        Apply custom preprocessing to a data split.

        Args:
            ddf: Polars LazyFrame
            split: Split name (train, valid, test)

        Returns:
            Preprocessed LazyFrame
        """
        # Generate impression_id if configured
        if self.preprocess_opts.get('generate_impression_id', True):
            impression_col = self.config.get('impression_id_col', 'impression_id')
            ddf = generate_impression_id(ddf, impression_col)

        if split == 'train':
            min_positive_count = self._get_train_filter_min_positive_count()
            if min_positive_count > 1:
                if self._train_head_items_ddf is None:
                    self._maybe_initialize_train_filter(ddf)
                item_id_col = self.config.get('item_id_col', 'item_id')
                before_count = ddf.select(pl.len()).collect().item()
                ddf = ddf.join(self._train_head_items_ddf, on=item_id_col, how='inner')
                after_count = ddf.select(pl.len()).collect().item()
                self.logger.info(
                    f"[train] Filtered by positive item freq >= {min_positive_count}: "
                    f"{before_count} -> {after_count}"
                )

        if split in ['valid', 'test']:
            # Filter positive samples for eval splits
            if self.preprocess_opts.get('positive_only_eval', True):
                before_count = ddf.select(pl.len()).collect().item()
                label_col = self.label_cols[0]['name']
                ddf = filter_positive_samples(ddf, label_col)
                after_count = ddf.select(pl.len()).collect().item()
                self.logger.info(f"[{split}] Filtered positive samples: {before_count} -> {after_count}")

            min_eval_positive_count = self._get_eval_filter_min_positive_count()
            if min_eval_positive_count > 1:
                if self._eval_keep_items_ddf is None:
                    raise RuntimeError(
                        "Eval item-frequency filter requested but not initialized. "
                        "Call _maybe_initialize_eval_filter() before preprocess_split()."
                    )
                item_id_col = self.config.get('item_id_col', 'item_id')
                before_count = ddf.select(pl.len()).collect().item()
                ddf = ddf.join(self._eval_keep_items_ddf, on=item_id_col, how='inner')
                after_count = ddf.select(pl.len()).collect().item()
                self.logger.info(
                    f"[{split}] Filtered by train-derived positive item freq >= "
                    f"{min_eval_positive_count}: {before_count} -> {after_count}"
                )

            min_item_count = self.preprocess_opts.get('min_item_count_eval', 1)
            if min_item_count > 1:
                item_id_col = self.config.get('item_id_col', 'item_id')
                before_count = ddf.select(pl.len()).collect().item()
                ddf = filter_rare_items(ddf, item_id_col, min_item_count)
                after_count = ddf.select(pl.len()).collect().item()
                self.logger.info(
                    f"[{split}] Filtered rare items (<{min_item_count}): "
                    f"{before_count} -> {after_count}"
                )

        return ddf

    def run(self, force_rebuild: bool = False) -> None:
        """
        Run the preprocessing pipeline.

        Args:
            force_rebuild: Force rebuild even if output exists
        """
        # Check if already processed
        feature_map_path = os.path.join(self.output_dir, 'feature_map.json')

        if os.path.exists(feature_map_path) and not force_rebuild:
            self.logger.info(f"Feature map already exists: {feature_map_path}")
            self.logger.info("Use --force_rebuild to reprocess.")
            return

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize FeatureProcessor
        self.logger.info("Initializing FeatureProcessor...")
        feature_processor = FeatureProcessor(
            feature_cols=self.feature_cols,
            label_col=self.label_cols,
            dataset_id=self.dataset_id,
            data_root=os.path.dirname(self.output_dir),
            min_categr_count=self.preprocess_opts.get('min_categr_count', 1)
        )

        # Get raw data format
        data_format = self.config.get('raw_data_format', 'csv')

        # =====================================================================
        # Read all data splits
        # =====================================================================
        raw_split_ddfs = {}
        split_ddfs = {}  # Store preprocessed LazyFrames for each split

        for split_name in ['train', 'valid', 'test']:
            split_path = self.get_raw_path(split_name)
            if not os.path.exists(split_path):
                continue
            self.logger.info(f"Reading {split_name} data: {split_path}")
            raw_split_ddfs[split_name] = feature_processor.read_data(
                split_path,
                data_format=data_format,
                n_rows=self.n_rows
            )

        train_ddf = raw_split_ddfs.get('train')
        if train_ddf is None:
            raise ValueError("A training split is required to fit the feature processor.")
        self._maybe_initialize_train_filter(train_ddf)
        self._maybe_initialize_eval_filter(train_ddf)

        for split_name in ['train', 'valid', 'test']:
            split_ddf = raw_split_ddfs.get(split_name)
            if split_ddf is None:
                continue
            split_ddf = self.preprocess_split(split_ddf, split_name)
            split_ddfs[split_name] = split_ddf

        # =====================================================================
        # Build vocabulary from TRAIN data only
        # =====================================================================
        self.logger.info("Building vocabulary from the training split only...")
        train_for_fit = feature_processor.preprocess(split_ddfs['train'])

        # Fit only on train so valid/test categories, sequence values, and
        # numeric statistics cannot influence the feature map.
        self.logger.info("Fitting feature processor on TRAIN data only...")
        feature_processor.fit(
            train_for_fit,
            min_categr_count=self.preprocess_opts.get('min_categr_count', 1),
            rebuild_dataset=True
        )

        # Clear the fit frame from memory before transforming individual splits.
        del train_for_fit
        gc.collect()

        # =====================================================================
        # Transform and save each split separately
        # =====================================================================
        for split_name, split_ddf in split_ddfs.items():
            self.logger.info(f"Transforming and saving {split_name} data...")

            # Re-read the data (since we consumed it during concatenation)
            split_path = self.get_raw_path(split_name)
            split_ddf = feature_processor.read_data(
                split_path,
                data_format=data_format,
                n_rows=self.n_rows
            )

            # Apply custom preprocessing
            split_ddf = self.preprocess_split(split_ddf, split_name)

            # Apply FuxiCTR preprocessing, then reattach the non-model request
            # ID which FeatureProcessor.preprocess intentionally projects out.
            source_ddf = split_ddf
            split_ddf = feature_processor.preprocess(source_ddf)
            split_ddf = preserve_auxiliary_columns(
                split_ddf,
                source_ddf,
                [self.impression_id_col],
            )

            # Transform and save
            transform(
                feature_processor,
                split_ddf,
                split_name,
                block_size=0,  # Auto-detect based on memory
                saved_format='parquet'
            )

            del split_ddf
            gc.collect()

        self.logger.info("=" * 60)
        self.logger.info("Preprocessing complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("Vocabulary built from: ['train']")
        self.logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw data using FuxiCTR FeatureProcessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process TaobaoOpenMCC dataset
    python run_preprocess.py \\
        --raw_data_root /path/to/TaobaoOpenMCC/with_item_id \\
        --dataset_id TaobaoOpenMCC

    # Process with limited rows for testing
    python run_preprocess.py \\
        --raw_data_root /path/to/raw_data \\
        --dataset_id TaobaoOpenMCC \\
        --n_rows 1000 \\
        --force_rebuild
        """
    )

    parser.add_argument(
        '--raw_data_root',
        type=str,
        required=True,
        help='Path to raw data directory containing train/valid/test files'
    )

    parser.add_argument(
        '--dataset_id',
        type=str,
        required=True,
        help='Dataset ID from dataset_config.yaml (e.g., TaobaoOpenMCC, TaobaoAd)'
    )

    parser.add_argument(
        '--config_dir',
        type=str,
        default='./cloud_device_recsys/config',
        help='Path to config directory (default: ./cloud_device_recsys/config)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory (default: from config)'
    )

    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='Force rebuild even if output already exists'
    )

    parser.add_argument(
        '--n_rows',
        type=int,
        default=None,
        help='Limit number of rows to process (for testing)'
    )

    args = parser.parse_args()

    # Run preprocessing
    preprocessor = DataPreprocessor(
        raw_data_root=args.raw_data_root,
        dataset_id=args.dataset_id,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        n_rows=args.n_rows
    )

    preprocessor.run(force_rebuild=args.force_rebuild)


if __name__ == '__main__':
    main()
