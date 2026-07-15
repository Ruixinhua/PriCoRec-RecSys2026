# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Positive Data Utilities for Pairwise Training.

This module provides utilities for managing positive-only training data
used in pairwise ranking with negative sampling.

Key concepts:
- Pointwise training: uses full data (positive + negative samples)
- Pairwise training: uses positive-only data + dynamic negative sampling

Training mode selection is based on `num_negatives` config parameter:
- num_negatives = 0 → Pointwise (full data)
- num_negatives > 0 → Pairwise (positive-only + negative sampling)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_positive_train_path(train_path: str, data_format: str = None) -> str:
    """
    Get the path for positive-only training data.

    Convention: If train data is at `path/train.parquet`,
    positive-only data is at `path/train_positive.parquet`.

    Args:
        train_path: Path to the original training data file

    Returns:
        Path to the positive-only training data file
    """
    path = Path(str(train_path))
    if (data_format or "").lower() in {"tfrecord", "tf_record"}:
        name = path.name
        if name.endswith(".tfrecord.gz"):
            return str(path.parent / f"{name[:-len('.tfrecord.gz')]}_positive.tfrecord.gz")
        if name.endswith(".tfrecord"):
            return str(path.parent / f"{name[:-len('.tfrecord')]}_positive.tfrecord")
        return str(path.parent / f"{name}_positive.tfrecord")

    stem = path.stem  # e.g., "train"
    suffix = path.suffix  # e.g., ".parquet"
    positive_name = f"{stem}_positive{suffix}"
    return str(path.parent / positive_name)


def ensure_positive_train_data(
    train_path: str,
    label_col: str = "label",
    logger: Optional[logging.Logger] = None,
    data_format: str = None,
    tfrecord_load_conf: Optional[dict] = None,
) -> str:
    """
    Ensure positive-only training data exists, creating it if necessary.

    This function is idempotent - it will not recreate the file if it exists.

    Args:
        train_path: Path to the original training data (containing both positive and negative)
        label_col: Name of the label column (1 = positive, 0 = negative)
        logger: Optional logger instance

    Returns:
        Path to the positive-only training data file

    Raises:
        FileNotFoundError: If the original train_path doesn't exist
        ValueError: If the label column is not found
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data_format = (data_format or Path(str(train_path)).suffix.lstrip(".")).lower()
    positive_path = get_positive_train_path(train_path, data_format=data_format)

    # Check if already exists
    if os.path.exists(positive_path):
        logger.info(f"Positive-only training data already exists: {positive_path}")
        return positive_path

    if data_format in {"tfrecord", "tf_record"}:
        return ensure_positive_tfrecord_data(
            train_path=train_path,
            positive_path=positive_path,
            label_col=label_col,
            tfrecord_load_conf=tfrecord_load_conf,
            logger=logger,
        )

    # Check source exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info(f"Creating positive-only training data from {train_path}...")

    # Load original data
    if train_path.endswith('.parquet'):
        df = pd.read_parquet(train_path)
    elif train_path.endswith('.csv'):
        df = pd.read_csv(train_path)
    else:
        raise ValueError(f"Unsupported file format: {train_path}")

    original_count = len(df)
    logger.info(f"Original dataset size: {original_count} samples")

    # Check label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")

    # Filter positive samples
    positive_df = df[df[label_col] == 1]
    positive_count = len(positive_df)

    logger.info(f"Filtered to {positive_count} positive samples ({positive_count/original_count*100:.1f}%)")

    # Save positive-only data
    if train_path.endswith('.parquet'):
        positive_df.to_parquet(positive_path, index=False)
    else:
        positive_df.to_csv(positive_path, index=False)

    logger.info(f"Saved positive-only training data to: {positive_path}")

    return positive_path


def ensure_positive_tfrecord_data(
    train_path: str,
    positive_path: str,
    label_col: str = "label",
    tfrecord_load_conf: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Create a positive-only TFRecord by copying raw serialized examples."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if os.path.exists(positive_path):
        logger.info(f"Positive-only training data already exists: {positive_path}")
        return positive_path

    try:
        from fuxictr.pytorch.dataloaders.tfrecord_dataloader import TFRecordDataLoader
        import tensorflow as tf
        from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only
    except ImportError as exc:
        raise ImportError("TensorFlow is required to create positive-only TFRecord data.") from exc

    configure_tensorflow_cpu_only(tf, logger)

    tfrecord_load_conf = dict(tfrecord_load_conf or {})
    resolver = object.__new__(TFRecordDataLoader)
    filenames = resolver._resolve_filenames(train_path)
    compression_type = resolver._resolve_compression_type(
        tfrecord_load_conf.get("compression_type", "AUTO"),
        filenames,
    )
    writer_options = (
        tf.io.TFRecordOptions(compression_type=compression_type)
        if compression_type
        else None
    )

    os.makedirs(os.path.dirname(positive_path) or ".", exist_ok=True)
    logger.info(f"Creating positive-only TFRecord data from {filenames}...")

    total_count = 0
    positive_count = 0
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type=compression_type,
        buffer_size=tfrecord_load_conf.get("buffer_size"),
        num_parallel_reads=(
            None
            if tfrecord_load_conf.get("num_parallel_reads") in (None, 0)
            else tfrecord_load_conf.get("num_parallel_reads")
        ),
    )
    with tf.io.TFRecordWriter(positive_path, options=writer_options) as writer:
        for raw_record in dataset:
            total_count += 1
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feature = example.features.feature.get(label_col)
            if feature is None:
                raise ValueError(f"Label column '{label_col}' not found in TFRecord example.")
            if feature.float_list.value:
                label_value = feature.float_list.value[0]
            elif feature.int64_list.value:
                label_value = feature.int64_list.value[0]
            else:
                label_value = 0
            if label_value > 0:
                writer.write(raw_record.numpy())
                positive_count += 1

    if total_count == 0:
        logger.warning("No TFRecord examples found while creating positive-only data.")
    else:
        logger.info(
            "Filtered TFRecord positives: %d/%d (%.1f%%)",
            positive_count,
            total_count,
            positive_count / total_count * 100,
        )
    logger.info(f"Saved positive-only TFRecord data to: {positive_path}")
    return positive_path


def get_train_path_for_mode(
    train_path: str,
    num_negatives: int,
    label_col: str = "label",
    logger: Optional[logging.Logger] = None,
    force_positive_only: bool = False,
    data_format: str = None,
    tfrecord_load_conf: Optional[dict] = None,
) -> str:
    """
    Get the appropriate training data path based on training mode.

    Args:
        train_path: Path to the original training data
        num_negatives: Number of negatives per positive (0 = pointwise mode)
        label_col: Name of the label column
        logger: Optional logger instance
        force_positive_only: Force positive-only data even when num_negatives <= 0.
            This is used by training modes that form negatives inside the batch.

    Returns:
        Path to the training data to use:
        - Original path if num_negatives <= 0 (pointwise mode)
        - Positive-only path if num_negatives > 0 (pairwise mode)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if force_positive_only:
        logger.info("Training mode: Forced positive-only")
        return ensure_positive_train_data(
            train_path, label_col, logger,
            data_format=data_format,
            tfrecord_load_conf=tfrecord_load_conf,
        )

    if num_negatives <= 0:
        # Pointwise mode: use full dataset
        logger.info(f"Training mode: Pointwise ({train_path})")
        return train_path
    else:
        # Pairwise mode: use positive-only dataset
        logger.info(f"Training mode: Pairwise with {num_negatives} negatives per positive")
        return ensure_positive_train_data(
            train_path, label_col, logger,
            data_format=data_format,
            tfrecord_load_conf=tfrecord_load_conf,
        )
