# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Stage Output Data Structures

This module defines the data structures for passing data between pipeline stages.
DataFrame-first architecture for high performance with large candidate sets.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime


class UnsafeStageOutputFormatError(ValueError):
    """Raised when code attempts to use a legacy executable stage artifact."""


class StageOutput:
    """
    Container for stage output including results and metrics.

    DataFrame-first architecture:
    - Primary storage: candidates_df, user_features_df (DataFrames)
    - candidate_sets property: Lazy-generated for backward compatibility
    - Save/Load: Direct Parquet I/O without object creation overhead
    """

    def __init__(
        self,
        stage_name: str,
        candidates_df: pd.DataFrame = None,
        user_features_df: pd.DataFrame = None,
        metrics: Dict[str, float] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.stage_name = stage_name
        self.metrics = metrics or {}
        self.metadata = metadata or {}

        # DataFrame-first storage
        self._candidates_df = candidates_df
        self._user_features_df = user_features_df

        # Request offsets cache (for efficient batch processing)
        self._request_offsets: np.ndarray = None

        # Timing info
        self.start_time: Optional[str] = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.duration_seconds: float = 0.0
    # ==================== DataFrame-first API ====================
    @property
    def candidates_df(self) -> pd.DataFrame:
        """
        Get candidates as DataFrame [request_id, item_id, score, label].
        This is the primary data access method for performance-critical code.
        """
        if self._candidates_df is not None:
            return self._candidates_df
        self._candidates_df = pd.DataFrame(columns=['request_id', 'item_id', 'score', 'label'])
        return self._candidates_df

    @property
    def user_features_df(self) -> pd.DataFrame:
        """Get user features as DataFrame [request_id, user_id, ...]."""
        if self._user_features_df is not None:
            return self._user_features_df
        self._user_features_df = pd.DataFrame(columns=['request_id', 'user_id'])

        return self._user_features_df

    def get_request_offsets(self) -> np.ndarray:
        """
        Get array of request offsets for efficient batch processing.
        Returns array where offsets[i] is the start index of request i in candidates_df.
        Last element is total length.
        """
        if self._request_offsets is not None:
            return self._request_offsets

        df = self.candidates_df
        if len(df) == 0:
            self._request_offsets = np.array([0])
            return self._request_offsets

        # FastPath: compute offsets using numpy
        request_ids = df['request_id'].values
        _, first_indices, counts = np.unique(request_ids, return_index=True, return_counts=True)

        # Sort by first occurrence order
        sort_order = np.argsort(first_indices)
        counts_sorted = counts[sort_order]

        # Build offsets: [0, len(req0), len(req0)+len(req1), ...]
        self._request_offsets = np.concatenate([[0], np.cumsum(counts_sorted)])
        return self._request_offsets

    @classmethod
    def from_dataframes(
        cls,
        stage_name: str,
        candidates_df: pd.DataFrame,
        user_features_df: pd.DataFrame = None,
        metrics: Dict[str, float] = None,
        metadata: Dict[str, Any] = None,
    ) -> 'StageOutput':
        """Create StageOutput directly from DataFrames (fast path)."""
        output = cls(
            stage_name=stage_name,
            candidates_df=candidates_df,
            user_features_df=user_features_df,
            metrics=metrics,
            metadata=metadata,
        )
        return output

    # ==================== Utility Methods ====================

    def mark_complete(self) -> None:
        """Mark the stage as complete and compute duration"""
        self.end_time = datetime.now().isoformat()
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.duration_seconds = (end - start).total_seconds()

    def get_total_candidates(self) -> int:
        """Get total number of candidates across all sets"""
        if self._candidates_df is not None:
            return len(self._candidates_df)
        return 0

    def get_num_requests(self) -> int:
        """Get number of requests"""
        if self._candidates_df is not None:
            return self._candidates_df['request_id'].nunique()
        return 0

    # ==================== Serialization ====================

    def save(self, filepath: str) -> None:
        """Reject legacy pickle output; use :meth:`save_parquet` instead."""
        del filepath
        raise UnsafeStageOutputFormatError(
            "Pickle StageOutput serialization is disabled because pickle artifacts can execute code. "
            "Use save_parquet() instead."
        )

    @staticmethod
    def load(filepath: str) -> 'StageOutput':
        """Reject legacy pickle input; use :meth:`load_parquet` instead."""
        del filepath
        raise UnsafeStageOutputFormatError(
            "Pickle StageOutput loading is disabled because pickle artifacts can execute code. "
            "Regenerate the stage output in Parquet format."
        )

    def save_parquet(self, dirpath: str) -> None:
        """
        Save stage output to Parquet format (fast path).

        Creates a directory with:
        - candidates.parquet: [request_id, item_id, score, label]
        - user_features.parquet: [request_id, user_id, ...]
        - metadata.json: Stage metadata
        """
        os.makedirs(dirpath, exist_ok=True)

        # Get DataFrames (may trigger conversion from legacy format)
        candidates_df = self.candidates_df
        user_features_df = self.user_features_df

        # Optimize dtypes
        if len(candidates_df) > 0:
            candidates_df = candidates_df.copy()
            candidates_df['score'] = candidates_df['score'].astype('float32')
            if 'label' in candidates_df.columns and candidates_df['label'].notna().any():
                candidates_df['label'] = candidates_df['label'].astype('Int8')

        # Save to Parquet
        candidates_df.to_parquet(
            os.path.join(dirpath, 'candidates.parquet'),
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        user_features_df.to_parquet(
            os.path.join(dirpath, 'user_features.parquet'),
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Save metadata
        metadata = {
            'stage_name': self.stage_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'num_requests': self.get_num_requests(),
            'num_candidates': len(candidates_df)
        }
        with open(os.path.join(dirpath, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_parquet(dirpath: str) -> 'StageOutput':
        """
        Load stage output from Parquet format (fast path).

        This does NOT create Python objects - data stays as DataFrames.
        """
        # Load metadata
        with open(os.path.join(dirpath, 'metadata.json'), 'r') as f:
            meta = json.load(f)

        # Load DataFrames directly (no object creation!)
        candidates_df = pd.read_parquet(os.path.join(dirpath, 'candidates.parquet'))
        user_features_df = pd.read_parquet(os.path.join(dirpath, 'user_features.parquet'))

        # Create StageOutput with DataFrames
        output = StageOutput(
            stage_name=meta['stage_name'],
            candidates_df=candidates_df,
            user_features_df=user_features_df,
            metrics=meta.get('metrics', {}),
            metadata=meta.get('metadata', {}),
        )
        output.start_time = meta.get('start_time')
        output.end_time = meta.get('end_time')
        output.duration_seconds = meta.get('duration_seconds', 0.0)

        return output
