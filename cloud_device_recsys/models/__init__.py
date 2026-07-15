# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Unified Models Module

This module provides a centralized model registry and build_model function
for all pipeline stages (retrieval, preranking, reranking).
"""

from .registry import MODEL_REGISTRY, build_model, get_available_models

# Re-export individual model classes for convenience
from .dual_tower_retrieval import DualTowerRetrieval
from .din_ranker import DINRanker
from .device_reranker import DeviceReranker

# Export loss utilities
from .losses import DiversityLossMixin, compute_diversity_loss, compute_item_similarity_matrix

__all__ = [
    # Registry functions
    'MODEL_REGISTRY',
    'build_model',
    'get_available_models',
    # Model classes
    'DualTowerRetrieval',
    'DINRanker',
    'DeviceReranker',
    # Loss utilities
    'DiversityLossMixin',
    'compute_diversity_loss',
    'compute_item_similarity_matrix',
]
