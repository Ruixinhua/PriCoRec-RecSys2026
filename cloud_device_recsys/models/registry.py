# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Model Registry

Provides a centralized registry for all pipeline models and a unified
build_model function that instantiates models based on configuration.
"""

import datetime
import logging
from typing import Dict, Any, Type, Optional

from fuxictr.features import FeatureMap
from .losses import wrap_model_with_diversity

logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_trainable_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _log_largest_parameters(model, top_k: int = 10) -> None:
    if top_k <= 0:
        return
    entries = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        entries.append((param.numel(), name, tuple(param.shape)))
    if not entries:
        return
    entries.sort(key=lambda item: item[0], reverse=True)
    preview = "; ".join(
        f"{name} shape={shape} params={count:,}"
        for count, name, shape in entries[:top_k]
    )
    logger.info("Largest trainable parameter tensors (top %d): %s", min(top_k, len(entries)), preview)


def _lazy_import_models():
    """Lazy import to avoid circular dependencies."""
    from .dual_tower_retrieval import DualTowerRetrieval
    from .din_ranker import DINRanker
    from .device_reranker import DeviceReranker

    models = {
        # Retrieval models
        "DualTowerRetrieval": DualTowerRetrieval,

        # Preranking models
        "DINRanker": DINRanker,

        # Reranking models
        "DeviceReranker": DeviceReranker,
    }

    try:
        from model_zoo import DNN, PNN, DCNv3, DeepFM, MaskNet, FiBiNET, DPSGD, DualRec, FedCAR, FedCIA
    except Exception as exc:
        logger.warning("model_zoo is unavailable: %s", exc)
    else:
        models.update({
            "DNN": DNN,
            "PNN": PNN,
            "DCNv3": DCNv3,
            "DeepFM": DeepFM,
            "MaskNet": MaskNet,
            "FiBiNET": FiBiNET,
            "DPSGD": DPSGD,
            "DualRec": DualRec,
            "FedCAR": FedCAR,
            "FedCIA": FedCIA,
        })

    return models


# Global registry - populated lazily
MODEL_REGISTRY: Dict[str, Type] = {}


def get_available_models() -> list:
    """Get list of available model names."""
    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_lazy_import_models())
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    feature_map: FeatureMap,
    model_params: Dict[str, Any],
    output_dir: Optional[str] = None,
    add_timestamp: bool = True,
    **kwargs
):
    """
    Build a model instance from the registry.

    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        feature_map: FuxiCTR FeatureMap instance
        model_params: Model-specific parameters
        output_dir: Output directory for model checkpoints
        add_timestamp: Whether to add timestamp to model_id
        **kwargs: Additional parameters passed to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not in registry
    """
    model_cls = None

    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_lazy_import_models())

    if model_name in MODEL_REGISTRY:
        model_cls = MODEL_REGISTRY[model_name]
        logger.info(f"Found model '{model_name}' in registry.")

    if model_cls is None:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Prepare default parameters
    default_params = {
        'verbose': 1,
        'metrics': ['AUC', 'logloss'],
        'gpu': -1,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
    }

    # Add model_root if output_dir provided
    if output_dir:
        default_params['model_root'] = output_dir

    # Merge: defaults < model_params < kwargs
    params = {**default_params, **model_params, **kwargs}
    model_params = params

    # Remove 'model' key if present (it's not a model parameter)
    params.pop('model', None)

    # Extract diversity loss params before model construction.
    use_diversity_loss = params.pop('use_diversity_loss', False)
    diversity_lambda = params.pop('diversity_lambda', 0.7)
    diversity_theta = params.pop('diversity_theta', 0.7)
    diversity_item_features = params.pop('diversity_item_features', None)

    # Add unique timestamp to model_id to prevent overwrites
    if add_timestamp:
        base_model_id = params.get("model_id", model_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        params["model_id"] = f"{base_model_id}_{timestamp}"

    logger.info(f"Building model: {model_name}")
    model = model_cls(feature_map, **params)

    # Apply vocabulary pruning if prune_info is available on the feature_map
    vocab_prune_info = getattr(feature_map, '_vocab_prune_info', None)
    if vocab_prune_info and vocab_prune_info.features:
        from .compact_embedding import apply_vocab_pruning_to_model
        logger.info(f"Applying vocabulary pruning to {model_name}...")
        apply_vocab_pruning_to_model(model, vocab_prune_info, feature_map)

    # Apply diversity loss wrapper if requested
    if use_diversity_loss:
        model = wrap_model_with_diversity(
            model,
            use_diversity_loss=True,
            diversity_lambda=diversity_lambda,
            diversity_theta=diversity_theta,
            diversity_item_features=diversity_item_features,
        )

    # Log parameter count if available
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    total_params = _count_trainable_parameters(model)
    _log_largest_parameters(
        model,
        top_k=_as_int(params.get("parameter_breakdown_top_k", 10), 10),
    )
    large_model_warning_parameters = _as_int(
        params.get("large_model_warning_parameters", 500_000_000),
        500_000_000,
    )
    if large_model_warning_parameters > 0 and total_params >= large_model_warning_parameters:
        logger.warning(
            "Model %s has %d trainable parameters, which is likely to be slow on a single GPU. "
            "Check feature_map vocab sizes and shared embeddings.",
            model_name,
            total_params,
        )
    max_model_parameters = _as_int(params.get("max_model_parameters", 0), 0)
    if max_model_parameters > 0 and total_params > max_model_parameters:
        raise ValueError(
            f"Model {model_name} has {total_params} trainable parameters, "
            f"exceeding max_model_parameters={max_model_parameters}."
        )

    # Set FP16 saving flag if configured
    if getattr(feature_map, '_save_fp16', False):
        model._save_fp16 = True
        logger.info("[FP16] Model weights will be saved in half-precision (FP16)")

    return model
