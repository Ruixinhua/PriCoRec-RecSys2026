# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Pre-ranking Stage Implementation

This module wraps the preranking model as a pipeline stage.
"""

import os
import csv
import json
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss, compute_diversity_loss_per_user
from ..data.negative_sampler import NegativeSampler
from ..utils import filter_feature_map

from fuxictr.features import FeatureMap


class PrerankingStage(BaseStage):
    """
    Pre-ranking stage for efficient candidate scoring.

    Takes candidates from retrieval and produces a refined set.
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    """

    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 allowed_feature_groups: List[FeatureGroup] = None,
                 output_dir: str = "./outputs/preranking",
                 top_k: int = 100,
                 **kwargs):
        """
        Initialize pre-ranking stage.

        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            model_params: Parameters for preranking model
            allowed_feature_groups: Allowed feature groups
            output_dir: Output directory
            top_k: Number of candidates to pass to next stage
            use_diversity: Whether to apply diversity in selection
        """
        if allowed_feature_groups is None:
            allowed_feature_groups = [FeatureGroup.FG1, FeatureGroup.FG2]

        super().__init__(
            stage_name="preranking",
            stage_type=StageType.PRERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )

        # Filter feature_map to only include allowed features (FG1, FG2)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self._log_active_feature_summary()
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.use_logit = model_params.get('use_logit', True)
        self.top_k = top_k
        self.use_diversity_loss = self._as_bool(model_params.get('use_diversity_loss'), False)
        if self.use_diversity_loss:
            self.logger.info(
                f"Computing diversity loss theta: {model_params.get('diversity_theta', 0.7)} "
                f"lambda: {model_params.get('diversity_lambda', 0.7)} "
                f"kernel: {model_params.get('diversity_kernel', 'gram')} "
                f"gamma: {model_params.get('diversity_gamma', 1.0)}")
        # Delayed diversity loss parameters
        self.diversity_start_epoch = model_params.get('diversity_start_epoch', -1)
        self.diversity_epochs = model_params.get('diversity_epochs', 5)
        self.diversity_warmup_epochs = model_params.get('diversity_warmup_epochs', 0)
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.monitor = model_params.get('monitor', 'Recall@100')
        self.patience = model_params.get('patience', 2)
        self.model: Optional[Any] = None
        self.best_weights_path = None
        # Negative sampling parameters
        self.num_negatives = self._as_non_negative_int(model_params.get('num_negatives', 0), 0)
        self.diversity_num_negatives = self._as_non_negative_int(
            model_params.get('diversity_num_negatives', self.num_negatives),
            self.num_negatives,
        )
        if self.use_diversity_loss and self.num_negatives <= 0 and self.diversity_num_negatives > 0:
            self.logger.info(
                "Diversity loss requested with num_negatives=0; using diversity_num_negatives=%d "
                "for explicit negative sampling.",
                self.diversity_num_negatives,
            )
            self.num_negatives = self.diversity_num_negatives
        self.evaluate_pool_diversity = model_params.get('evaluate_pool_diversity', False)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax'
        self.margin = model_params.get('margin', 1.0)
        self.training_mode = str(model_params.get('training_mode', 'auto') or 'auto').lower()
        self.use_hybrid_loss = (
            self.training_mode == 'hybrid'
            or self._as_bool(model_params.get('hybrid_loss'), False)
        )
        self.hybrid_pointwise_loss_weight = self._as_non_negative_float(
            model_params.get('hybrid_pointwise_loss_weight', 1.0),
            1.0,
        )
        self.hybrid_pairwise_loss_weight = self._as_non_negative_float(
            model_params.get('hybrid_pairwise_loss_weight', 0.1 if self.use_hybrid_loss else 1.0),
            0.1 if self.use_hybrid_loss else 1.0,
        )
        self.use_in_batch_negatives = self._as_bool(model_params.get('use_in_batch_negatives'), False)
        self.in_batch_negative_chunk_size = max(1, int(model_params.get('in_batch_negative_chunk_size', 128)))
        self.in_batch_negative_sample_size = int(model_params.get('in_batch_negative_sample_size', 255))
        self.negative_sampling_strategy = str(model_params.get('negative_sampling_strategy', 'uniform') or 'uniform').lower()
        negative_sampling_popularity_alpha = model_params.get('negative_sampling_popularity_alpha', 0.75)
        if negative_sampling_popularity_alpha in (None, ""):
            negative_sampling_popularity_alpha = 0.75
        self.negative_sampling_popularity_alpha = float(negative_sampling_popularity_alpha)
        self.progress_interval = max(0, int(model_params.get('progress_interval', 0) or 0))
        self.eval_interval_batches = max(0, int(model_params.get('eval_interval_batches', 0) or 0))
        self.eval_interval_epochs = self._as_optional_positive_float(model_params.get('eval_interval_epochs'))
        self.class_loss_weight = self._parse_class_loss_weight(model_params)
        self.diagnostic_batches = self._as_non_negative_int(model_params.get('diagnostic_batches', 1), 1)
        self.diagnostic_epochs = self._as_non_negative_int(model_params.get('diagnostic_epochs', 1), 1)
        self.dense_diagnostic_top_k = self._as_non_negative_int(model_params.get('dense_diagnostic_top_k', 10), 10)
        self.negative_sampler: Optional[NegativeSampler] = None
        # Inference batch size for process_and_rank_candidates
        self.inference_batch_size = model_params.get('inference_batch_size', 50000)
        self.popularity_blend_weight = float(model_params.get('popularity_blend_weight', 0.0) or 0.0)
        self.popularity_blend_transform = model_params.get('popularity_blend_transform', 'log1p')
        self.popularity_blend_normalize = model_params.get('popularity_blend_normalize', 'zscore')
        self.popularity_model_bias_weight = float(model_params.get('popularity_model_bias_weight', 0.0) or 0.0)
        self.popularity_prior_counts = None
        self.popularity_prior_stats = None
        self.popularity_item_id_col = None
        # Evaluation item features are used only for valid/test listwise
        # processing. Explicit training negatives use the separate train-only
        # pool below.
        self.item_features_df = None
        self.train_negative_item_features_df = None

    def set_popularity_prior(
            self,
            counts: Dict[Any, int],
            item_id_col: str,
            transform: str = "log1p",
            normalize: str = "zscore") -> None:
        self.popularity_prior_counts = dict(counts or {})
        self.popularity_item_id_col = item_id_col
        self.popularity_blend_transform = transform or self.popularity_blend_transform
        self.popularity_blend_normalize = normalize or self.popularity_blend_normalize
        values = np.asarray(list(self.popularity_prior_counts.values()) or [0.0], dtype=np.float64)
        values = self._transform_popularity(values, self.popularity_blend_transform)
        self.popularity_prior_stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "maxabs": float(np.max(np.abs(values))) if values.size else 0.0,
            "min": float(np.min(values)) if values.size else 0.0,
            "max": float(np.max(values)) if values.size else 0.0,
        }

    @staticmethod
    def _safe_logit_np(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
        return np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid_np(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _transform_popularity(values: np.ndarray, transform: str) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        transform = str(transform or "log1p").lower()
        if transform in {"log", "log1p"}:
            return np.log1p(np.maximum(scores, 0.0))
        if transform == "sqrt":
            return np.sqrt(np.maximum(scores, 0.0))
        return scores

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _normalize_popularity(self, values: np.ndarray) -> np.ndarray:
        scores = self._transform_popularity(values, self.popularity_blend_transform)
        stats = self.popularity_prior_stats or {}
        normalize = str(self.popularity_blend_normalize or "zscore").lower()
        if normalize in {"zscore", "standard", "standardize"}:
            std = float(stats.get("std") or 0.0)
            if std > 1e-12:
                return (scores - float(stats.get("mean", 0.0))) / std
        elif normalize in {"max", "maxabs"}:
            denom = float(stats.get("maxabs") or 0.0)
            if denom > 1e-12:
                return scores / denom
        elif normalize == "minmax":
            lo = float(stats.get("min", 0.0))
            hi = float(stats.get("max", 0.0))
            if hi - lo > 1e-12:
                return (scores - lo) / (hi - lo)
        return scores

    def _batch_popularity_prior(
            self,
            batch_data: Dict[str, Any],
            require_blend_weight: bool = True) -> Optional[np.ndarray]:
        if require_blend_weight and self.popularity_blend_weight == 0.0:
            return None
        if not self.popularity_prior_counts:
            return None
        item_id_col = self.popularity_item_id_col or getattr(self.feature_map, "dataset_config", {}).get("item_id_col", "cand_item_id")
        if item_id_col not in batch_data:
            return None
        raw_values = batch_data[item_id_col].detach().cpu().numpy().reshape(-1)
        counts = np.asarray([
            float(self.popularity_prior_counts.get(value.item() if hasattr(value, "item") else value, 0.0))
            for value in raw_values
        ], dtype=np.float64)
        return self._normalize_popularity(counts)

    def _apply_popularity_model_bias(
            self,
            return_dict: Dict[str, Any],
            batch_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.popularity_model_bias_weight == 0.0:
            return return_dict
        popularity_prior = self._batch_popularity_prior(batch_data, require_blend_weight=False)
        if popularity_prior is None or "y_pred" not in return_dict:
            return return_dict

        y_pred = return_dict["y_pred"]
        prior_tensor = torch.as_tensor(
            popularity_prior,
            device=y_pred.device,
            dtype=y_pred.dtype,
        ).reshape(y_pred.shape)
        if "logit" in return_dict:
            base_logit = return_dict["logit"]
        else:
            base_logit = torch.logit(torch.clamp(y_pred, 1e-7, 1.0 - 1e-7))
        biased_logit = base_logit + self.popularity_model_bias_weight * prior_tensor
        biased = dict(return_dict)
        biased["logit"] = biased_logit
        biased["y_pred"] = torch.sigmoid(torch.clamp(biased_logit, -50.0, 50.0))
        return biased

    @staticmethod
    def _as_optional_positive_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _as_non_negative_int(value: Any, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(default))

    @staticmethod
    def _as_non_negative_float(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return max(0.0, float(default))

    @staticmethod
    def _parse_class_loss_weight(model_params: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if not PrerankingStage._as_bool(model_params.get("use_class_loss_weight"), False):
            return None
        loss_weight = model_params.get("loss_weight")
        if loss_weight in (None, ""):
            return None
        if isinstance(loss_weight, str):
            raw_values = [part.strip() for part in loss_weight.replace(";", ",").split(",") if part.strip()]
        else:
            raw_values = list(loss_weight) if isinstance(loss_weight, (list, tuple)) else [loss_weight]
        if len(raw_values) != 2:
            return None
        try:
            negative_weight = float(raw_values[0])
            positive_weight = float(raw_values[1])
        except (TypeError, ValueError):
            return None
        return negative_weight, positive_weight

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _tensor_stats(tensor: Any) -> Dict[str, Any]:
        if tensor is None or not hasattr(tensor, "detach"):
            return {}
        with torch.no_grad():
            values = tensor.detach().float().reshape(-1)
            count = int(values.numel())
            if count == 0:
                return {"count": 0}
            finite_mask = torch.isfinite(values)
            finite = values[finite_mask]
            finite_count = int(finite.numel())
            stats = {
                "count": count,
                "finite_count": finite_count,
                "non_finite_count": count - finite_count,
            }
            if finite_count > 0:
                stats.update({
                    "min": float(finite.min().item()),
                    "max": float(finite.max().item()),
                    "mean": float(finite.mean().item()),
                    "std": float(finite.std(unbiased=False).item()) if finite_count > 1 else 0.0,
                })
            return stats

    @classmethod
    def _label_tensor_stats(cls, tensor: Any) -> Dict[str, Any]:
        stats = cls._tensor_stats(tensor)
        if not stats or not hasattr(tensor, "detach"):
            return stats
        with torch.no_grad():
            values = tensor.detach().float().reshape(-1)
            finite = values[torch.isfinite(values)]
            finite_count = int(finite.numel())
            if finite_count <= 0:
                return stats
            zero_mask = torch.isclose(finite, torch.zeros_like(finite), rtol=0.0, atol=1e-6)
            one_mask = torch.isclose(finite, torch.ones_like(finite), rtol=0.0, atol=1e-6)
            non_binary = ~(zero_mask | one_mask)
            positive_gt0 = finite > 0
            negative_lt0 = finite < 0
            zero_count = int(zero_mask.sum().item())
            one_count = int(one_mask.sum().item())
            positive_count = int(positive_gt0.sum().item())
            non_binary_count = int(non_binary.sum().item())
            stats.update({
                "zero_count": zero_count,
                "one_count": one_count,
                "positive_count_gt0": positive_count,
                "negative_count_lt0": int(negative_lt0.sum().item()),
                "non_binary_count": non_binary_count,
                "positive_rate_gt0": float(positive_count / finite_count),
                "positive_rate_eq1": float(one_count / finite_count),
                "non_binary_rate": float(non_binary_count / finite_count),
            })
            return stats

    def _dense_batch_summary(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.dense_diagnostic_top_k <= 0:
            return []
        dense_stats = []
        for feature, spec in self.feature_map.features.items():
            if spec.get("type") != "numeric" or feature not in batch_data:
                continue
            tensor = batch_data[feature]
            if not hasattr(tensor, "detach"):
                continue
            stats = self._tensor_stats(tensor)
            if not stats or stats.get("finite_count", 0) <= 0:
                continue
            min_value = stats.get("min")
            max_value = stats.get("max")
            abs_max = max(abs(min_value), abs(max_value)) if min_value is not None and max_value is not None else None
            dense_stats.append({
                "feature": feature,
                "abs_max": abs_max,
                "min": min_value,
                "max": max_value,
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "non_finite_count": stats.get("non_finite_count", 0),
            })
        dense_stats.sort(key=lambda item: item.get("abs_max") or 0.0, reverse=True)
        return dense_stats[:self.dense_diagnostic_top_k]

    def _should_log_pointwise_diagnostics(self, epoch: int, batch_index: int) -> bool:
        return (
            self.diagnostic_epochs > 0
            and self.diagnostic_batches > 0
            and epoch < self.diagnostic_epochs
            and batch_index < self.diagnostic_batches
        )

    def _log_pointwise_batch_diagnostics(
            self,
            phase_name: str,
            epoch: int,
            batch_index: int,
            batch_data: Dict[str, Any],
            return_dict: Dict[str, Any],
            y_true: torch.Tensor,
            supervised_loss: torch.Tensor,
            regularization_loss: Any,
            total_loss: torch.Tensor,
            weighted_supervised_loss: Optional[torch.Tensor] = None) -> None:
        diagnostics = {
            "phase": phase_name,
            "epoch": epoch + 1,
            "batch": batch_index + 1,
            "batch_size": self._batch_size_from_batch(batch_data),
            "label": self._label_tensor_stats(y_true),
            "prediction": self._tensor_stats(return_dict.get("y_pred")),
            "logit": self._tensor_stats(return_dict.get("logit")),
            "loss": {
                "supervised": self._float_or_none(supervised_loss),
                "weighted_supervised": self._float_or_none(weighted_supervised_loss),
                "regularization": self._float_or_none(regularization_loss),
                "total": self._float_or_none(total_loss),
            },
            "dense_top_abs": self._dense_batch_summary(batch_data),
        }
        self.logger.info(
            "[%s] Pointwise training diagnostics: %s",
            phase_name,
            json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _batch_size_from_batch(batch_data: Dict[str, Any]) -> int:
        for value in batch_data.values():
            if hasattr(value, "size"):
                return int(value.size(0))
            try:
                return len(value)
            except TypeError:
                continue
        return 0

    @staticmethod
    def _expected_batch_count(data_generator) -> Any:
        try:
            expected_batches = len(data_generator)
        except TypeError:
            return "unknown"
        return expected_batches if expected_batches and expected_batches > 0 else "unknown"

    def _resolve_expected_batches(self, data_generator, batch_size: Optional[int] = None) -> Any:
        expected_batches = self._expected_batch_count(data_generator)
        if expected_batches != "unknown":
            return expected_batches

        num_batches = getattr(data_generator, "num_batches", None)
        if isinstance(num_batches, (int, np.integer)) and int(num_batches) > 0:
            return int(num_batches)

        effective_batch_size = int(batch_size or getattr(data_generator, "batch_size", 0) or 0)
        num_samples = getattr(data_generator, "num_samples", None)
        if effective_batch_size > 0 and isinstance(num_samples, (int, np.integer)) and int(num_samples) > 0:
            return int(np.ceil(int(num_samples) / effective_batch_size))

        stats_path = getattr(self.feature_map, "dataset_config", {}).get("_dataset_statistics_path")
        if not stats_path or effective_batch_size <= 0 or not os.path.isfile(stats_path):
            return "unknown"
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            train_rows = int(((stats.get("splits") or {}).get("train") or {}).get("rows", 0) or 0)
        except Exception:
            return "unknown"
        if train_rows <= 0:
            return "unknown"
        return int(np.ceil(train_rows / effective_batch_size))

    def _resolve_eval_interval_batches(self, expected_batches: Any) -> int:
        if self.eval_interval_batches > 0:
            return self.eval_interval_batches
        if self.eval_interval_epochs is None:
            return 0
        if not isinstance(expected_batches, (int, np.integer)) or int(expected_batches) <= 0:
            self.logger.warning(
                "preranking eval_interval_epochs=%.4g requested, but expected train batches are unknown; "
                "set preranking_eval_interval_batches to enable step-based evaluation.",
                self.eval_interval_epochs,
            )
            return 0
        return max(1, int(np.ceil(float(expected_batches) * self.eval_interval_epochs)))

    def _evaluate_during_training(
            self,
            phase_name: str,
            valid_data: Any,
            metrics: Dict[str, float],
            best_metric: float,
            stopping_steps: int,
            patience: int,
            mode: str,
            epoch: int,
            steps: int,
            total_examples: int,
            avg_loss: float,
            epoch_progress: Optional[float] = None,
            **kwargs) -> Tuple[float, int, bool]:
        if valid_data is None:
            self.logger.warning("[%s] Validation data is unavailable; skipping evaluation.", phase_name)
            return best_metric, stopping_steps, False

        if steps > 0:
            progress = epoch_progress if epoch_progress is not None else float(epoch + 1)
            self.logger.info(
                "[%s] Evaluating at epoch %.3f (epoch=%d, batch=%d, samples=%d, avg_loss=%.6f)...",
                phase_name,
                progress,
                epoch + 1,
                steps,
                total_examples,
                avg_loss,
            )
        else:
            self.logger.info("[%s] Evaluating epoch %d...", phase_name, epoch + 1)

        if kwargs.get("evaluation_mode") == "pointwise":
            valid_metrics = self.evaluate_pointwise(
                valid_data,
                metrics=kwargs.get("pointwise_metrics"),
                group_id_col=kwargs.get("pointwise_group_id_col"),
                split_name="valid",
            )
            validation_label = "Pointwise"
        else:
            valid_metrics = self.evaluate(valid_data)
            validation_label = "Ranking"
        metrics.update(valid_metrics)
        self.logger.info(f"[{phase_name}] Validation ({validation_label}): {valid_metrics}")

        curr_val = valid_metrics.get(self.monitor, 0.0)
        is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)
        should_stop = False

        if is_best:
            best_metric = curr_val
            stopping_steps = 0
            self.model.save_weights(self.best_weights_path)
            self.logger.info(f"[{phase_name}] New Best {self.monitor}={curr_val:.6f}! Model Saved.")
        else:
            stopping_steps += 1
            self.logger.info(f"[{phase_name}] No improvement. Patience {stopping_steps}/{patience}")

            if kwargs.get("reduce_lr_on_plateau", True):
                old_lr = self.model.optimizer.param_groups[0]['lr']
                new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                self.logger.info(f"[{phase_name}] Decay LR: {old_lr:.6f} -> {new_lr:.6f}")

            if stopping_steps >= patience:
                self.logger.info(f"[{phase_name}] Early Stopping.")
                should_stop = True

        return best_metric, stopping_steps, should_stop

    def _log_training_progress(
            self,
            phase_name: str,
            total_batches: int,
            total_examples: int,
            total_loss: float,
            epoch_start_time: float,
            last_log_time: float,
            last_log_examples: int,
            expected_batches: Any) -> Tuple[float, int]:
        now = time.time()
        elapsed = max(now - epoch_start_time, 1e-9)
        interval_elapsed = max(now - last_log_time, 1e-9)
        recent_examples = total_examples - last_log_examples
        recent_rps = recent_examples / interval_elapsed
        overall_rps = total_examples / elapsed
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        self.logger.info(
            "[%s] Epoch %d progress: batch=%d/%s, samples=%d, avg_loss=%.6f, "
            "records/s recent=%.1f overall=%.1f, elapsed=%.1fs, device=%s",
            phase_name,
            self.model._epoch_index + 1,
            total_batches,
            expected_batches,
            total_examples,
            avg_loss,
            recent_rps,
            overall_rps,
            elapsed,
            next(self.model.parameters()).device,
        )
        return now, total_examples

    @staticmethod
    def _safe_binary_metric(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a binary metric and return nan when the label set is degenerate."""
        try:
            if np.unique(y_true).size < 2:
                return float("nan")
            return float(fn(y_true, y_pred))
        except Exception:
            return float("nan")

    @staticmethod
    def _compute_pointwise_gauc(y_true: np.ndarray, y_pred: np.ndarray, group_id: Optional[np.ndarray]) -> Dict[str, float]:
        if group_id is None or len(group_id) == 0:
            return {}

        df = pd.DataFrame({
            "group_id": group_id,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        group_stats = df.groupby("group_id", sort=False)["y_true"].agg(["count", "sum"])
        valid_groups = group_stats[(group_stats["sum"] > 0) & (group_stats["sum"] < group_stats["count"])]
        if valid_groups.empty:
            return {
                "gAUC": float("nan"),
                "gauc_valid_groups": 0,
                "gauc_valid_samples": 0,
            }

        valid_group_ids = set(valid_groups.index.tolist())
        weighted_auc = 0.0
        total_weight = 0
        for gid, group_df in df[df["group_id"].isin(valid_group_ids)].groupby("group_id", sort=False):
            group_true = group_df["y_true"].to_numpy()
            group_pred = group_df["y_pred"].to_numpy()
            weight = int(group_true.size)
            weighted_auc += float(roc_auc_score(group_true, group_pred)) * weight
            total_weight += weight

        gauc = weighted_auc / total_weight if total_weight > 0 else float("nan")
        return {
            "gAUC": float(gauc),
            "gauc_valid_groups": int(len(valid_groups)),
            "gauc_valid_samples": int(total_weight),
        }

    def evaluate_pointwise(
            self,
            data_generator: Any,
            metrics: Optional[List[str]] = None,
            group_id_col: Optional[str] = None,
            score_path: Optional[str] = None,
            split_name: str = "valid") -> Dict[str, float]:
        """
        Evaluate the ranker directly on observed pointwise examples.

        This path mirrors platform-style CTR evaluation and intentionally avoids
        materializing per-request candidate pools.
        """
        if self.model is None:
            self.build_model()

        metrics = metrics or ["AUC", "logloss", "pcoc", "prauc"]
        metric_names = {str(metric).lower() for metric in metrics}
        y_pred_batches = []
        y_blend_pred_batches = []
        y_true_batches = []
        group_id_batches = []
        total_rows = 0

        self.model.eval()
        with torch.no_grad():
            for batch_data in data_generator:
                return_dict = self._apply_popularity_model_bias(
                    self.model.forward(batch_data),
                    batch_data,
                )
                batch_pred = return_dict["y_pred"].detach().cpu().numpy().reshape(-1)
                batch_logit = (
                    return_dict["logit"].detach().cpu().numpy().reshape(-1)
                    if "logit" in return_dict
                    else self._safe_logit_np(batch_pred)
                )
                popularity_prior = self._batch_popularity_prior(batch_data)
                if popularity_prior is not None:
                    blend_logit = batch_logit + self.popularity_blend_weight * popularity_prior
                    y_blend_pred_batches.append(self._sigmoid_np(blend_logit))
                batch_true = self.model.get_labels(batch_data).detach().cpu().numpy().reshape(-1)
                y_pred_batches.append(batch_pred)
                y_true_batches.append(batch_true)
                total_rows += int(batch_true.size)
                if group_id_col and group_id_col in batch_data:
                    group_id_batches.append(batch_data[group_id_col].detach().cpu().numpy().reshape(-1))

        if not y_true_batches:
            self.logger.warning("[PointwiseEval:%s] No samples were produced by data_generator.", split_name)
            return {}

        y_true = np.concatenate(y_true_batches).astype(np.float64)
        y_pred = np.concatenate(y_pred_batches).astype(np.float64)
        y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        y_blend_pred = None
        if y_blend_pred_batches and len(y_blend_pred_batches) == len(y_pred_batches):
            y_blend_pred = np.concatenate(y_blend_pred_batches).astype(np.float64)
            y_blend_pred = np.clip(y_blend_pred, 1e-7, 1.0 - 1e-7)
        group_id = np.concatenate(group_id_batches) if group_id_batches else None

        results: Dict[str, float] = {
            "sample_count": int(total_rows),
            "positive_ratio": float(np.mean(y_true)) if total_rows > 0 else float("nan"),
            "pred_min": float(np.min(y_pred)),
            "pred_mean": float(np.mean(y_pred)),
            "pred_max": float(np.max(y_pred)),
            "pred_p01": float(np.quantile(y_pred, 0.01)),
            "pred_p50": float(np.quantile(y_pred, 0.50)),
            "pred_p99": float(np.quantile(y_pred, 0.99)),
        }
        if "auc" in metric_names:
            results["AUC"] = self._safe_binary_metric(roc_auc_score, y_true, y_pred)
        if "logloss" in metric_names or "binary_crossentropy" in metric_names:
            try:
                results["logloss"] = float(log_loss(y_true, y_pred, labels=[0, 1]))
            except Exception:
                results["logloss"] = float("nan")
        if "prauc" in metric_names or "ap" in metric_names:
            results["prauc"] = self._safe_binary_metric(average_precision_score, y_true, y_pred)

        sum_label = float(np.sum(y_true))
        sum_pred = float(np.sum(y_pred))
        if "pcoc" in metric_names:
            results["pcoc"] = float(sum_pred / sum_label) if sum_label > 0 else float("nan")
        if "copc" in metric_names or "bucket_copc" in metric_names:
            results["copc"] = float(sum_label / sum_pred) if sum_pred > 0 else float("nan")
        if "gauc" in metric_names:
            results.update(self._compute_pointwise_gauc(y_true, y_pred, group_id))

        if y_blend_pred is not None:
            blend_prefix = "PopularityBlend"
            results[f"{blend_prefix}_pred_min"] = float(np.min(y_blend_pred))
            results[f"{blend_prefix}_pred_mean"] = float(np.mean(y_blend_pred))
            results[f"{blend_prefix}_pred_max"] = float(np.max(y_blend_pred))
            if "auc" in metric_names:
                results[f"{blend_prefix}AUC"] = self._safe_binary_metric(roc_auc_score, y_true, y_blend_pred)
            if "logloss" in metric_names or "binary_crossentropy" in metric_names:
                try:
                    results[f"{blend_prefix}logloss"] = float(log_loss(y_true, y_blend_pred, labels=[0, 1]))
                except Exception:
                    results[f"{blend_prefix}logloss"] = float("nan")
            if "prauc" in metric_names or "ap" in metric_names:
                results[f"{blend_prefix}prauc"] = self._safe_binary_metric(average_precision_score, y_true, y_blend_pred)
            blend_sum_pred = float(np.sum(y_blend_pred))
            if "pcoc" in metric_names:
                results[f"{blend_prefix}pcoc"] = float(blend_sum_pred / sum_label) if sum_label > 0 else float("nan")
            if "copc" in metric_names or "bucket_copc" in metric_names:
                results[f"{blend_prefix}copc"] = float(sum_label / blend_sum_pred) if blend_sum_pred > 0 else float("nan")
            if "gauc" in metric_names:
                blend_gauc = self._compute_pointwise_gauc(y_true, y_blend_pred, group_id)
                results.update({f"{blend_prefix}{key}": value for key, value in blend_gauc.items()})

        self.logger.info(
            "[PointwiseEval:%s] sample_count=%d positive_ratio=%.6f metrics=%s",
            split_name,
            total_rows,
            results["positive_ratio"],
            {k: v for k, v in results.items() if k not in {"sample_count", "positive_ratio"}},
        )

        if score_path:
            os.makedirs(os.path.dirname(score_path), exist_ok=True)
            score_df = pd.DataFrame({
                group_id_col or "row_id": group_id if group_id is not None else np.arange(total_rows),
                "label": y_true,
                "label_predict": y_pred,
            })
            if y_blend_pred is not None:
                score_df["label_predict_popularity_blend"] = y_blend_pred
            score_df.to_csv(score_path, index=False)
            self.logger.info("[PointwiseEval:%s] Saved score CSV to %s", split_name, score_path)

        return results

    def _log_active_feature_summary(self):
        """Log the feature groups that are actually retained in this stage's filtered feature_map."""
        impression_col = getattr(self.feature_map, 'dataset_config', {}).get('impression_id_col', 'impression_id')
        special_cols = {impression_col, 'group_id', 'click', 'clk', 'label', *self.feature_map.labels}

        active_model_features = [
            name for name in self.feature_map.features.keys()
            if name not in special_cols
        ]

        group_buckets = {
            FeatureGroup.FG1: [],
            FeatureGroup.FG2: [],
            FeatureGroup.FG3: [],
        }
        unassigned = []

        for feat_name in active_model_features:
            group = self.feature_group_manager.feature_assignments.get(feat_name)
            if group in group_buckets:
                group_buckets[group].append(feat_name)
            else:
                unassigned.append(feat_name)

        self.logger.info(
            "Active model features after filtering: total=%d, FG1=%d, FG2=%d, FG3=%d, unassigned=%d",
            len(active_model_features),
            len(group_buckets[FeatureGroup.FG1]),
            len(group_buckets[FeatureGroup.FG2]),
            len(group_buckets[FeatureGroup.FG3]),
            len(unassigned),
        )
        self.logger.info("Active FG1 Features: %s", ", ".join(group_buckets[FeatureGroup.FG1]) or "(none)")
        self.logger.info("Active FG2 Features: %s", ", ".join(group_buckets[FeatureGroup.FG2]) or "(none)")
        self.logger.info("Active FG3 Features: %s", ", ".join(group_buckets[FeatureGroup.FG3]) or "(none)")
        if unassigned:
            self.logger.info("Active Unassigned Features: %s", ", ".join(unassigned))

    def _read_item_feature_pool(self, item_pool_path: str, pool_name: str):
        """Read and index one item-feature pool without changing stage state."""
        self.logger.info("Loading %s item features from %s...", pool_name, item_pool_path)
        try:
            item_features_df = pd.read_parquet(item_pool_path)
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            if item_id_col in item_features_df.columns:
                item_features_df = item_features_df.set_index(item_id_col)
            return item_features_df
        except Exception as exc:
            self.logger.error("Failed to load %s item features: %s", pool_name, exc)
            raise

    def load_item_features(self, item_pool_path: str):
        """Load the valid/test item pool used by listwise evaluation and process()."""
        self.item_features_df = self._read_item_feature_pool(item_pool_path, "evaluation")
        self.logger.info("Loaded %d evaluation items into feature memory.", len(self.item_features_df))

    def load_negative_item_features(self, item_pool_path: str):
        """Load the train-only item pool used exclusively for explicit negatives."""
        self.train_negative_item_features_df = self._read_item_feature_pool(
            item_pool_path, "train-negative"
        )
        # Rebuild the sampler at train() so a newly loaded pool cannot leave a
        # stale sampler backed by a previous corpus.
        self.negative_sampler = None
        self.logger.info(
            "Loaded %d train-negative items into sampling memory.",
            len(self.train_negative_item_features_df),
        )

    def build_model(self):
        """Build and initialize the pre-ranking model using unified registry"""
        # Ensure output directories exist
        model_dir = os.path.join(self.output_dir, self.feature_map.dataset_id)
        os.makedirs(model_dir, exist_ok=True)

        # Default to the model_zoo PNN ranker for platform-style pointwise validation.
        model_name = self.model_params.get('model', 'PNN')

        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
        )
        try:
            parameter_device = next(self.model.parameters()).device
        except StopIteration:
            parameter_device = getattr(self.model, "device", "unknown")
        self.logger.info(
            "Built %s model, saving to %s; requested_gpu=%s, model.device=%s, parameter_device=%s",
            model_name,
            model_dir,
            self.model_params.get("gpu", -1),
            getattr(self.model, "device", "unknown"),
            parameter_device,
        )
        return self.model

    def _set_diversity_enabled(self, enabled: bool):
        """
        Toggle diversity loss on the model at runtime.

        Supports both wrapper-based models (model._diversity_enabled) and
        mixin-based models (model._use_diversity_loss).
        """
        if self.model is None:
            return
        # Wrapper-based model (from wrap_model_with_diversity)
        if hasattr(self.model, '_diversity_enabled'):
            self.model._diversity_enabled = enabled
        # Mixin-based model (DiversityLossMixin)
        if hasattr(self.model, '_use_diversity_loss'):
            self.model._use_diversity_loss = enabled
        self.logger.info(f"Diversity loss {'enabled' if enabled else 'disabled'} on model")

    def _set_diversity_lambda(self, lambda_value: float):
        """
        Set diversity lambda on the model at runtime.
        """
        if self.model is None:
            return
        if hasattr(self.model, '_diversity_lambda'):
            self.model._diversity_lambda = lambda_value
        if hasattr(self.model, '_diversity_logger'):
             # Optional: log adjust if needed, but per-epoch log is better
             pass

    def _compute_pointwise_loss(
            self,
            batch_data: Dict[str, Any],
            diagnostic_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return_dict = self._apply_popularity_model_bias(
            self.model.forward(batch_data),
            batch_data,
        )
        y_true = self.model.get_labels(batch_data)
        supervised_loss = self.model.loss_fn(return_dict["y_pred"], y_true, reduction="mean")
        weighted_supervised_loss = None
        if self.class_loss_weight is None:
            loss = self.model.compute_loss(return_dict, y_true)
            regularization_loss = self.model.regularization_loss() if hasattr(self.model, "regularization_loss") else 0.0
        else:
            per_sample_loss = self.model.loss_fn(return_dict["y_pred"], y_true, reduction="none")
            negative_weight, positive_weight = self.class_loss_weight
            weights = torch.where(
                y_true > 0.5,
                torch.as_tensor(positive_weight, device=y_true.device, dtype=per_sample_loss.dtype),
                torch.as_tensor(negative_weight, device=y_true.device, dtype=per_sample_loss.dtype),
            )
            weighted_supervised_loss = (per_sample_loss * weights).mean()
            regularization_loss = self.model.regularization_loss() if hasattr(self.model, "regularization_loss") else 0.0
            loss = weighted_supervised_loss + regularization_loss
        if diagnostic_context is not None:
            self._log_pointwise_batch_diagnostics(
                phase_name=diagnostic_context["phase_name"],
                epoch=diagnostic_context["epoch"],
                batch_index=diagnostic_context["batch_index"],
                batch_data=batch_data,
                return_dict=return_dict,
                y_true=y_true,
                supervised_loss=supervised_loss,
                regularization_loss=regularization_loss,
                total_loss=loss,
                weighted_supervised_loss=weighted_supervised_loss,
            )
        return loss

    def _train_step_pointwise(
            self,
            batch_data: Dict[str, Any],
            diagnostic_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if self.class_loss_weight is None and diagnostic_context is None:
            return self.model.train_step(batch_data)

        self.model.optimizer.zero_grad()
        loss = self._compute_pointwise_loss(
            batch_data=batch_data,
            diagnostic_context=diagnostic_context,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            getattr(self.model, "_max_gradient_norm", 10.0),
        )
        self.model.optimizer.step()
        return loss

    def _run_training_phase(self, phase_name, train_data, valid_data,
                            epochs, patience, mode, use_pairwise_training,
                            use_hybrid_training,
                            item_id_col, best_metric, metrics,
                            diversity_warmup_epochs=0, target_diversity_lambda=0.01,
                            **kwargs):
        """
        Run a single training phase (used by both Phase 1 and Phase 2).

        Args:
            phase_name: Name for logging (e.g., "Phase 1", "Phase 2")
            train_data: Training data generator
            valid_data: Validation data generator
            epochs: Max epochs for this phase
            patience: Early stopping patience
            mode: 'max' or 'min' for monitor metric
            use_pairwise_training: Whether to use pairwise/in-batch training
            use_hybrid_training: Whether to combine pointwise BCE with pairwise/in-batch auxiliary loss
            item_id_col: Name of item ID column
            best_metric: Starting best metric value
            metrics: Metrics dict to update (mutated in-place)
            diversity_warmup_epochs: Number of epochs to warmup diversity lambda
            target_diversity_lambda: Final target value for diversity lambda
            **kwargs: Additional training parameters

        Returns:
            Updated best_metric value
        """
        stopping_steps = 0

        for epoch in range(epochs):
            self.model._epoch_index = epoch

            # --- Diversity Warmup Logic ---
            if diversity_warmup_epochs > 0 and self.use_diversity_loss:
                if epoch < diversity_warmup_epochs:
                    warmup_lambda = (epoch + 1) / diversity_warmup_epochs * target_diversity_lambda
                    # Clamp to target
                    current_lambda = min(warmup_lambda, target_diversity_lambda)
                    self._set_diversity_lambda(current_lambda)
                    self.logger.info(f"[{phase_name}] Diversity Warmup: lambda={current_lambda:.6f} (Epoch {epoch+1}/{diversity_warmup_epochs})")
                else:
                    # Ensure it is at target
                    if getattr(self.model, '_diversity_lambda', 0) != target_diversity_lambda:
                        self._set_diversity_lambda(target_diversity_lambda)
                        self.logger.info(f"[{phase_name}] Diversity Warmup Complete: lambda={target_diversity_lambda}")

            self.logger.info(f"*** [{phase_name}] Epoch {epoch + 1}/{epochs} ***")

            # Training Loop
            self.model.train()
            total_loss = 0.0
            steps = 0
            total_examples = 0
            expected_batches = self._resolve_expected_batches(
                train_data,
                batch_size=kwargs.get("batch_size"),
            )
            eval_interval_batches = self._resolve_eval_interval_batches(expected_batches)
            epoch_start_time = time.time()
            last_log_time = epoch_start_time
            last_log_examples = 0
            phase_should_stop = False
            if self.progress_interval > 0:
                self.logger.info(
                    "[%s] Epoch %d started: expected_batches=%s, progress_interval=%d, "
                    "eval_interval_batches=%s, class_loss_weight=%s, device=%s",
                    phase_name,
                    epoch + 1,
                    expected_batches,
                    self.progress_interval,
                    eval_interval_batches if eval_interval_batches > 0 else "epoch_end",
                    self.class_loss_weight,
                    next(self.model.parameters()).device,
                )

            for batch_data in train_data:
                self.model._batch_index = steps
                self.model._total_steps += 1
                if use_hybrid_training:
                    diagnostic_context = None
                    if self._should_log_pointwise_diagnostics(epoch, steps):
                        diagnostic_context = {
                            "phase_name": phase_name,
                            "epoch": epoch,
                            "batch_index": steps,
                        }
                    loss = self._train_step_hybrid(
                        batch_data,
                        item_id_col,
                        torch,
                        diagnostic_context=diagnostic_context,
                    )
                elif use_pairwise_training:
                    loss = self._train_step_with_negatives(batch_data, item_id_col, torch)
                else:
                    diagnostic_context = None
                    if self._should_log_pointwise_diagnostics(epoch, steps):
                        diagnostic_context = {
                            "phase_name": phase_name,
                            "epoch": epoch,
                            "batch_index": steps,
                        }
                    loss = self._train_step_pointwise(batch_data, diagnostic_context=diagnostic_context)

                total_loss += loss.item()
                steps += 1
                total_examples += self._batch_size_from_batch(batch_data)
                if self.progress_interval > 0 and steps % self.progress_interval == 0:
                    last_log_time, last_log_examples = self._log_training_progress(
                        phase_name=phase_name,
                        total_batches=steps,
                        total_examples=total_examples,
                        total_loss=total_loss,
                        epoch_start_time=epoch_start_time,
                        last_log_time=last_log_time,
                        last_log_examples=last_log_examples,
                        expected_batches=expected_batches,
                    )
                should_eval_mid_epoch = (
                    eval_interval_batches > 0
                    and steps % eval_interval_batches == 0
                    and not (isinstance(expected_batches, (int, np.integer)) and steps >= int(expected_batches))
                )
                if should_eval_mid_epoch:
                    avg_loss_so_far = total_loss / steps if steps > 0 else 0.0
                    if isinstance(expected_batches, (int, np.integer)) and int(expected_batches) > 0:
                        epoch_progress = epoch + min(float(steps) / float(expected_batches), 1.0)
                    else:
                        epoch_progress = None
                    best_metric, stopping_steps, phase_should_stop = self._evaluate_during_training(
                        phase_name=phase_name,
                        valid_data=valid_data,
                        metrics=metrics,
                        best_metric=best_metric,
                        stopping_steps=stopping_steps,
                        patience=patience,
                        mode=mode,
                        epoch=epoch,
                        steps=steps,
                        total_examples=total_examples,
                        avg_loss=avg_loss_so_far,
                        epoch_progress=epoch_progress,
                        **kwargs,
                    )
                    if phase_should_stop:
                        break
                    self.model.train()

            avg_loss = total_loss / steps if steps > 0 else 0.0
            elapsed = max(time.time() - epoch_start_time, 1e-9)
            records_per_second = total_examples / elapsed if total_examples > 0 else 0.0
            if use_hybrid_training:
                self.logger.info(
                    f"[{phase_name}] Train Loss (hybrid {self.loss_type}): {avg_loss:.6f} "
                    f"(samples={total_examples}, batches={steps}, records/s={records_per_second:.1f})"
                )
            elif use_pairwise_training:
                self.logger.info(
                    f"[{phase_name}] Train Loss ({self.loss_type}): {avg_loss:.6f} "
                    f"(samples={total_examples}, batches={steps}, records/s={records_per_second:.1f})"
                )
            else:
                self.logger.info(
                    f"[{phase_name}] Train Loss: {avg_loss:.6f} "
                    f"(samples={total_examples}, batches={steps}, records/s={records_per_second:.1f})"
                )

            if phase_should_stop:
                break

            # Validation
            best_metric, stopping_steps, phase_should_stop = self._evaluate_during_training(
                phase_name=phase_name,
                valid_data=valid_data,
                metrics=metrics,
                best_metric=best_metric,
                stopping_steps=stopping_steps,
                patience=patience,
                mode=mode,
                epoch=epoch,
                steps=steps,
                total_examples=total_examples,
                avg_loss=avg_loss,
                epoch_progress=float(epoch + 1),
                **kwargs,
            )
            if phase_should_stop:
                break

        return best_metric

    @staticmethod
    def _uses_dp_gradient_perturbation(model: Any) -> bool:
        """Return True when a model exposes DP-aware gradient update hooks."""
        return (
            hasattr(model, "_clip_gradients")
            and callable(getattr(model, "_clip_gradients"))
            and hasattr(model, "_add_dp_noise")
            and callable(getattr(model, "_add_dp_noise"))
            and hasattr(model, "max_grad_norm_per_sample")
            and hasattr(model, "noise_multiplier")
        )

    @classmethod
    def _apply_gradient_update(cls, model: Any, loss: torch.Tensor, batch_size: int):
        """
        Apply one optimizer update, dispatching to model-specific DP logic when available.

        The preranking pairwise loop bypasses ``model.train_step()``, so DP models such
        as DPSGD need their clipping/noise path re-applied here.
        """
        model.optimizer.zero_grad()
        loss.backward()

        cls._finalize_gradient_update(model, batch_size)

    @classmethod
    def _finalize_gradient_update(cls, model: Any, batch_size: int):
        """Finalize one optimizer update after gradients have already been accumulated."""

        if cls._uses_dp_gradient_perturbation(model):
            model._clip_gradients()
            model._add_dp_noise(batch_size)
            if hasattr(model, "_dp_steps"):
                model._dp_steps += 1
            if hasattr(model, "_total_samples"):
                model._total_samples += batch_size
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model._max_gradient_norm)

        model.optimizer.step()

    def _get_pairwise_scores(self, model_output: Dict[str, Any]) -> torch.Tensor:
        """Extract the score tensor used by pairwise and in-batch losses."""
        if self.use_logit:
            return model_output.get('logit', model_output['y_pred'])
        return model_output['y_pred']

    @staticmethod
    def _supports_pairwise_auxiliary_loss(model: Any) -> bool:
        """Return True when pairwise training should preserve model-specific auxiliary losses."""
        return type(model).__name__ in {"DualRec", "FedCAR", "FedCIA"}

    def _compute_pairwise_auxiliary_loss(self, model_output: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Re-apply model-specific auxiliary losses when pairwise training bypasses model.train_step().

        The pairwise preranking loop optimizes ranking losses directly and therefore skips the
        BaseModel -> add_loss() path where several privacy-preserving models define their
        collaborative regularizers. This helper mirrors only those auxiliary terms, leaving the
        pairwise ranking loss as the primary supervised signal for positives vs. negatives.
        """
        model_name = type(self.model).__name__
        aux_loss = None

        if model_name == "DualRec":
            personalized_mask = model_output["personalized_mask"]
            if personalized_mask.any():
                kd_loss = self.model._compute_kd_loss(
                    model_output["device_logit"][personalized_mask],
                    model_output["cloud_logit"][personalized_mask].detach(),
                )
                aux_loss = self.model.kd_loss_weight * kd_loss

                if self.model.mutual_reg_weight > 0:
                    reverse_kd = self.model._compute_kd_loss(
                        model_output["cloud_logit"][personalized_mask],
                        model_output["device_logit"][personalized_mask].detach(),
                    )
                    aux_loss = aux_loss + self.model.mutual_reg_weight * reverse_kd

            if self.model.odr_loss_weight > 0:
                odr_loss = self.model._compute_odr_loss(
                    model_output["device_logit"],
                    model_output["cloud_logit"],
                )
                aux_loss = odr_loss * self.model.odr_loss_weight if aux_loss is None else aux_loss + self.model.odr_loss_weight * odr_loss

            return aux_loss

        if model_name == "FedCAR":
            if self.model.contrastive_weight > 0:
                contrastive_loss = self.model._info_nce_loss(
                    model_output["cloud_proj"].detach(),
                    model_output["device_proj"],
                )
                aux_loss = self.model.contrastive_weight * contrastive_loss

            if self.model.use_prototype and self.model.training:
                self.model._update_prototype(model_output["cloud_proj"])
                device_proj_norm = torch.nn.functional.normalize(model_output["device_proj"], dim=-1)
                proto_norm = torch.nn.functional.normalize(self.model.global_prototype.unsqueeze(0), dim=-1)
                prototype_loss = 1 - (device_proj_norm * proto_norm).sum(dim=-1).mean()
                weighted_proto_loss = self.model.prototype_weight * prototype_loss
                aux_loss = weighted_proto_loss if aux_loss is None else aux_loss + weighted_proto_loss

            return aux_loss

        if model_name == "FedCIA":
            cloud_sim = self.model._compute_similarity_matrix(
                model_output["cloud_latent"].detach(),
                add_noise=True,
            )
            device_sim = self.model._compute_similarity_matrix(
                model_output["device_latent"],
                add_noise=False,
            )
            align_loss = self.model.similarity_align_weight * torch.nn.functional.mse_loss(device_sim, cloud_sim)
            aux_loss = align_loss

            if self.model.reverse_align_weight > 0:
                cloud_sim_live = self.model._compute_similarity_matrix(
                    model_output["cloud_latent"],
                    add_noise=False,
                )
                device_sim_detached = self.model._compute_similarity_matrix(
                    model_output["device_latent"].detach(),
                    add_noise=False,
                )
                reverse_loss = self.model.reverse_align_weight * torch.nn.functional.mse_loss(
                    cloud_sim_live,
                    device_sim_detached,
                )
                aux_loss = aux_loss + reverse_loss

            return aux_loss

        return None

    def _get_item_feature_keys(self, item_id_col: str) -> set:
        """Infer which batch columns should be swapped when replacing candidate items."""
        item_feature_keys = {item_id_col}
        if self.item_features_df is not None:
            item_feature_keys.update(self.item_features_df.columns)
        if self.feature_group_manager is not None:
            for feat_name, group in self.feature_group_manager.feature_assignments.items():
                if group == FeatureGroup.FG1:
                    item_feature_keys.add(feat_name)
        return item_feature_keys

    def _filter_batch_rows(
            self,
            batch_data: Dict[str, Any],
            row_mask: torch.Tensor) -> Tuple[Dict[str, Any], int]:
        """Return a batch containing only rows where row_mask is true."""
        flat_mask = row_mask.detach().reshape(-1).bool()
        batch_size = int(flat_mask.numel())
        indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
        selected = int(indices.numel())
        filtered = {}

        for key, val in batch_data.items():
            if hasattr(val, 'size') and callable(getattr(val, 'size')):
                try:
                    if int(val.size(0)) == batch_size:
                        filtered[key] = val.index_select(0, indices.to(val.device))
                        continue
                except (TypeError, IndexError, RuntimeError):
                    pass
            if isinstance(val, np.ndarray) and len(val) == batch_size:
                filtered[key] = val[flat_mask.detach().cpu().numpy()]
                continue
            try:
                if len(val) == batch_size and not isinstance(val, (str, bytes)):
                    keep = set(indices.detach().cpu().numpy().tolist())
                    filtered[key] = [item for idx, item in enumerate(val) if idx in keep]
                    continue
            except TypeError:
                pass
            filtered[key] = val

        return filtered, selected

    def _positive_anchor_batch(
            self,
            batch_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int]:
        """Extract positive rows from a mixed pointwise batch for ranking auxiliary loss."""
        y_true = self.model.get_labels(batch_data)
        positive_mask = y_true.detach().reshape(-1) > 0.5
        if not bool(positive_mask.any().item()):
            return None, 0
        return self._filter_batch_rows(batch_data, positive_mask)

    def _compute_explicit_negative_loss(
            self,
            batch_data: Dict[str, Any],
            item_id_col: str,
            torch_module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute explicit sampled-negative ranking loss without applying gradients."""
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])

        neg_batch_dict = None
        neg_scores_flat = None
        neg_scores = None
        pos_scores = None
        pos_output = None
        aux_loss = None

        if self.num_negatives > 0 or self._supports_pairwise_auxiliary_loss(self.model):
            raw_pos_output = self.model.forward(batch_data)
            pos_output = self._apply_popularity_model_bias(raw_pos_output, batch_data)
            if self._supports_pairwise_auxiliary_loss(self.model):
                aux_loss = self._compute_pairwise_auxiliary_loss(raw_pos_output)

        if self.num_negatives > 0:
            if self.negative_sampler is None:
                raise ValueError("Negative sampler not initialized. Call train() after loading item features.")

            pos_scores = self._get_pairwise_scores(pos_output)
            pos_item_ids = batch_dict[item_id_col].cpu().numpy()
            neg_item_ids = self.negative_sampler.sample_negatives_batch(
                pos_item_ids, self.num_negatives
            )

            neg_ids_flat = neg_item_ids.reshape(-1)
            neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)

            neg_batch_dict = {}
            for key, val in batch_dict.items():
                if key == item_id_col:
                    neg_batch_dict[key] = torch_module.tensor(neg_ids_flat, device=self.model.device)
                elif key in neg_features.columns:
                    feature_values = neg_features[key].to_numpy(copy=False)
                    if not np.isscalar(feature_values[0]):
                        feature_values = np.vstack(feature_values)
                    neg_batch_dict[key] = torch_module.tensor(feature_values, device=self.model.device)
                else:
                    if hasattr(val, 'to'):
                        val = val.to(self.model.device)
                        neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                    else:
                        neg_batch_dict[key] = val

            neg_output = self._apply_popularity_model_bias(
                self.model.forward(neg_batch_dict),
                neg_batch_dict,
            )
            neg_scores_flat = self._get_pairwise_scores(neg_output)
            neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)

        explicit_loss = None
        if neg_scores is not None:
            if self.loss_type == 'bpr':
                explicit_loss = bpr_loss(pos_scores, neg_scores)
            elif self.loss_type == 'margin':
                explicit_loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
            elif self.loss_type in ('softmax', 'sampled_softmax'):
                explicit_loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return explicit_loss, aux_loss, neg_batch_dict, neg_scores_flat, pos_scores

    def _build_in_batch_chunk(
        self,
        batch_dict: Dict[str, Any],
        item_feature_keys: set,
        row_start: int,
        row_end: int,
        item_indices: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build one in-batch chunk with explicit candidate indices for each user row."""
        candidates_per_row = item_indices.size(1)
        flat_item_indices = item_indices.reshape(-1)
        pair_batch_dict = {}

        for key, val in batch_dict.items():
            if hasattr(val, 'to'):
                val = val.to(self.model.device)
                if key in item_feature_keys:
                    pair_batch_dict[key] = val.index_select(0, flat_item_indices)
                else:
                    user_chunk = val[row_start:row_end]
                    pair_batch_dict[key] = user_chunk.repeat_interleave(candidates_per_row, dim=0)
            else:
                pair_batch_dict[key] = val
        return pair_batch_dict

    @staticmethod
    def _sample_in_batch_negative_indices(
        batch_size: int,
        row_start: int,
        row_end: int,
        negatives_per_row: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample batch-local negatives with replacement while excluding each row's own positive index.
        """
        row_count = row_end - row_start
        sampled = torch.randint(0, batch_size - 1, (row_count, negatives_per_row), device=device)
        row_indices = torch.arange(row_start, row_end, device=device).unsqueeze(1)
        return sampled + (sampled >= row_indices).long()

    def _backward_in_batch_loss(
        self,
        batch_dict: Dict[str, Any],
        item_id_col: str,
        loss_weight: float = 1.0,
    ) -> float:
        """
        Backprop the in-batch cross-entropy loss chunk-by-chunk to cap peak activation memory.
        """
        batch_size = len(batch_dict[item_id_col])
        chunk_size = min(self.in_batch_negative_chunk_size, batch_size)
        max_negatives = max(batch_size - 1, 0)
        use_sampled_negatives = (
            self.in_batch_negative_sample_size > 0
            and self.in_batch_negative_sample_size < max_negatives
        )
        item_feature_keys = self._get_item_feature_keys(item_id_col)
        total_loss_value = 0.0

        for row_start in range(0, batch_size, chunk_size):
            row_end = min(row_start + chunk_size, batch_size)
            row_count = row_end - row_start
            row_indices = torch.arange(row_start, row_end, device=self.model.device)

            if use_sampled_negatives:
                neg_indices = self._sample_in_batch_negative_indices(
                    batch_size=batch_size,
                    row_start=row_start,
                    row_end=row_end,
                    negatives_per_row=self.in_batch_negative_sample_size,
                    device=self.model.device,
                )
                item_indices = torch.cat([row_indices.unsqueeze(1), neg_indices], dim=1)
                in_batch_targets = torch.zeros(row_count, dtype=torch.long, device=self.model.device)
            else:
                item_indices = torch.arange(batch_size, device=self.model.device).unsqueeze(0).expand(row_count, -1)
                in_batch_targets = row_indices

            pair_batch_dict = self._build_in_batch_chunk(
                batch_dict=batch_dict,
                item_feature_keys=item_feature_keys,
                row_start=row_start,
                row_end=row_end,
                item_indices=item_indices,
            )
            pair_output = self.model.forward(pair_batch_dict)
            pair_scores = self._get_pairwise_scores(pair_output).reshape(row_count, item_indices.size(1))
            chunk_loss = torch.nn.functional.cross_entropy(
                pair_scores,
                in_batch_targets,
                reduction='sum',
            )
            scaled_chunk_loss = loss_weight * chunk_loss / batch_size
            total_loss_value += scaled_chunk_loss.detach().item()
            scaled_chunk_loss.backward()

        return total_loss_value

    def _initialize_negative_sampler(self, item_id_col: str) -> bool:
        if self.num_negatives <= 0:
            self.negative_sampler = None
            return False
        train_negative_item_features_df = getattr(self, "train_negative_item_features_df", None)
        if train_negative_item_features_df is None:
            self.negative_sampler = None
            self.logger.warning(
                "num_negatives > 0 but the train-only negative item pool is not loaded. "
                "Explicit negative sampling disabled rather than sampling from an evaluation pool."
            )
            return False

        item_popularity = None
        if self.negative_sampling_strategy in {"popularity", "popular", "hot"}:
            item_popularity = self.popularity_prior_counts
            if not item_popularity:
                self.logger.warning(
                    "Popularity-weighted negative sampling requested but popularity prior is unavailable; "
                    "falling back to uniform negatives."
                )
        self.negative_sampler = NegativeSampler(
            train_negative_item_features_df,
            item_id_col=item_id_col,
            item_popularity=item_popularity,
            popularity_alpha=self.negative_sampling_popularity_alpha,
        )
        self.logger.info(
            "Negative Sampling: %d negatives per positive, loss_type=%s, strategy=%s, popularity_alpha=%.3g",
            self.num_negatives,
            self.loss_type,
            self.negative_sampling_strategy,
            self.negative_sampling_popularity_alpha,
        )
        return True

    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the pre-ranking model with custom loop and best model monitoring.

        Supports two-phase training when use_diversity_loss is enabled:
        - Phase 1: Train without diversity loss until early stopping
        - Phase 2: Load best weights, enable diversity loss, fine-tune

        Args:
            train_data: Training data generator (positive examples only for negative sampling)
            valid_data: Validation data generator (used for internal training)
            **kwargs: Training parameters including:
                - epochs: Number of training epochs
                - patience: Early stopping patience
                - mode: 'max' or 'min' for monitor metric
                - reduce_lr_on_plateau: Whether to decay LR on no improvement
                - lr_decay_factor: LR decay factor (default 0.1)

        Returns:
            Training metrics
        """

        if self.model is None:
            self.build_model()

        try:
            parameter_device = next(self.model.parameters()).device
        except StopIteration:
            parameter_device = getattr(self.model, "device", "unknown")
        self.logger.info(
            "Starting pre-ranking model training (Custom Loop); requested_gpu=%s, model.device=%s, parameter_device=%s",
            self.model_params.get("gpu", -1),
            getattr(self.model, "device", "unknown"),
            parameter_device,
        )
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")

        epochs = kwargs.pop('epochs', 1)
        patience = kwargs.pop('patience', self.patience)
        mode = kwargs.pop('mode', 'max')
        initial_lr = kwargs.pop('learning_rate', self.model_params.get('learning_rate', 1e-3))
        metrics = {}
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')

        # Initialize negative sampler if using explicit negative sampling
        use_negative_sampling = self._initialize_negative_sampler(item_id_col)
        use_pairwise_training = use_negative_sampling or self.use_in_batch_negatives
        use_hybrid_training = self.use_hybrid_loss and use_pairwise_training
        if self.use_hybrid_loss and not use_pairwise_training:
            self.logger.warning(
                "hybrid_loss=true but no explicit or in-batch negatives are enabled; "
                "falling back to pointwise training."
            )
        if self.use_in_batch_negatives:
            if self.in_batch_negative_sample_size > 0:
                self.logger.info(
                    "In-batch negatives enabled (sampled cross-entropy, chunk_size=%d, negatives_per_example=%d)",
                    self.in_batch_negative_chunk_size,
                    self.in_batch_negative_sample_size,
                )
            else:
                self.logger.info(
                    "In-batch negatives enabled (full-batch cross-entropy, chunk_size=%d)",
                    self.in_batch_negative_chunk_size,
                )
        if use_pairwise_training and self._supports_pairwise_auxiliary_loss(self.model):
            self.logger.info(
                "Pairwise training will preserve model-specific auxiliary losses for %s.",
                type(self.model).__name__,
            )

        # Ensure optimizer is initialized
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
             self.logger.info("Initializing optimizer...")
             self.model.compile(
                 optimizer=kwargs.get("optimizer", self.model_params.get("optimizer", "adam")),
                 loss=self.model_params.get("loss", "binary_crossentropy"),
                 lr=initial_lr
             )

        # Setup model for manual training (required by train_step)
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0
        self.progress_interval = max(
            0,
            int(kwargs.get("progress_interval", self.model_params.get("progress_interval", self.progress_interval)) or 0),
        )
        self.eval_interval_batches = max(
            0,
            int(kwargs.get(
                "eval_interval_batches",
                self.model_params.get("eval_interval_batches", self.eval_interval_batches),
            ) or 0),
        )
        self.eval_interval_epochs = self._as_optional_positive_float(
            kwargs.get(
                "eval_interval_epochs",
                self.model_params.get("eval_interval_epochs", self.eval_interval_epochs),
            )
        )
        self.logger.info(
            "Preranking training config: model=%s, batch_size=%s, learning_rate=%.6g, "
            "monitor=%s, progress_interval=%d, eval_interval_batches=%d, "
            "eval_interval_epochs=%s, use_inner=%s, use_outer=%s, "
            "use_feature_encoder=%s, class_loss_weight=%s, diagnostic_batches=%d, "
            "diagnostic_epochs=%d, dense_diagnostic_top_k=%d, "
            "training_mode=%s, hybrid_loss=%s, hybrid_pointwise_weight=%.6g, "
            "hybrid_pairwise_weight=%.6g, "
            "loss_type=%s, num_negatives=%d, use_in_batch_negatives=%s, "
            "use_diversity_loss=%s, diversity_num_negatives=%d, "
            "negative_sampling_strategy=%s, "
            "popularity_blend_weight=%.6g, popularity_model_bias_weight=%.6g",
            self.model_params.get("model", type(self.model).__name__),
            kwargs.get("batch_size", "<loader>"),
            initial_lr,
            self.monitor,
            self.progress_interval,
            self.eval_interval_batches,
            self.eval_interval_epochs if self.eval_interval_epochs is not None else "<epoch_end>",
            self.model_params.get("use_inner", "<unset>"),
            self.model_params.get("use_outer", "<unset>"),
            self.model_params.get("use_feature_encoder", "<unset>"),
            self.class_loss_weight,
            self.diagnostic_batches,
            self.diagnostic_epochs,
            self.dense_diagnostic_top_k,
            self.training_mode,
            self.use_hybrid_loss,
            self.hybrid_pointwise_loss_weight,
            self.hybrid_pairwise_loss_weight,
            self.loss_type,
            self.num_negatives,
            self.use_in_batch_negatives,
            self.use_diversity_loss,
            self.diversity_num_negatives,
            self.negative_sampling_strategy,
            self.popularity_blend_weight,
            self.popularity_model_bias_weight,
        )

        best_metric = -np.inf if mode == "max" else np.inf

        # ======================================================================
        # Determine training strategy
        # ======================================================================
        use_two_phase = self.use_diversity_loss  # Two-phase only when diversity is requested

        if use_two_phase:
            # --- Phase 1: Train WITHOUT diversity loss ---
            self._set_diversity_enabled(False)
            # Also disable diversity in the stage-level flag for _train_step_with_negatives
            phase1_use_diversity = self.use_diversity_loss
            self.use_diversity_loss = False

            if self.diversity_start_epoch != 0:
                phase1_epochs = self.diversity_start_epoch if self.diversity_start_epoch > 0 else epochs
                self.logger.info(
                    f"=== Phase 1: Base Training (no diversity) ==="
                    f" epochs={phase1_epochs}, monitor={self.monitor}, patience={patience}"
                )
                best_metric = self._run_training_phase(
                    phase_name="Phase 1",
                    train_data=train_data,
                    valid_data=valid_data,
                    epochs=phase1_epochs,
                    patience=patience,
                    mode=mode,
                    use_pairwise_training=use_pairwise_training,
                    use_hybrid_training=use_hybrid_training,
                    item_id_col=item_id_col,
                    best_metric=best_metric,
                    metrics=metrics,
                    **kwargs
                )
                # Restore best weights from Phase 1 as starting point for Phase 2
                if os.path.exists(self.best_weights_path):
                    self.model.load_weights(self.best_weights_path)
                    self.logger.info(f"Phase 1 complete. Best {self.monitor}={best_metric:.6f}. Loaded best weights.")

            # --- Phase 2: Fine-tune WITH diversity loss ---
            self.use_diversity_loss = phase1_use_diversity  # Restore the flag
            self._set_diversity_enabled(True)

            # Reset learning rate to initial value for Phase 2
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = initial_lr
            self.logger.info(f"Reset learning rate to {initial_lr} for Phase 2")

            self.logger.info(
                f"=== Phase 2: Diversity Fine-tuning ==="
                f" epochs={self.diversity_epochs}, monitor={self.monitor}, patience={patience}, warmup={self.diversity_warmup_epochs}"
            )

            # Verify target lambda
            try:
                target_lambda = float(self.model_params.get('diversity_lambda', 0.7))
            except (TypeError, ValueError):
                target_lambda = 0.7
            lr_decay_factor = kwargs.pop("lr_decay_factor", 0.5)
            self.num_negatives = self.diversity_num_negatives  # Update num_negatives for Phase 2 if specified
            phase2_use_negative_sampling = self._initialize_negative_sampler(item_id_col)
            phase2_use_pairwise_training = phase2_use_negative_sampling or self.use_in_batch_negatives
            phase2_use_hybrid_training = self.use_hybrid_loss and phase2_use_pairwise_training
            if self.use_diversity_loss and not phase2_use_negative_sampling:
                self.logger.warning(
                    "Diversity loss is enabled but explicit negative sampling is unavailable; "
                    "diversity loss will not be computed in Phase 2."
                )
            best_metric = self._run_training_phase(
                phase_name="Phase 2 (Diversity)",
                train_data=train_data,
                valid_data=valid_data,
                epochs=self.diversity_epochs,
                patience=patience,
                mode=mode,
                use_pairwise_training=phase2_use_pairwise_training,
                use_hybrid_training=phase2_use_hybrid_training,
                item_id_col=item_id_col,
                best_metric=best_metric,
                metrics=metrics,
                diversity_warmup_epochs=self.diversity_warmup_epochs,
                target_diversity_lambda=target_lambda,
                lr_decay_factor=lr_decay_factor,
                **kwargs
            )
        else:
            # --- Single-phase training (no diversity) ---
            self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")

            best_metric = self._run_training_phase(
                phase_name="Training",
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                patience=patience,
                mode=mode,
                use_pairwise_training=use_pairwise_training,
                use_hybrid_training=use_hybrid_training,
                item_id_col=item_id_col,
                best_metric=best_metric,
                metrics=metrics,
                **kwargs
            )
        self.logger.info(f"Training complete. Best {self.monitor}={best_metric:.6f}.")

        # Restore best weights
        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Restored best weights from {self.best_weights_path}")

        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "valid_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])

        return metrics

    def _train_step_with_negatives(self, batch_data, item_id_col: str, torch):
        """
        Single training step with explicit negatives and/or in-batch negatives.

        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column
            torch: Torch module reference

        Returns:
            Loss tensor
        """
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])
        explicit_loss, aux_loss, neg_batch_dict, neg_scores_flat, pos_scores = (
            self._compute_explicit_negative_loss(batch_data, item_id_col, torch)
        )

        if explicit_loss is None and not self.use_in_batch_negatives:
            raise ValueError("Pairwise training requires explicit negatives or use_in_batch_negatives=True.")

        in_batch_loss_weight = 0.0
        if self.use_in_batch_negatives:
            if explicit_loss is not None:
                explicit_loss = 0.5 * explicit_loss
                in_batch_loss_weight = 0.5
            else:
                in_batch_loss_weight = 1.0

        loss = None
        if explicit_loss is not None:
            loss = explicit_loss

        # Add per-user diversity loss if enabled
        if self.use_diversity_loss and neg_batch_dict is not None and neg_scores_flat is not None:
            diversity_theta = self.model_params.get('diversity_theta', 0.7)
            diversity_lambda = getattr(self.model, '_diversity_lambda', self.model_params.get('diversity_lambda', 0.01))
            diversity_kernel = self.model_params.get('diversity_kernel', 'gram')
            diversity_gamma = self.model_params.get('diversity_gamma', 1.0)
            diversity_delta = compute_diversity_loss_per_user(
                model=self.model,
                pos_inputs=batch_data,
                neg_inputs=neg_batch_dict,
                pos_scores=pos_scores,
                neg_scores_flat=neg_scores_flat,
                num_negatives=self.num_negatives,
                theta=diversity_theta,
                lambda_=diversity_lambda,
                kernel=diversity_kernel,
                gamma=diversity_gamma,
            )
            loss = diversity_delta if loss is None else loss + diversity_delta

        if aux_loss is not None:
            loss = aux_loss if loss is None else loss + aux_loss

        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            reg_loss = self.model.regularization_loss()
            if hasattr(reg_loss, "detach"):
                if loss is None:
                    if getattr(reg_loss, "requires_grad", False):
                        loss = reg_loss
                else:
                    loss = loss + reg_loss
            elif reg_loss and loss is not None:
                loss = loss + float(reg_loss)

        self.model.optimizer.zero_grad()
        total_loss_value = 0.0

        if loss is not None and hasattr(loss, "detach"):
            total_loss_value += loss.detach().item()
            loss.backward()
        elif loss is not None:
            total_loss_value += float(loss)

        if in_batch_loss_weight > 0:
            total_loss_value += self._backward_in_batch_loss(
                batch_dict=batch_dict,
                item_id_col=item_id_col,
                loss_weight=in_batch_loss_weight,
            )

        self._finalize_gradient_update(self.model, batch_size)

        return torch.tensor(total_loss_value, device=self.model.device)

    def _train_step_hybrid(
            self,
            batch_data,
            item_id_col: str,
            torch,
            diagnostic_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Train on observed pointwise labels and add ranking-aware auxiliary loss.

        The pointwise term uses the full exposure batch, while explicit/in-batch
        negative losses use only label-positive rows as anchors. This keeps AUC
        calibration tied to observed labels and avoids treating observed negatives
        as positives for sampled-negative training.
        """
        batch_size = self._batch_size_from_batch(batch_data)
        self.model.optimizer.zero_grad()

        pointwise_loss = self._compute_pointwise_loss(
            batch_data=batch_data,
            diagnostic_context=diagnostic_context,
        )
        loss = self.hybrid_pointwise_loss_weight * pointwise_loss

        positive_batch, positive_count = self._positive_anchor_batch(batch_data)
        explicit_loss = None
        in_batch_loss_weight = 0.0
        if positive_batch is not None and positive_count > 0:
            explicit_loss, aux_loss, neg_batch_dict, neg_scores_flat, pos_scores = (
                self._compute_explicit_negative_loss(positive_batch, item_id_col, torch)
            )

            if explicit_loss is not None and self.use_in_batch_negatives:
                explicit_loss = 0.5 * explicit_loss
                in_batch_loss_weight = 0.5
            elif self.use_in_batch_negatives:
                in_batch_loss_weight = 1.0

            ranking_loss = explicit_loss
            if self.use_diversity_loss and neg_batch_dict is not None and neg_scores_flat is not None:
                diversity_theta = self.model_params.get('diversity_theta', 0.7)
                diversity_lambda = getattr(self.model, '_diversity_lambda', self.model_params.get('diversity_lambda', 0.01))
                diversity_kernel = self.model_params.get('diversity_kernel', 'gram')
                diversity_gamma = self.model_params.get('diversity_gamma', 1.0)
                diversity_delta = compute_diversity_loss_per_user(
                    model=self.model,
                    pos_inputs=positive_batch,
                    neg_inputs=neg_batch_dict,
                    pos_scores=pos_scores,
                    neg_scores_flat=neg_scores_flat,
                    num_negatives=self.num_negatives,
                    theta=diversity_theta,
                    lambda_=diversity_lambda,
                    kernel=diversity_kernel,
                    gamma=diversity_gamma,
                )
                ranking_loss = diversity_delta if ranking_loss is None else ranking_loss + diversity_delta

            if aux_loss is not None:
                ranking_loss = aux_loss if ranking_loss is None else ranking_loss + aux_loss

            if ranking_loss is not None:
                loss = loss + self.hybrid_pairwise_loss_weight * ranking_loss

        total_loss_value = 0.0
        if loss is not None and hasattr(loss, "detach"):
            total_loss_value += loss.detach().item()
            loss.backward()
        elif loss is not None:
            total_loss_value += float(loss)

        if positive_batch is not None and positive_count > 0 and in_batch_loss_weight > 0:
            total_loss_value += self._backward_in_batch_loss(
                batch_dict=dict(positive_batch),
                item_id_col=item_id_col,
                loss_weight=self.hybrid_pairwise_loss_weight * in_batch_loss_weight,
            )

        self._finalize_gradient_update(self.model, batch_size)
        return torch.tensor(total_loss_value, device=self.model.device)

    def process(self,
                input_data: StageOutput,
                **kwargs) -> Tuple[StageOutput, Dict[str, float]]:
        """
        Process candidates from retrieval stage - scores and selects top-K.

        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override, compute_metrics)

        Returns:
            Tuple of (StageOutput with Top-K candidates, metrics dict)
        """
        from ..metric_utils import process_and_rank_candidates
        if os.path.exists(self.best_weights_path) and kwargs.pop('load_best_model', True):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Loaded best weights from {self.best_weights_path}")
        else:
            if not os.path.exists(self.best_weights_path):
                self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")
            else:
                self.logger.info(f"Skipping loading best weights as per argument. Using current model state.")

        # Ensure item features are loaded
        if self.item_features_df is None:
            raise ValueError("Item features not loaded. Call load_item_features() first.")

        compute_metrics = kwargs.pop('compute_metrics', True)
        evaluate_pool_diversity = kwargs.pop('evaluate_pool_diversity', self.evaluate_pool_diversity)
        return_output = kwargs.pop('return_output', True)
        top_k = kwargs.pop('top_k', self.top_k)
        kwargs.setdefault('popularity_blend_weight', self.popularity_blend_weight)
        kwargs.setdefault('popularity_blend_transform', self.popularity_blend_transform)
        kwargs.setdefault('popularity_blend_normalize', self.popularity_blend_normalize)
        kwargs.setdefault('popularity_model_bias_weight', self.popularity_model_bias_weight)

        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=return_output,
            compute_metrics=compute_metrics,
            top_k=top_k,
            logger=self.logger,
            metrics_k=self.metrics_k,
            evaluate_pool_diversity=evaluate_pool_diversity,
            inference_batch_size=self.inference_batch_size,
            **kwargs
        )

    def evaluate(self,
                 input_data: StageOutput,
                 metrics_k: List[int] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate pre-ranking model with list-wise metrics.

        Args:
            input_data: StageOutput containing candidate sets with labels
            metrics_k: List of K values for Recall@K and nDCG@K
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation metrics
        """
        from ..metric_utils import process_and_rank_candidates

        if self.item_features_df is None:
            raise ValueError("Item features not loaded. Call load_item_features() first.")
        evaluate_pool_diversity = kwargs.pop('evaluate_pool_diversity', self.evaluate_pool_diversity)
        kwargs.setdefault('popularity_blend_weight', self.popularity_blend_weight)
        kwargs.setdefault('popularity_blend_transform', self.popularity_blend_transform)
        kwargs.setdefault('popularity_blend_normalize', self.popularity_blend_normalize)
        kwargs.setdefault('popularity_model_bias_weight', self.popularity_model_bias_weight)
        _, metrics = process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=False,
            compute_metrics=True,
            metrics_k=metrics_k or self.metrics_k,
            logger=self.logger,
            evaluate_pool_diversity=evaluate_pool_diversity,
            inference_batch_size=self.inference_batch_size,
            **kwargs
        )
        return metrics
