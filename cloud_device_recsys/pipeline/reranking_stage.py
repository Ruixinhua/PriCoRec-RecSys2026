# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Re-ranking Stage Implementation

This module wraps the DeviceReranker model as a pipeline stage.
Supports configurable cloud teacher model for:
  - inject mode: cloud score (logit) injected as numeric feature
  - distill mode: KD loss between student and teacher logits
"""
import torch
import os
import csv
import copy
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager, FeatureGroup
from ..models import build_model as registry_build_model
from ..models import DeviceReranker  # For type hints
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss, compute_kd_loss
from ..data.negative_sampler import NegativeSampler
from ..utils import filter_feature_map

from fuxictr.features import FeatureMap


class RerankingStage(BaseStage):
    """
    Re-ranking stage for final recommendation.

    Runs on device with access to all features including FG3 (private).
    Produces final top-K recommendations.

    Supports optional cloud teacher model via `cloud_teacher_params`:
      - inject: Teacher logits → z-score normalized → cloud_score feature
      - distill: KD loss(student_logit, teacher_logit) added to total loss
    """

    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/reranking",
                 top_k: int = 10,
                 cloud_teacher_params: Optional[Dict[str, Any]] = None,
                 cloud_teacher_feature_map: Optional[FeatureMap] = None,
                 **kwargs):
        """
        Initialize re-ranking stage.

        Args:
            feature_map: FuxiCTR FeatureMap (with FG3 features)
            feature_group_manager: Feature group manager
            model_params: Parameters for DeviceReranker model
            output_dir: Output directory
            top_k: Number of final recommendations
            cloud_teacher_params: Cloud teacher config dict with keys:
                - cloud_teacher_model: model architecture name
                - cloud_teacher_model_params: model hyperparameters
                - cloud_teacher_model_path: path to pre-trained weights (required)
                - cloud_teacher_features: feature groups list (e.g., ["FG1", "FG2"])
                - mode: "inject" or "distill"
                - kd_loss_weight: weight for distillation loss (default 0.1)
                - kd_loss_type: "mse", "kl_div", or "cosine" (default "mse")
                - kd_temperature: temperature for KL-div (default 1.0)
            cloud_teacher_feature_map: Pre-built feature map for teacher model
        """
        # Re-ranking uses ALL feature groups
        self.allowed_feature_groups = kwargs.pop('allowed_feature_groups', [FeatureGroup.FG1, FeatureGroup.FG2, FeatureGroup.FG3])
        super().__init__(
            stage_name="reranking",
            stage_type=StageType.RERANKING,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups = self.allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )
        # Filter feature_map to only include allowed features (FG1, FG2, FG3)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=model_params.get("use_feature_encoder", False))
        self.feature_map.default_emb_dim = model_params['embedding_dim']
        self.use_logit = model_params.get('use_logit', True)
        self.top_k = top_k
        self.model_params = model_params
        self.metrics_k = model_params['metrics_k']
        self.model: Optional[DeviceReranker] = None
        self.monitor = model_params.get('monitor', 'Recall@1')
        self.best_weights_path = None
        # Negative sampling parameters
        self.num_negatives = model_params.get('num_negatives', 0)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax'
        self.margin = model_params.get('margin', 1.0)
        self.negative_sampler: Optional[NegativeSampler] = None

        # Cloud teacher configuration (new config-driven approach)
        self.cloud_teacher_params = cloud_teacher_params or {}
        self.cloud_teacher_feature_map = cloud_teacher_feature_map
        self.cloud_teacher_model = None
        self.cloud_teacher_mode = self.cloud_teacher_params.get('mode', None)  # 'inject', 'distill', 'residual_inject', 'hybrid_inject'

        # Vocab Pruning variables
        self.remap_dicts = None
        self._unmap_tensors = None

        # Legacy cloud score injection (backward compatibility)
        self.use_cloud_score = model_params.get('use_cloud_score', False)
        self.cloud_score_teacher = None  # Pre-ranking model reference (legacy)

        # Determine effective cloud score injection
        # New config takes precedence over legacy use_cloud_score
        if self.cloud_teacher_mode == 'inject':
            self.use_cloud_score = True
        elif self.cloud_teacher_mode == 'residual_inject':
            self.use_cloud_score = False  # cloud_score NOT registered as feature
        elif self.cloud_teacher_mode == 'hybrid_inject':
            self.use_cloud_score = True   # cloud_score IS registered as feature

        # KD parameters (distill mode only)
        self.kd_loss_weight = self.cloud_teacher_params.get('kd_loss_weight', 0.1)
        self.kd_loss_type = self.cloud_teacher_params.get('kd_loss_type', 'mse')
        self.kd_temperature = self.cloud_teacher_params.get('kd_temperature', 1.0)
        self.cloud_score_scale = self.cloud_teacher_params.get('cloud_score_scale', 1.0)
        # Residual inject parameters
        self.residual_weight = self.cloud_teacher_params.get('residual_weight', 1.0)

        # Inference batch size for process_and_rank_candidates
        self.inference_batch_size = model_params.get('inference_batch_size', 50000)
        # Evaluation item features are reserved for valid/test listwise
        # processing. Explicit training negatives use a distinct train-only
        # corpus loaded through load_negative_item_features().
        self.item_features_df = None
        self.train_negative_item_features_df = None

    def _read_item_feature_pool(self, item_pool_path: str, pool_name: str):
        """Read and index one item-feature pool without changing stage state."""
        import pandas as pd
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
        """Load the valid/test item pool used by evaluate() and process()."""
        self.item_features_df = self._read_item_feature_pool(item_pool_path, "evaluation")
        self.logger.info("Loaded %d evaluation items into feature memory.", len(self.item_features_df))

    def load_negative_item_features(self, item_pool_path: str):
        """Load the train-only item pool used exclusively for explicit negatives."""
        self.train_negative_item_features_df = self._read_item_feature_pool(
            item_pool_path, "train-negative"
        )
        self.negative_sampler = None
        self.logger.info(
            "Loaded %d train-negative items into sampling memory.",
            len(self.train_negative_item_features_df),
        )

    def _initialize_negative_sampler(self, item_id_col: str) -> bool:
        """Initialize explicit negatives from the train-only pool, never eval data."""
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
        self.negative_sampler = NegativeSampler(
            train_negative_item_features_df,
            item_id_col=item_id_col,
        )
        self.logger.info(
            "Negative Sampling: %d train-only negatives per positive, loss_type=%s",
            self.num_negatives,
            self.loss_type,
        )
        return True

    def set_cloud_score_teacher(self, teacher_model):
        """Set pre-ranking model to generate cloud scores during training (legacy).

        DEPRECATED: Use cloud_teacher YAML config section instead.

        The teacher model (pre-ranking, FG1+FG2 only) provides cloud_score
        as a numeric feature for the reranking model. This does NOT violate
        privacy: the teacher only uses cloud-available features.

        Note: The teacher keeps its sigmoid activation. We apply torch.logit()
        during injection to recover discriminative raw logits from the
        compressed sigmoid output (~0.9999).
        """
        self.cloud_score_teacher = teacher_model
        self.cloud_score_teacher.eval()
        for param in self.cloud_score_teacher.parameters():
            param.requires_grad = False
        self.logger.info("Cloud score teacher model set (frozen, eval mode) [legacy]")

    def _build_cloud_teacher(self):
        """Build and load the cloud teacher model from YAML config.

        The teacher model is always frozen (pre-trained weights required).
        Used in both 'inject' and 'distill' modes.
        """
        # Update configurations that may have been injected post-instantiation
        self.cloud_teacher_mode = self.cloud_teacher_params.get('mode', self.cloud_teacher_mode)
        self.kd_loss_weight = self.cloud_teacher_params.get('kd_loss_weight', getattr(self, 'kd_loss_weight', 0.1))
        self.kd_loss_type = self.cloud_teacher_params.get('kd_loss_type', getattr(self, 'kd_loss_type', 'mse'))
        self.kd_temperature = self.cloud_teacher_params.get('kd_temperature', getattr(self, 'kd_temperature', 1.0))
        self.cloud_score_scale = self.cloud_teacher_params.get('cloud_score_scale', getattr(self, 'cloud_score_scale', 1.0))
        self.cloud_feature_scale = self.cloud_teacher_params.get('cloud_feature_scale', getattr(self, 'cloud_feature_scale', self.cloud_score_scale))
        self.cloud_residual_scale = self.cloud_teacher_params.get('cloud_residual_scale', getattr(self, 'cloud_residual_scale', self.cloud_score_scale))
        self.cloud_feature_dropout = self.cloud_teacher_params.get('cloud_feature_dropout', getattr(self, 'cloud_feature_dropout', 0.0))
        self.residual_weight = self.cloud_teacher_params.get('residual_weight', getattr(self, 'residual_weight', 1.0))

        # Auto-downgrade hybrid_inject if one of the paths is strictly 0
        if self.cloud_teacher_mode == 'hybrid_inject':
            if self.cloud_feature_scale == 0.0 and self.residual_weight == 0.0:
                self.logger.info("[Hybrid Inject] Both feature_scale and residual_weight are 0. Cloud teacher disabled.")
                self.cloud_teacher_mode = "base"
                self.use_cloud_score = False
            elif self.cloud_feature_scale == 0.0:
                self.logger.info("[Hybrid Inject] feature_scale is 0, auto-downgrading to residual_inject mode.")
                self.cloud_teacher_mode = 'residual_inject'
                self.use_cloud_score = False
            elif self.residual_weight == 0.0:
                self.logger.info("[Hybrid Inject] residual_weight is 0, auto-downgrading to inject mode.")
                self.cloud_teacher_mode = 'inject'

        if self.cloud_teacher_mode:
            self.logger.info(f"Cloud teacher configured: mode={self.cloud_teacher_mode}")
            if self.cloud_teacher_mode == 'distill':
                self.logger.info(f"  KD: loss_type={self.kd_loss_type}, weight={self.kd_loss_weight}, "
                                 f"temperature={self.kd_temperature}")
            elif self.cloud_teacher_mode == 'inject' and self.cloud_feature_scale != 1.0:
                self.logger.info(f"  [Inject Mode] Cloud score will be divided by feature_scale: {self.cloud_feature_scale}")
            elif self.cloud_teacher_mode == 'residual_inject':
                self.logger.info(f"  [Residual Inject] residual_weight={self.residual_weight}, "
                                 f"cloud_residual_scale={self.cloud_residual_scale}")
                self.logger.info(f"  Formula: final_logit = student_logit + {self.residual_weight} * (teacher_logit / {self.cloud_residual_scale})")
            elif self.cloud_teacher_mode == 'hybrid_inject':
                self.logger.info(f"  [Hybrid Inject] residual_weight={self.residual_weight}, "
                                 f"feature_scale={self.cloud_feature_scale}, residual_scale={self.cloud_residual_scale}")
                self.logger.info(f"  Formula: final_logit = student_logit(feature / {self.cloud_feature_scale}) + {self.residual_weight} * (teacher_logit / {self.cloud_residual_scale})")
                if self.cloud_feature_dropout > 0.0:
                    self.logger.info(f"  [Feature Dropout] Active: cloud_feature_dropout={self.cloud_feature_dropout}")

        if not self.cloud_teacher_params or not self.cloud_teacher_mode:
            return

        teacher_model_path = self.cloud_teacher_params.get('cloud_teacher_model_path')
        if not teacher_model_path:
            self.logger.warning("cloud_teacher_model_path not specified. Cloud teacher disabled.")
            self.cloud_teacher_mode = None
            return

        if not os.path.exists(teacher_model_path):
            self.logger.warning(f"Cloud teacher model file not found: {teacher_model_path}. Cloud teacher disabled.")
            self.cloud_teacher_mode = None
            return

        if self.cloud_teacher_feature_map is None:
            self.logger.warning("Cloud teacher feature map not provided. Cloud teacher disabled.")
            self.cloud_teacher_mode = None
            return

        # Merge base model params with teacher-specific params
        teacher_model_name = self.cloud_teacher_params.get('cloud_teacher_model',
                                                            self.model_params.get('model', 'DeviceReranker'))
        teacher_params = copy.deepcopy(self.model_params)
        teacher_specific = self.cloud_teacher_params.get('cloud_teacher_model_params', {})
        teacher_params.update(teacher_specific)
        teacher_params['model'] = teacher_model_name

        # Build teacher model
        teacher_output_dir = os.path.join(self.output_dir, 'cloud_teacher')
        self.cloud_teacher_model = registry_build_model(
            model_name=teacher_model_name,
            feature_map=self.cloud_teacher_feature_map,
            model_params=teacher_params,
            output_dir=teacher_output_dir,
        )

        # Load pre-trained weights
        self.cloud_teacher_model.load_weights(teacher_model_path)
        self.logger.info(f"Loaded cloud teacher weights from: {teacher_model_path}")

        # Freeze teacher (always frozen — pre-trained only)
        for param in self.cloud_teacher_model.parameters():
            param.requires_grad = False
        self.cloud_teacher_model.eval()

        self.logger.info(f"Cloud teacher built: {teacher_model_name} "
                         f"({sum(p.numel() for p in self.cloud_teacher_model.parameters())} params, frozen)")

        # For inject/residual_inject/hybrid_inject mode, set the legacy cloud_score_teacher reference
        if self.cloud_teacher_mode in ('inject', 'residual_inject', 'hybrid_inject'):
            self.cloud_score_teacher = self.cloud_teacher_model

    def build_model(self) -> DeviceReranker:
        """Build and initialize the re-ranking model using unified registry"""

        # Build cloud teacher model FIRST so parameters like mode and use_cloud_score
        # can downgrade and correctly influence the student model's feature map.
        self._build_cloud_teacher()
        # Register cloud_score as numeric feature BEFORE model construction
        if self.use_cloud_score and 'cloud_score' not in self.feature_map.features:
            self.feature_map.features['cloud_score'] = {
                'type': 'numeric',
                'source': '',
            }
            self.feature_map.num_fields = self.feature_map.get_num_fields()
            self.feature_map.set_column_index()
            self.logger.info("Registered 'cloud_score' as numeric feature in feature_map")

        # Ensure output directories exist
        model_dir = os.path.join(self.output_dir, self.feature_map.dataset_id)
        os.makedirs(model_dir, exist_ok=True)

        # Get model name from config, default to DeviceReranker
        model_name = self.model_params.get('model', 'DeviceReranker')

        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
        )

        # Ensure _save_fp16 is set on the model (more robust than relying on feature_map propagation)
        save_fp16 = self.model_params.get('save_fp16', getattr(self.feature_map, '_save_fp16', False))
        if save_fp16:
            self.model._save_fp16 = True
            self.logger.info("[FP16] Model weights will be saved in half-precision (FP16)")

        self.logger.info(f"Built {model_name} model, saving to {model_dir}")

        return self.model

    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              teacher_model: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the re-ranking model with best model monitoring.

        Args:
            train_data: Training data generator (positive examples only for negative sampling)
            valid_data: Validation data generator
            teacher_model: Optional teacher model for distillation (legacy)
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

        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
             self.logger.info("Initializing optimizer...")
             self.model.compile(
                 optimizer=kwargs.get("optimizer", "adam"),
                 loss="binary_crossentropy",
                 lr=kwargs.get("learning_rate", 1e-3)
             )

        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        self.logger.info("Starting re-ranking model training (Custom Loop)")

        epochs = kwargs.get('epochs', 1)
        patience = kwargs.get('patience', 2)
        mode = kwargs.get('mode', 'max')
        metrics = {}

        # Explicit negatives must come from the train-only pool. `item_features_df`
        # remains the evaluation pool used by evaluate() and process().
        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
        use_negative_sampling = self._initialize_negative_sampler(item_id_col)

        # Setup model for manual training
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._epoch_index = 0

        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}, patience={patience}")

        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0

        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')

        # Standard training (Manual Loop) with monitor
        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"*** Epoch {epoch + 1}/{epochs} ***")

            self.model.train()
            total_loss = 0.0
            total_kd_loss = 0.0
            steps = 0

            for batch_data in train_data:
                if use_negative_sampling:
                    loss, kd_loss_val = self._train_step_with_negatives(batch_data, item_id_col)
                else:
                    loss, kd_loss_val = self._train_step_standard(batch_data)
                total_loss += loss.item()
                total_kd_loss += kd_loss_val
                steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0.0
            loss_msg = f"Train Loss: {avg_loss:.6f}"
            if use_negative_sampling:
                loss_msg = f"Train Loss ({self.loss_type}): {avg_loss:.6f}"
            if self.cloud_teacher_mode == 'distill' and total_kd_loss > 0:
                avg_kd = total_kd_loss / steps if steps > 0 else 0.0
                loss_msg += f" | KD Loss: {avg_kd:.6f}"
            self.logger.info(loss_msg)

            if valid_data is not None:
                self.logger.info(f"Evaluating epoch {epoch + 1}...")

                # List-wise metrics (nDCG/Recall)
                valid_metrics = self.evaluate(valid_data)
                metrics.update(valid_metrics)
                self.logger.info(f"Validation (Ranking): {valid_metrics}")

                # Monitor-based best model saving
                curr_val = valid_metrics.get(self.monitor, 0.0)
                is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)

                if is_best:
                    best_metric = curr_val
                    stopping_steps = 0
                    self.model.save_weights(self.best_weights_path)
                    self.logger.info(f"New Best {self.monitor}={curr_val:.6f}! Model Saved.")
                else:
                    stopping_steps += 1
                    self.logger.info(f"No improvement. Patience {stopping_steps}/{patience}")

                    # Decay LR on plateau
                    if kwargs.get("reduce_lr_on_plateau", True):
                        old_lr = self.model.optimizer.param_groups[0]['lr']
                        new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                        self.logger.info(f"Decay LR: {old_lr:.6f} -> {new_lr:.6f}")

                    if stopping_steps >= patience:
                        self.logger.info("Early Stopping.")
                        break

        # Restore best weights
        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Restored best weights from {self.best_weights_path}")

        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            for name, value in sorted(metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])

        return metrics

    def _ensure_unmap_tensors(self, device):
        if hasattr(self, '_unmap_tensors') and self._unmap_tensors is not None:
            return
        self._unmap_tensors = {}
        if not hasattr(self, 'remap_dicts') or not self.remap_dicts:
            return

        for feat, mapping in self.remap_dicts.items():
            if not mapping:
                continue
            max_new_id = max(mapping.values())
            # Initialize with 0 (padding)
            unmap_tensor = torch.zeros(max_new_id + 1, dtype=torch.long, device=device)
            # mapping: old_id -> new_id
            for old_id, new_id in mapping.items():
                unmap_tensor[new_id] = old_id
            self._unmap_tensors[feat] = unmap_tensor

    def _get_teacher_logits(self, batch_dict: dict) -> Optional[torch.Tensor]:
        """Get teacher logits for a batch (used in both inject and distill modes).

        Uses the cloud_teacher_model (new config-driven) or cloud_score_teacher (legacy).
        Returns raw logits after applying torch.logit() to sigmoid output.
        """
        teacher = self.cloud_teacher_model or self.cloud_score_teacher
        if teacher is None:
            return None

        device = next(teacher.parameters()).device if hasattr(teacher, 'parameters') else torch.device('cpu')
        self._ensure_unmap_tensors(device)

        if hasattr(self, '_unmap_tensors') and self._unmap_tensors:
            # self.logger.info("Unmapping compact feature IDs back to original IDs for teacher input...")
            teacher_batch = dict(batch_dict)
            for feat, unmap_tensor in self._unmap_tensors.items():
                if feat in teacher_batch and teacher_batch[feat] is not None:
                    feat_tensor = teacher_batch[feat]
                    if isinstance(feat_tensor, torch.Tensor):
                        # Use torch.clamp to avoid out-of-bounds index for unknown/new compact IDs
                        max_valid = unmap_tensor.size(0) - 1
                        safe_indices = torch.clamp(feat_tensor.long(), min=0, max=max_valid)

                        # Preserve dtype (e.g., int32) if necessary, though FuxiCTR uses long/int internally.
                        original_dtype = feat_tensor.dtype
                        teacher_batch[feat] = unmap_tensor[safe_indices].to(original_dtype)
        else:
            teacher_batch = batch_dict

        with torch.no_grad():
            teacher_out = teacher.forward(teacher_batch)
            logit_score = torch.logit(
                teacher_out['y_pred'].detach().squeeze(-1), eps=1e-7
            )
        return logit_score

    def _inject_cloud_score(self, batch_dict: dict, teacher_logits: torch.Tensor, scale=None):
        """Inject cloud teacher logits as cloud_score into batch_dict (inject mode).

        Uses raw logits (no z-score normalization) to avoid train/eval mismatch:
        training normalizes per mini-batch but evaluation normalizes over all
        candidates, producing inconsistent scales.
        Optionally scales the logits down by `cloud_feature_scale` to prevent numeric explosion.
        """
        if scale is None:
            scale = getattr(self, 'cloud_feature_scale', getattr(self, 'cloud_score_scale', 1.0))

        feature_val = teacher_logits / scale if scale != 1.0 else teacher_logits

        # Prevent Shortcut Learning via Feature Masking (Dropout)
        # Randomly zeroes out the cloud_score feature during training so PNN
        # is forced to learn robust representations instead of collapsing its weights.
        if hasattr(self, 'model') and self.model.training:
            dropout_p = getattr(self, 'cloud_feature_dropout', 0.0)
            if dropout_p > 0.0:
                mask = torch.empty_like(feature_val).bernoulli_(1 - dropout_p)
                feature_val = feature_val * mask / (1 - dropout_p)

        batch_dict['cloud_score'] = feature_val

    def _train_step_standard(self, batch_data) -> Tuple[torch.Tensor, float]:
        """Standard training step (no negative sampling).

        Returns:
            Tuple of (loss_tensor, kd_loss_value_for_logging)
        """
        kd_loss_val = 0.0

        # Inject mode: add cloud_score to batch
        if self.cloud_teacher_mode == 'inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            batch_dict = dict(batch_data)
            teacher_logits = self._get_teacher_logits(batch_dict)
            if teacher_logits is not None:
                self._inject_cloud_score(batch_dict, teacher_logits)
            loss = self.model.train_step(batch_dict)

        # Hybrid inject mode: add cloud_score to batch AND add as residual
        elif self.cloud_teacher_mode == 'hybrid_inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            batch_dict = dict(batch_data)
            teacher_logits = self._get_teacher_logits(batch_dict)
            if teacher_logits is not None:
                self._inject_cloud_score(batch_dict, teacher_logits)

            # Forward pass through student (WITH cloud_score feature)
            self.model.train()
            student_out = self.model.forward(batch_dict)
            student_logit = student_out.get('logit', student_out['y_pred']).squeeze(-1)

            # Add residual: final = student + weight * (teacher / scale)
            scale = getattr(self, 'cloud_residual_scale', getattr(self, 'cloud_score_scale', 1.0))
            weight = getattr(self, 'residual_weight', 1.0)
            if teacher_logits is not None:
                residual = weight * (teacher_logits.detach() / scale)
                combined_logit = student_logit + residual
            else:
                combined_logit = student_logit

            # Task loss (BCE) using combined logit
            y_true = self.model.get_labels(batch_dict)
            combined_pred = self.model.output_activation(combined_logit.unsqueeze(-1))
            task_loss = self.model.loss_fn(combined_pred, y_true, reduction='mean')
            if hasattr(self.model, 'regularization_loss'):
                task_loss = task_loss + self.model.regularization_loss()

            loss = task_loss

            # Backprop manually
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
            self.model.optimizer.step()

        # Distill mode: compute task loss + KD loss
        elif self.cloud_teacher_mode == 'distill' and self.cloud_teacher_model is not None:
            batch_dict = dict(batch_data)
            teacher_logits = self._get_teacher_logits(batch_dict)

            # Forward pass through student
            self.model.train()
            student_out = self.model.forward(batch_dict)
            student_logit = student_out.get('logit', student_out['y_pred']).squeeze(-1)

            # Task loss (BCE)
            y_true = self.model.get_labels(batch_dict)
            task_loss = self.model.loss_fn(student_out['y_pred'], y_true, reduction='mean')
            if hasattr(self.model, 'regularization_loss'):
                task_loss = task_loss + self.model.regularization_loss()

            # KD loss
            target_logits = teacher_logits.detach()
            if getattr(self, 'cloud_score_scale', 1.0) != 1.0:
                target_logits = target_logits / self.cloud_score_scale

            kd_loss = compute_kd_loss(
                student_logit, target_logits,
                kd_loss_type=self.kd_loss_type,
                temperature=self.kd_temperature,
            )
            kd_loss_val = kd_loss.item()

            loss = task_loss + self.kd_loss_weight * kd_loss

            # Diversity loss (if enabled)
            if getattr(self.model, 'use_diversity_loss', False):
                feat_emb_dict = student_out.get('feat_emb_dict')
                if feat_emb_dict is not None:
                    div_loss = self.model.compute_diversity_regularization(
                        feat_emb_dict, student_out['y_pred']
                    )
                    if div_loss is not None:
                        loss = self.model.add_diversity_to_loss(loss, div_loss)

            # Backprop manually
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
            self.model.optimizer.step()

        # Residual inject mode: forward without cloud_score, add teacher logit as residual
        elif self.cloud_teacher_mode == 'residual_inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            batch_dict = dict(batch_data)
            teacher_logits = self._get_teacher_logits(batch_dict)

            # Forward pass through student (no cloud_score feature)
            self.model.train()
            student_out = self.model.forward(batch_dict)
            student_logit = student_out.get('logit', student_out['y_pred']).squeeze(-1)

            # Add residual: final = student + weight * (teacher / scale)
            scale = getattr(self, 'cloud_residual_scale', getattr(self, 'cloud_score_scale', 1.0))
            weight = getattr(self, 'residual_weight', 1.0)
            if teacher_logits is not None:
                residual = weight * (teacher_logits.detach() / scale)
                combined_logit = student_logit + residual
            else:
                combined_logit = student_logit

            # Task loss (BCE) using combined logit
            y_true = self.model.get_labels(batch_dict)
            combined_pred = self.model.output_activation(combined_logit.unsqueeze(-1))
            task_loss = self.model.loss_fn(combined_pred, y_true, reduction='mean')
            if hasattr(self.model, 'regularization_loss'):
                task_loss = task_loss + self.model.regularization_loss()

            loss = task_loss

            # Backprop manually
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
            self.model.optimizer.step()

        # Legacy cloud_score injection (use_cloud_score without cloud_teacher config)
        elif self.use_cloud_score and self.cloud_score_teacher is not None:
            batch_dict = dict(batch_data)
            teacher_logits = self._get_teacher_logits(batch_dict)
            if teacher_logits is not None:
                self._inject_cloud_score(batch_dict, teacher_logits)
            loss = self.model.train_step(batch_dict)

        else:
            loss = self.model.train_step(batch_data)

        return loss, kd_loss_val

    def _train_step_with_negatives(self, batch_data, item_id_col: str) -> Tuple[torch.Tensor, float]:
        """
        Single training step with negative sampling for pairwise ranking.

        Optimized: batches all negatives into a single forward pass instead of
        processing each negative separately.

        Args:
            batch_data: Positive example batch
            item_id_col: Name of item ID column

        Returns:
            Tuple of (loss_tensor, kd_loss_value_for_logging)
        """
        kd_loss_val = 0.0
        batch_dict = dict(batch_data)
        batch_size = len(batch_dict[item_id_col])

        # Get positive item IDs
        pos_item_ids = batch_dict[item_id_col].cpu().numpy()

        # Sample negative items for each positive: [B, num_negatives]
        neg_item_ids = self.negative_sampler.sample_negatives_batch(
            pos_item_ids, self.num_negatives
        )

        # === Build batched negative dict: repeat user features, use negative item features ===
        # Flatten: [B, num_neg] -> [B * num_neg]
        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)

        neg_batch_dict = {}
        for key, val in batch_dict.items():
            if key == item_id_col:
                neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=self.model.device)
            elif key == 'cloud_score':
                continue  # Will compute via teacher below
            elif key in neg_features:
                # Use negative item feature
                val = neg_features[key].to_numpy(copy=False)
                if not np.isscalar(val[0]):
                    val = np.vstack(val)  # Ensure 2D for multi-valued features
                neg_batch_dict[key] = torch.tensor(val, device=self.model.device)
            else:
                # Repeat user features along batch dimension: [B, ...] -> [B * num_neg, ...]
                if hasattr(val, 'to'):
                    val = val.to(self.model.device)
                    neg_batch_dict[key] = val.repeat_interleave(self.num_negatives, dim=0)
                else:
                    neg_batch_dict[key] = val

        # === Cloud Score Injection (positive + negative) ===
        # Uses raw logits — no z-score normalization to avoid train/eval mismatch.
        pos_teacher_logit = None
        neg_teacher_logit = None
        if self.cloud_teacher_mode == 'inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            pos_logit = self._get_teacher_logits(batch_dict)
            neg_logit = self._get_teacher_logits(neg_batch_dict)
            if pos_logit is not None and neg_logit is not None:
                self._inject_cloud_score(batch_dict, pos_logit)
                self._inject_cloud_score(neg_batch_dict, neg_logit)
        elif self.cloud_teacher_mode == 'residual_inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            # Pre-compute teacher logits but do NOT inject into batch
            pos_teacher_logit = self._get_teacher_logits(batch_dict)
            neg_teacher_logit = self._get_teacher_logits(neg_batch_dict)
        elif self.cloud_teacher_mode == 'hybrid_inject' and (self.cloud_teacher_model or self.cloud_score_teacher):
            # Pre-compute teacher logits AND inject into batch
            pos_teacher_logit = self._get_teacher_logits(batch_dict)
            neg_teacher_logit = self._get_teacher_logits(neg_batch_dict)
            if pos_teacher_logit is not None and neg_teacher_logit is not None:
                self._inject_cloud_score(batch_dict, pos_teacher_logit)
                self._inject_cloud_score(neg_batch_dict, neg_teacher_logit)
        elif self.use_cloud_score and self.cloud_score_teacher is not None:
            # Legacy path
            pos_logit = self._get_teacher_logits(batch_dict)
            neg_logit = self._get_teacher_logits(neg_batch_dict)
            if pos_logit is not None and neg_logit is not None:
                self._inject_cloud_score(batch_dict, pos_logit)
                self._inject_cloud_score(neg_batch_dict, neg_logit)

        # Forward passes
        pos_output = self.model.forward(batch_dict)

        neg_output = self.model.forward(neg_batch_dict)
        if self.use_logit:
            pos_scores = pos_output.get('logit', pos_output['y_pred'])  # [B, 1]
            neg_scores_flat = neg_output.get('logit', neg_output['y_pred'])  # [B * num_neg, 1]
        else:
            pos_scores = pos_output['y_pred']
            neg_scores_flat = neg_output['y_pred']  # [B * num_neg, 1]

        # === Residual Inject / Hybrid Inject: add teacher logit as residual to scores ===
        if self.cloud_teacher_mode in ('residual_inject', 'hybrid_inject') and pos_teacher_logit is not None and neg_teacher_logit is not None:
            scale = getattr(self, 'cloud_residual_scale', getattr(self, 'cloud_score_scale', 1.0))
            weight = getattr(self, 'residual_weight', 1.0)
            pos_residual = weight * (pos_teacher_logit.detach() / scale)
            neg_residual = weight * (neg_teacher_logit.detach() / scale)
            pos_scores = pos_scores + pos_residual.view_as(pos_scores)
            neg_scores_flat = neg_scores_flat + neg_residual.view_as(neg_scores_flat)

        # Reshape back: [B * num_neg, 1] -> [B, num_neg]
        neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)

        # Compute pairwise ranking loss
        if self.loss_type == 'bpr':
            loss = bpr_loss(pos_scores, neg_scores)
        elif self.loss_type == 'margin':
            loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
        elif self.loss_type == 'softmax':
            loss = softmax_cross_entropy_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # === Distillation loss (distill mode with negatives) ===
        if self.cloud_teacher_mode == 'distill' and self.cloud_teacher_model is not None:
            teacher_pos_logit = self._get_teacher_logits(batch_dict)
            if teacher_pos_logit is not None:
                student_pos_logit = pos_output.get('logit', pos_output['y_pred']).squeeze(-1)
                target_logits = teacher_pos_logit.detach()
                if getattr(self, 'cloud_score_scale', 1.0) != 1.0:
                    target_logits = target_logits / self.cloud_score_scale

                kd_loss = compute_kd_loss(
                    student_pos_logit, target_logits,
                    kd_loss_type=self.kd_loss_type,
                    temperature=self.kd_temperature,
                )
                kd_loss_val = kd_loss.item()
                loss = loss + self.kd_loss_weight * kd_loss

        # Add diversity loss if enabled (only on positive samples for recommendation diversity)
        if getattr(self.model, 'use_diversity_loss', False):
            # Get embeddings from positive output (must use pos_output, not _last_feat_emb_dict
            # which gets overwritten by negative forward pass)
            pos_feat_emb_dict = pos_output.get('feat_emb_dict')
            if pos_feat_emb_dict is not None:
                div_loss = self.model.compute_diversity_regularization(
                    pos_feat_emb_dict, pos_scores
                )
                if div_loss is not None:
                    loss = self.model.add_diversity_to_loss(loss, div_loss)

        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            loss = loss + self.model.regularization_loss()

        # Backprop
        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
        self.model.optimizer.step()

        return loss, kd_loss_val

    def process(self,
                input_data: StageOutput,
                **kwargs) -> Tuple[StageOutput, Dict[str, float]]:
        """
        Re-rank candidates to produce final recommendations - scores and selects top-K.

        Args:
            input_data: StageOutput from previous stage containing candidate sets
            **kwargs: Additional parameters (e.g., top_k override, compute_metrics)

        Returns:
            Tuple of (StageOutput with re-ranked Top-K candidates, metrics dict)
        """
        from ..metric_utils import process_and_rank_candidates

        if os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)
            self.logger.info(f"Loaded best weights from {self.best_weights_path}")
        else:
            self.logger.warning(f"No best weights found at {self.best_weights_path}. Using current model state.")

        # Ensure item features are loaded
        if self.item_features_df is None:
            self.logger.error("Item features not loaded. Call load_item_features() first.")
            return StageOutput(stage_name=self.stage_name), {}

        # Ensure _unmap_tensors are ready if using cloud teacher
        cloud_teacher = self.cloud_teacher_model or self.cloud_score_teacher
        unmap_tensors = None
        is_residual_inject = self.cloud_teacher_mode in ('residual_inject', 'hybrid_inject')
        if cloud_teacher is not None and (self.use_cloud_score or is_residual_inject):
            device = next(cloud_teacher.parameters()).device
            self._ensure_unmap_tensors(device)
            unmap_tensors = getattr(self, '_unmap_tensors', None)

        return process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=input_data,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=True,
            compute_metrics=kwargs.pop('compute_metrics', True),
            top_k=kwargs.get('top_k', self.top_k),
            logger=self.logger,
            metrics_k=self.metrics_k,
            inject_cloud_score=self.use_cloud_score,
            ranking_candidates_df=kwargs.pop('preranking_candidates_df', None),
            inference_batch_size=self.inference_batch_size,
            cloud_teacher_model=cloud_teacher,
            cloud_teacher_unmap_tensors=unmap_tensors,
            cloud_score_scale=getattr(self, 'cloud_score_scale', 1.0),
            residual_inject=is_residual_inject,
            residual_weight=getattr(self, 'residual_weight', 1.0),
            **kwargs
        )

    def evaluate(self,
                 input_data: StageOutput,
                 metrics_k: List[int] = None,
                 retrieval_output: Optional[StageOutput] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate re-ranking model with list-wise metrics (nDCG, Recall).

        Args:
            input_data: StageOutput from previous stage (preranking), containing
                        candidate sets with labels (e.g. top-100 candidates).
                        Used for Recall@K/nDCG@K computation.
            metrics_k: List of K values for Recall@K and nDCG@K
            retrieval_output: Optional StageOutput with retrieval candidates (e.g. top-1000).
                              When provided, model scores ALL retrieval candidates and
                              gAUC/MRR are computed on this larger pool to align with
                              preranking evaluation scope. Recall@K/nDCG@K are still
                              restricted to the input_data (preranking) subset.
                              When absent, all metrics use input_data only.
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation metrics
        """
        from ..metric_utils import process_and_rank_candidates

        if self.item_features_df is None:
            self.logger.error("Item features not loaded. Call load_item_features() first.")
            return {}

        # When retrieval_output is provided:
        #   - scoring_input = retrieval_output (score ALL ~1000 retrieval candidates)
        #   - ranking_candidates_df = input_data (restrict Recall/nDCG to preranking ~100)
        #   - gAUC/MRR use the full retrieval pool (via full_pool_scores in compute_ranking_metrics)
        # When absent:
        #   - scoring_input = input_data (preranking candidates only)
        #   - ranking_candidates_df = None (no filtering)
        ranking_candidates_df = None
        if retrieval_output is not None:
            scoring_input = retrieval_output
            ranking_candidates_df = input_data.candidates_df
            self.logger.info(f"Fair eval: scoring on {scoring_input.get_total_candidates()} retrieval candidates, "
                             f"ranking restricted to {len(ranking_candidates_df)} preranking candidates")
        else:
            scoring_input = input_data

        # Ensure _unmap_tensors are ready if using cloud teacher
        cloud_teacher = self.cloud_teacher_model or self.cloud_score_teacher
        unmap_tensors = None
        is_residual_inject = self.cloud_teacher_mode in ('residual_inject', 'hybrid_inject')
        if cloud_teacher is not None and (self.use_cloud_score or is_residual_inject):
            device = next(cloud_teacher.parameters()).device
            self._ensure_unmap_tensors(device)
            unmap_tensors = getattr(self, '_unmap_tensors', None)

        _, metrics = process_and_rank_candidates(
            model=self.model,
            feature_map=self.feature_map,
            input_data=scoring_input,
            item_features_df=self.item_features_df,
            stage_name=self.stage_name,
            return_output=False,
            compute_metrics=True,
            metrics_k=metrics_k or self.metrics_k,
            logger=self.logger,
            inject_cloud_score=self.use_cloud_score,
            ranking_candidates_df=ranking_candidates_df,
            inference_batch_size=self.inference_batch_size,
            cloud_teacher_model=cloud_teacher,
            cloud_teacher_unmap_tensors=unmap_tensors,
            cloud_score_scale=getattr(self, 'cloud_score_scale', 1.0),
            residual_inject=is_residual_inject,
            residual_weight=getattr(self, 'residual_weight', 1.0),
            **kwargs
        )
        return metrics
