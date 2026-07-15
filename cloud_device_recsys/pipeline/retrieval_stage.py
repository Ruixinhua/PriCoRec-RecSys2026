# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Retrieval Stage Implementation

This module wraps the DualTowerRetrieval model as a pipeline stage.
"""

import os
import sys
import csv
import copy
import time
import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Optional, Any
import torch
from tqdm import tqdm

from ..pipeline.base_stage import BaseStage, StageType
from ..pipeline.stage_output import StageOutput
from ..config.feature_groups import FeatureGroupManager
from ..models import build_model as registry_build_model
from ..models import DualTowerRetrieval  # For type hints
from ..models.losses import bpr_loss, margin_ranking_loss, softmax_cross_entropy_loss, compute_diversity_for_pairwise
from ..data.negative_sampler import NegativeSampler
from ..utils import filter_feature_map

from fuxictr.features import FeatureMap


class RetrievalStage(BaseStage):
    """
    Retrieval stage for candidate generation.

    Uses dual-tower model to retrieve top-K candidates from item pool.
    Only uses FG1 (non-personalized) and FG2 (cloud-personalized) features.
    """

    def __init__(self,
                 feature_map: FeatureMap,
                 feature_group_manager: FeatureGroupManager,
                 allowed_feature_groups,
                 model_params: Dict[str, Any],
                 output_dir: str = "./outputs/retrieval",
                 top_k: int = 1000,
                 **kwargs):
        """
        Initialize retrieval stage.

        Args:
            feature_map: FuxiCTR FeatureMap
            feature_group_manager: Feature group manager
            allowed_feature_groups: allowed feature groups
            model_params: Parameters for DualTowerRetrieval model
            output_dir: Output directory
            top_k: Number of candidates to retrieve
        """
        super().__init__(
            stage_name="retrieval",
            stage_type=StageType.RETRIEVAL,
            feature_group_manager=feature_group_manager,
            allowed_feature_groups=allowed_feature_groups,
            output_dir=output_dir,
            **kwargs
        )
        # Ensure output directory exists (BaseModel expects it)
        os.makedirs(os.path.join(output_dir, feature_map.dataset_id), exist_ok=True)

        self.full_feature_map = copy.deepcopy(feature_map)
        # Filter feature_map to only include allowed features
        # This fixes the issue where device features (FG3) were included in User Tower
        use_feature_encoder = model_params.get("use_feature_encoder", False)
        self.feature_map = filter_feature_map(feature_map, feature_group_manager, self.allowed_feature_groups,
                                              use_feature_encoder=use_feature_encoder)
        self.logger.info(f"Use feature encoder: {use_feature_encoder}")
        self.top_k = top_k
        self.model_params = model_params
        self.model: Optional[DualTowerRetrieval] = None
        self.best_weights_path = None
        self.metrics_k = model_params['metrics_k']
        self.monitor = model_params.get('monitor', 'Recall@1000')
        self.chunk_size = model_params.get('chunk_size', 50000)
        self.progress_interval = int(
            model_params.get(
                'progress_interval',
                model_params.get('train_progress_interval', 100),
            )
            or 0
        )
        # Negative sampling parameters
        self.num_negatives = model_params.get('num_negatives', 0)
        self.loss_type = model_params.get('loss_type', 'bpr')  # 'bpr', 'margin', 'softmax', 'sampled_softmax'
        self.margin = model_params.get('margin', 1.0)
        self.use_in_batch_negatives = model_params.get('use_in_batch_negatives', False)
        self.use_diversity_loss = model_params.get('use_diversity_loss', False)
        self.negative_sampler: Optional[NegativeSampler] = None
        # Item index for retrieval
        self.item_embeddings: Optional[torch.Tensor] = None
        self.item_ids: Optional[torch.Tensor] = None
        self.item_id_to_idx: Optional[Dict] = None
        # Popularity baseline cache aligned with self.item_ids
        self.item_popularity: Optional[np.ndarray] = None
        default_pop_metrics = model_params.get('popularity_metrics_k', [100])
        self.popularity_metrics_k = sorted({
            int(k) for k in default_pop_metrics
            if isinstance(k, (int, float)) and int(k) > 0
        })

    def build_model(self) -> DualTowerRetrieval:
        """Build and initialize the retrieval model using unified registry"""
        # Get model name from config, default to DualTowerRetrieval
        model_name = self.model_params.get('model', 'DualTowerRetrieval')

        self.model = registry_build_model(
            model_name=model_name,
            feature_map=self.feature_map,
            model_params=self.model_params,
            output_dir=self.output_dir,
        )
        self.logger.info(f"Built {model_name} model")
        return self.model

    def train(self, train_data, valid_data=None, item_data=None, item_features_df=None, **kwargs):
        """
        Train loop with validation.

        Args:
            train_data: Training data generator (positive examples only)
            valid_data: Validation data generator
            item_data: Item data generator (for building item index during validation)
            item_features_df: DataFrame with item features (for negative sampling)
            **kwargs: Additional training parameters
        """
        if self.model is None:
            self.build_model()
        self.best_weights_path = os.path.join(self.model.model_dir, self.model.model_id + ".model")
        epochs = kwargs.get("epochs", 1)
        patience = kwargs.get("patience", 2)
        mode = kwargs.get("mode", "max")

        # Initialize negative sampler if using negative sampling
        use_negative_sampling = self.num_negatives > 0 and item_features_df is not None
        use_pairwise_training = use_negative_sampling or self.use_in_batch_negatives
        if use_negative_sampling:
            item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')
            self.negative_sampler = NegativeSampler(item_features_df, item_id_col=item_id_col)
            self.logger.info(f"Negative Sampling: {self.num_negatives} negatives per positive, loss_type={self.loss_type}")
        if self.use_in_batch_negatives:
            self.logger.info("In-batch negatives enabled (cross-entropy over batch)")
        if not use_negative_sampling and self.num_negatives > 0 and item_features_df is None:
            self.logger.warning("num_negatives > 0 but item_features_df not provided. Using standard training.")

        self.logger.info(f"Start Training: epochs={epochs}, monitor={self.monitor}")

        best_metric = -np.inf if mode == "max" else np.inf
        stopping_steps = 0

        # Setup model for manual training
        self.model._total_steps = 0
        self.model._stop_training = False
        self.model._max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)
        self.model._verbose = kwargs.get("verbose", 1)
        self.model._eval_steps = 1e9
        self.progress_interval = int(kwargs.get("progress_interval", self.progress_interval) or 0)
        self.logger.info(
            "Retrieval model actual device: %s; progress_interval=%d batches",
            next(self.model.parameters()).device,
            self.progress_interval,
        )

        for epoch in range(epochs):
            self.model._epoch_index = epoch
            self.logger.info(f"*** Epoch {epoch+1} ***")

            # Choose training method based on negative sampling
            if use_pairwise_training:
                self.train_epoch_with_negatives(train_data)
            else:
                self.train_epoch(train_data)

            # Validation
            self.logger.info("Building Item Index...")
            self.build_item_index(item_data)

            metrics = self.evaluate(valid_data)
            curr_val = metrics.get(self.monitor, 0.0)
            is_best = (curr_val > best_metric) if mode == "max" else (curr_val < best_metric)

            if is_best:
                best_metric = curr_val
                stopping_steps = 0
                self.model.save_weights(self.best_weights_path)
                self.logger.info(f"New Best {self.monitor}! Model Saved.")
            else:
                stopping_steps += 1
                self.logger.info(f"No improve. Patience {stopping_steps}/{patience}")

                # Decay LR on plateau
                if kwargs.get("reduce_lr_on_plateau", True):
                    old_lr = self.model.optimizer.param_groups[0]['lr']
                    new_lr = self.model.lr_decay(factor=kwargs.get("lr_decay_factor", 0.1))
                    self.logger.info(f"Decay LR: {old_lr:.6f} -> {new_lr:.6f}")

                if stopping_steps >= patience:
                    self.logger.info("Early Stopping.")
                    break

        # Restore best
        if os.path.exists(self.best_weights_path):
             self.model.load_weights(self.best_weights_path)

    @staticmethod
    def _batch_size_from_batch(batch_data):
        for value in batch_data.values():
            if hasattr(value, "size"):
                return int(value.size(0))
            try:
                return len(value)
            except TypeError:
                continue
        return 0

    def _expected_batch_count(self, data_generator):
        try:
            expected_batches = len(data_generator)
        except TypeError:
            return "unknown"
        return expected_batches if expected_batches and expected_batches > 0 else "unknown"

    def _log_epoch_progress(
            self,
            total_batches,
            total_examples,
            train_loss,
            epoch_start_time,
            last_log_time,
            last_log_examples,
            expected_batches,
            loss_name="avg_loss"):
        now = time.time()
        elapsed = max(now - epoch_start_time, 1e-9)
        interval_elapsed = max(now - last_log_time, 1e-9)
        recent_examples = total_examples - last_log_examples
        recent_rps = recent_examples / interval_elapsed
        overall_rps = total_examples / elapsed
        avg_loss = train_loss / total_batches if total_batches > 0 else 0.0
        self.logger.info(
            "Retrieval epoch %d progress: batch=%d/%s, samples=%d, %s=%.6f, "
            "records/s recent=%.1f overall=%.1f, elapsed=%.1fs, device=%s",
            self.model._epoch_index + 1,
            total_batches,
            expected_batches,
            total_examples,
            loss_name,
            avg_loss,
            recent_rps,
            overall_rps,
            elapsed,
            next(self.model.parameters()).device,
        )
        return now, total_examples

    def train_epoch(self, data_generator):
        """
        Train the model for one epoch (standard training with BCE loss).
        Reference: fuxictr/pytorch/models/rank_model.py
        """
        self.model.train()
        train_loss = 0
        total_batches = 0
        total_examples = 0
        expected_batches = self._expected_batch_count(data_generator)
        epoch_start_time = time.time()
        last_log_time = epoch_start_time
        last_log_examples = 0
        if self.progress_interval > 0:
            self.logger.info(
                "Retrieval epoch %d started: expected_batches=%s, progress_interval=%d, device=%s",
                self.model._epoch_index + 1,
                expected_batches,
                self.progress_interval,
                next(self.model.parameters()).device,
            )

        if self.model._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=True, file=sys.stdout)

        for batch_index, batch_data in enumerate(batch_iterator):
            self.model._batch_index = batch_index
            self.model._total_steps += 1

            loss = self.model.train_step(batch_data)
            train_loss += loss.item()
            total_batches += 1
            total_examples += self._batch_size_from_batch(batch_data)
            if self.progress_interval > 0 and total_batches % self.progress_interval == 0:
                last_log_time, last_log_examples = self._log_epoch_progress(
                    total_batches=total_batches,
                    total_examples=total_examples,
                    train_loss=train_loss,
                    epoch_start_time=epoch_start_time,
                    last_log_time=last_log_time,
                    last_log_examples=last_log_examples,
                    expected_batches=expected_batches,
                )
            if self.model._stop_training:
                break

        if total_batches > 0:
            avg_loss = train_loss / total_batches
            reg_loss = self.model.regularization_loss() if hasattr(self.model, 'regularization_loss') else 0
            self.logger.info(f"Train loss: {avg_loss:.6f} (Reg Loss: {reg_loss:.6f})")

    def train_epoch_with_negatives(self, data_generator):
        """
        Train the model for one epoch with negative sampling and pairwise ranking loss.

        Optimized: batches all negative embeddings into a single call.
        """
        self.model.train()
        train_loss = 0
        total_batches = 0
        total_examples = 0
        expected_batches = self._expected_batch_count(data_generator)
        epoch_start_time = time.time()
        last_log_time = epoch_start_time
        last_log_examples = 0
        if self.progress_interval > 0:
            self.logger.info(
                "Retrieval epoch %d started: expected_batches=%s, progress_interval=%d, device=%s, loss_type=%s",
                self.model._epoch_index + 1,
                expected_batches,
                self.progress_interval,
                next(self.model.parameters()).device,
                self.loss_type,
            )

        if self.negative_sampler is None and self.num_negatives > 0:
            raise ValueError("Negative sampler not initialized. Call train() with item_features_df.")

        item_id_col = getattr(self.feature_map, 'dataset_config', {}).get('item_id_col', 'cand_item_id')

        if self.model._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=True, file=sys.stdout)

        for batch_index, batch_data in enumerate(batch_iterator):
            self.model._batch_index = batch_index
            self.model._total_steps += 1

            batch_dict = dict(batch_data)
            batch_size = len(batch_dict[item_id_col])

            # Get positive item IDs
            pos_item_ids = batch_dict[item_id_col].cpu().numpy()

            # Sample negative items for each positive: [B, num_negatives]
            if self.num_negatives > 0 and self.negative_sampler is not None:
                neg_item_ids = self.negative_sampler.sample_negatives_batch(
                    pos_item_ids, self.num_negatives
                )
            else:
                neg_item_ids = None

            # Compute positive embeddings
            pos_user_emb = self.model.get_user_embedding(batch_data)  # [B, D]
            pos_item_emb = self.model.get_item_embedding(batch_data)  # [B, D]
            # Use raw logits (no sigmoid) for pairwise losses to avoid double activation
            pos_scores = self.model.cal_similarity_raw(pos_user_emb, pos_item_emb)  # [B, 1]

            # === In-batch negatives: use other items in the batch as negatives ===
            if self.use_in_batch_negatives:
                # user_emb [B, D] x item_emb [B, D]^T -> [B, B] all-pairs scores
                in_batch_scores = torch.matmul(pos_user_emb, pos_item_emb.t()) / self.model.temperature  # [B, B]
                # Positive scores are on the diagonal
                # Use cross-entropy: target is diagonal (index i for row i)
                in_batch_targets = torch.arange(batch_size, device=self.model.device)
                in_batch_loss = torch.nn.functional.cross_entropy(in_batch_scores, in_batch_targets)

            # === Explicit negatives (if num_negatives > 0) ===
            if self.num_negatives > 0 and neg_item_ids is not None:
                # Flatten: [B, num_neg] -> [B * num_neg]
                neg_ids_flat = neg_item_ids.reshape(-1)

                # Get features for all negatives at once
                neg_features = self.negative_sampler.get_features_by_ids(neg_ids_flat)

                # Build batched negative dict
                neg_batch_dict = {}
                for key, val in batch_dict.items():
                    if key == item_id_col:
                        neg_batch_dict[key] = torch.tensor(neg_ids_flat, device=self.model.device)
                    elif key in neg_features.columns:
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

                # Single call for all negative embeddings: [B * num_neg, D]
                neg_item_emb_flat = self.model.get_item_embedding(neg_batch_dict)

                # Repeat user embedding and compute all similarities at once (raw logits)
                pos_user_emb_repeated = pos_user_emb.repeat_interleave(self.num_negatives, dim=0)
                neg_scores_flat = self.model.cal_similarity_raw(pos_user_emb_repeated, neg_item_emb_flat)

                # Reshape: [B * num_neg, 1] -> [B, num_neg]
                neg_scores = neg_scores_flat.view(batch_size, self.num_negatives)
            else:
                neg_scores = None

            # Compute pairwise ranking loss
            loss = torch.tensor(0.0, device=self.model.device)

            # Explicit negative loss (if we have sampled negatives)
            if neg_scores is not None:
                if self.loss_type == 'bpr':
                    loss = loss + bpr_loss(pos_scores, neg_scores)
                elif self.loss_type == 'margin':
                    loss = loss + margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
                elif self.loss_type in ('softmax', 'sampled_softmax'):
                    loss = loss + softmax_cross_entropy_loss(pos_scores, neg_scores)
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # In-batch negative loss (always cross-entropy over the batch)
            if self.use_in_batch_negatives:
                if neg_scores is not None:
                    # Combine: weight explicit + in-batch equally
                    loss = 0.5 * loss + 0.5 * in_batch_loss
                else:
                    loss = in_batch_loss

            # Add diversity loss if enabled (works for both wrapper and mixin models)
            if self.use_diversity_loss:
                diversity_delta = compute_diversity_for_pairwise(self.model, batch_data, pos_scores)
                loss = loss + diversity_delta

            # Add regularization
            if hasattr(self.model, 'regularization_loss'):
                loss = loss + self.model.regularization_loss()

            # Backprop
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model._max_gradient_norm)
            self.model.optimizer.step()

            train_loss += loss.item()
            total_batches += 1
            total_examples += batch_size
            if self.progress_interval > 0 and total_batches % self.progress_interval == 0:
                last_log_time, last_log_examples = self._log_epoch_progress(
                    total_batches=total_batches,
                    total_examples=total_examples,
                    train_loss=train_loss,
                    epoch_start_time=epoch_start_time,
                    last_log_time=last_log_time,
                    last_log_examples=last_log_examples,
                    expected_batches=expected_batches,
                    loss_name=f"{self.loss_type}_loss",
                )

            if self.model._stop_training:
                break

        if total_batches > 0:
            avg_loss = train_loss / total_batches
            self.logger.info(f"Train loss ({self.loss_type}): {avg_loss:.6f}")

    def build_item_index(self, item_data):
        """Build item embeddings index from iterator"""
        self.model.eval()
        device = next(self.model.parameters()).device
        emb_list, id_list = [], []
        # Get IDs if available, else sequential
        config = getattr(self.feature_map, 'dataset_config', {})
        item_id_col = config.get('item_id_col', 'cand_item_id')
        with torch.no_grad():
            for batch in tqdm(item_data, disable=True, file=sys.stdout):
                # Get embeddings using model helper
                embs = self.model.get_item_embedding(batch)
                emb_list.append(embs.detach())
                id_list.append(batch[item_id_col].cpu())
        self.item_embeddings = torch.vstack(emb_list).to(device)
        self.item_ids = torch.cat(id_list)
        self.item_id_to_idx = {item_id.item(): i for i, item_id in enumerate(self.item_ids)}

        self.logger.info(f"Index Built: {self.item_embeddings.shape} items on {self.item_embeddings.device}")

    def _ensure_item_popularity(self) -> bool:
        """
        Load train-split item popularity aligned with the retrieval item index.

        The resulting ``self.item_popularity`` array shares the same order as
        ``self.item_ids`` so chunk-time popularity lookups stay vectorized.
        """
        if self.item_popularity is not None:
            return True
        if self.item_ids is None:
            self.logger.warning("Cannot build popularity cache before item index exists.")
            return False

        dataset_config = getattr(self.feature_map, 'dataset_config', {}) or {}
        train_path = self._resolve_train_path_for_popularity(dataset_config)
        item_id_col = dataset_config.get('item_id_col', 'cand_item_id')
        data_format = str(dataset_config.get(
            'processed_data_format',
            dataset_config.get('data_format', 'parquet')
        )).lower()
        is_tfrecord = data_format in {'tfrecord', 'tf_record'} or str(train_path).endswith(('.tfrecord', '.tfrecord.gz'))

        if not train_path:
            self.logger.warning(
                f"Popularity metrics disabled: train split not found at {train_path!r}."
            )
            return False
        if not is_tfrecord and not os.path.exists(train_path):
            self.logger.warning(
                f"Popularity metrics disabled: train split not found at {train_path!r}."
            )
            return False

        if is_tfrecord:
            pop_dict = self._load_tfrecord_item_popularity(train_path, item_id_col, dataset_config)
            if pop_dict is None:
                return False
        elif data_format == 'csv' or str(train_path).endswith('.csv'):
            try:
                counts = pd.read_csv(train_path, usecols=[item_id_col])[item_id_col].value_counts()
                pop_dict = counts.to_dict()
            except Exception as ex:
                self.logger.warning(
                    f"Popularity metrics disabled: failed to load {train_path} ({ex})."
                )
                return False
        else:
            try:
                import polars as pl

                pop_df = (
                    pl.scan_parquet(train_path)
                    .group_by(item_id_col)
                    .agg(pl.len().alias('popularity'))
                    .collect()
                )
                pop_dict = dict(zip(pop_df[item_id_col].to_list(), pop_df['popularity'].to_list()))
            except Exception as fallback_ex:
                self.logger.warning(
                    f"Polars popularity loading failed ({fallback_ex}); falling back to pandas."
                )
                try:
                    counts = pd.read_parquet(train_path, columns=[item_id_col])[item_id_col].value_counts()
                    pop_dict = counts.to_dict()
                except Exception as pandas_ex:
                    self.logger.warning(
                        f"Popularity metrics disabled: failed to load {train_path} ({pandas_ex})."
                    )
                    return False

        item_ids_np = self.item_ids.cpu().numpy()
        self.item_popularity = np.fromiter(
            (float(pop_dict.get(item_id.item() if hasattr(item_id, 'item') else item_id, 0.0))
             for item_id in item_ids_np),
            dtype=np.float32,
            count=len(item_ids_np)
        )
        self.logger.info(
            "Loaded train popularity for %d indexed items from %s",
            len(self.item_popularity),
            train_path,
        )
        return True

    def _resolve_train_path_for_popularity(self, dataset_config: Dict[str, Any]) -> Optional[str]:
        """Resolve the train split path using the same precedence as pipeline data loading."""
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

    def _load_tfrecord_item_popularity(
        self,
        train_path: str,
        item_id_col: str,
        dataset_config: Dict[str, Any],
    ) -> Optional[Dict[Any, int]]:
        """Count item frequencies from encoded TFRecord data."""
        try:
            from collections import OrderedDict
            from fuxictr.pytorch.dataloaders.tfrecord_dataloader import TFRecordDataLoader

            item_feature_map = copy.deepcopy(self.full_feature_map)
            if item_id_col not in item_feature_map.features:
                self.logger.warning(
                    f"Popularity metrics disabled: item id column {item_id_col!r} not found in feature_map."
                )
                return None
            item_feature_map.features = OrderedDict([
                (item_id_col, item_feature_map.features[item_id_col])
            ])
            item_feature_map.labels = []
            item_feature_map.dataset_config = dataset_config
            item_feature_map.set_column_index()

            loader_conf = dict(dataset_config.get('tfrecord_load_conf', {}) or {})
            loader_conf['count_samples'] = False
            batch_size = int(loader_conf.pop('popularity_batch_size', 8192))
            loader = TFRecordDataLoader(
                feature_map=item_feature_map,
                data_path=train_path,
                split='popularity',
                batch_size=batch_size,
                shuffle=False,
                tfrecord_load_conf=loader_conf,
            )

            pop_dict: Dict[Any, int] = {}
            for batch in loader:
                values, counts = np.unique(batch[item_id_col].cpu().numpy(), return_counts=True)
                for value, count in zip(values.tolist(), counts.tolist()):
                    pop_dict[value] = pop_dict.get(value, 0) + int(count)
            return pop_dict
        except Exception as ex:
            self.logger.warning(
                f"Popularity metrics disabled: failed to load TFRecord train split {train_path} ({ex})."
            )
            return None

    def _retrieve_and_score(
        self,
        input_data: Any,
        item_data: Any = None,
        return_output: bool = True,
        compute_metrics: bool = False,
        metrics_k: List[int] = None,
        **kwargs
    ) -> tuple:
        """
        Unified retrieval and scoring method.

        This is the core method that handles both candidate generation (process)
        and metrics computation (evaluate) in a single pass.

        Args:
            input_data: FuxiCTR DataGenerator with user queries
            item_data: Item data for building index (if needed)
            return_output: Whether to return StageOutput with candidates
            compute_metrics: Whether to compute Recall@K metrics
            metrics_k: K values for Recall@K computation
            **kwargs: Additional parameters (chunk_size, etc.)

        Returns:
            Tuple of (StageOutput or None, metrics dict or None)
        """
        # 1. Ensure model is built and weights loaded
        if self.model is None:
            raise RuntimeError("No model found. Train model first!")

        # 2. Build item index if needed
        if self.item_embeddings is None or self.item_ids is None:
            if item_data is None:
                raise ValueError("item_data must be provided to build item index.")
            self.logger.info("Building item index...")
            self.build_item_index(item_data)

        if self.item_id_to_idx is None:
            self.item_id_to_idx = {item_id.item(): i for i, item_id in enumerate(self.item_ids)}

        self.model.eval()
        metrics_k = self.metrics_k if metrics_k is None else metrics_k
        total_recall = {k: 0.0 for k in metrics_k} if compute_metrics else None
        popularity_metrics_k = []
        total_popularity_recall = None
        if compute_metrics and return_output:
            popularity_metrics_k = sorted({
                k for k in self.popularity_metrics_k + [k for k in metrics_k if k <= self.top_k]
                if k > 0
            })
            if popularity_metrics_k and self._ensure_item_popularity():
                total_popularity_recall = {k: 0.0 for k in popularity_metrics_k}
            elif popularity_metrics_k:
                self.logger.warning("Popularity recall metrics requested but popularity cache is unavailable.")
        model_device = next(self.model.parameters()).device

        # Setup output
        output = StageOutput(stage_name=self.stage_name) if return_output else None

        dataset_config = getattr(self.feature_map, 'dataset_config', {})
        item_id_col = dataset_config.get('item_id_col', 'cand_item_id')
        impression_id_col = dataset_config.get('impression_id_col', 'impression_id')
        label_col = dataset_config.get('label_col', {}).get('name', 'label')

        # =====================================================================
        # Phase 1: Batch extract user embeddings and metadata
        # =====================================================================
        request_ids_list = []
        user_embs_list = []
        ground_truths_list = []  # List of sets for process, list of single items for evaluate
        user_features_list = [] if return_output else None
        all_user_features = self.feature_group_manager.get_user_features() if return_output else set()

        # Use a dict to deduplicate and aggregate ground truth per request
        request_info_temp = {}

        self.logger.info("Extracting user embeddings and ground truth from input data...")
        with torch.no_grad():
            for batch_data in tqdm(input_data, desc="Extracting embeddings", disable=True, file=sys.stdout):
                u_emb = self.model.get_user_embedding(batch_data)
                current_request_ids = batch_data[impression_id_col]
                current_gt_item_ids = batch_data[item_id_col]

                for i in range(len(u_emb)):
                    req_id = current_request_ids[i].item()
                    gt_item = current_gt_item_ids[i].item()

                    if req_id not in request_info_temp:
                        emb_idx = len(request_ids_list)
                        request_ids_list.append(req_id)
                        user_embs_list.append(u_emb[i].detach())

                        # Extract ALL user features (FG2 + FG3) for downstream stages
                        if return_output:
                            user_features = {}
                            for feat_name, feat_val in batch_data.items():
                                if feat_name in [item_id_col, impression_id_col, label_col]:
                                    continue
                                # Save all user features that are available (no filtering by FG2)
                                if feat_name not in all_user_features:
                                    continue
                                val = feat_val[i]
                                if isinstance(val, torch.Tensor):
                                    val = val.cpu().numpy()
                                if np.ndim(val) == 0:
                                    val = val.item()
                                user_features[feat_name] = val
                            user_features_list.append(user_features)

                        request_info_temp[req_id] = {
                            'emb_idx': emb_idx,
                            'ground_truth': set()
                        }

                    # Add ground truth item
                    request_info_temp[req_id]['ground_truth'].add(gt_item)

        if not request_ids_list:
            self.logger.warning("No user information extracted from input data.")
            return output, {} if compute_metrics else None

        # Stack all user embeddings
        user_embs = torch.vstack(user_embs_list)
        num_requests = len(request_ids_list)
        num_items = len(self.item_ids)
        item_ids_np = self.item_ids.cpu().numpy()
        request_ids_arr = np.asarray(request_ids_list)
        request_id_dtype = request_ids_arr.dtype if len(request_ids_arr) else np.int64

        # Build ground truth list aligned with request order
        for req_id in request_ids_list:
            ground_truths_list.append(request_info_temp[req_id]['ground_truth'])

        # Pre-compute ground truth indices for vectorized metric computation
        if compute_metrics:
            gt_idx_list = []
            for gt_set in ground_truths_list:
                idxs = np.array([self.item_id_to_idx[gid] for gid in gt_set if gid in self.item_id_to_idx], dtype=np.int64)
                gt_idx_list.append(idxs)

        self.logger.info(f"Scoring {num_requests} unique requests against {num_items} items...")

        # =====================================================================
        # Phase 2: Batch scoring with chunked processing
        # =====================================================================

        # Determine fetch_k based on what we need
        max_positives = max(len(gt) for gt in ground_truths_list) if ground_truths_list else 0
        if compute_metrics:
            max_k_for_metrics = max(metrics_k)
            fetch_k = min(max(self.top_k + max_positives, max_k_for_metrics), num_items)
        else:
            fetch_k = min(self.top_k + max_positives, num_items)

        candidate_frames = [] if return_output else None
        num_chunks = max((num_requests + self.chunk_size - 1) // self.chunk_size, 1)

        for chunk_idx, chunk_start in enumerate(range(0, num_requests, self.chunk_size), start=1):
            chunk_end = min(chunk_start + self.chunk_size, num_requests)
            chunk_len = chunk_end - chunk_start
            chunk_begin_time = datetime.datetime.now()

            self.logger.info(
                f"Scoring chunk {chunk_idx}/{num_chunks} for requests {chunk_start}:{chunk_end} "
                f"(size={chunk_len})"
            )

            with torch.no_grad():
                user_chunk_t = user_embs[chunk_start:chunk_end]  # [chunk, D]
                if user_chunk_t.device != model_device:
                    user_chunk_t = user_chunk_t.to(model_device, non_blocking=(model_device.type == 'cuda'))
                scores_t = torch.matmul(user_chunk_t, self.item_embeddings.T)    # [chunk, num_items]

                topk_k = min(fetch_k, num_items)
                topk_scores_t, topk_indices_t = torch.topk(scores_t, topk_k, dim=1, sorted=True)
                # sorted=True ensures topk_indices[:, :k] gives true top-k for any k <= topk_k

            # ---- GPU-vectorized Recall@K computation ----
            if compute_metrics:
                # Separate single-GT and multi-GT users for vectorized processing
                chunk_gt_idxs = gt_idx_list[chunk_start:chunk_end]
                gt_lens = np.array([len(g) for g in chunk_gt_idxs])

                single_mask = gt_lens == 1
                multi_mask = gt_lens > 1
                device_mask_single = None
                device_mask_multi = None

                # Pre-build single-GT index tensor on GPU (most common case)
                if single_mask.any():
                    single_gt = torch.tensor(
                        [chunk_gt_idxs[i][0] for i in range(chunk_len) if single_mask[i]],
                        dtype=torch.long, device=topk_indices_t.device
                    )  # [num_single]
                    device_mask_single = torch.from_numpy(single_mask).to(topk_indices_t.device)
                    single_topk = topk_indices_t[device_mask_single]  # [num_single, topk_k]

                # Pre-build multi-GT padded tensor on GPU (rare case)
                if multi_mask.any():
                    multi_indices = np.where(multi_mask)[0]
                    max_gt_len = gt_lens[multi_mask].max()
                    # Pad with -1 (impossible index)
                    multi_gt_padded = torch.full(
                        (len(multi_indices), max_gt_len), -1,
                        dtype=torch.long, device=topk_indices_t.device
                    )
                    for j, idx in enumerate(multi_indices):
                        g = chunk_gt_idxs[idx]
                        multi_gt_padded[j, :len(g)] = torch.from_numpy(g).to(topk_indices_t.device)
                    device_mask_multi = torch.from_numpy(multi_mask).to(topk_indices_t.device)
                    multi_topk = topk_indices_t[device_mask_multi]  # [num_multi, topk_k]

                for k in metrics_k:
                    k_eff = min(k, topk_indices_t.shape[1])
                    hits = 0

                    # Single-GT: fully vectorized on GPU
                    if single_mask.any():
                        # [num_single, k_eff] == [num_single, 1] → broadcast
                        hits += (single_topk[:, :k_eff] == single_gt.unsqueeze(1)).any(dim=1).sum().item()

                    # Multi-GT: vectorized on GPU with padded tensor
                    if multi_mask.any():
                        # [num_multi, k_eff, 1] == [num_multi, 1, max_gt] → broadcast
                        hits += (multi_topk[:, :k_eff].unsqueeze(2) == multi_gt_padded.unsqueeze(1)).any(dim=2).any(dim=1).sum().item()

                    total_recall[k] += hits

            # ---- Output-building path: transfer only top-k tensors to CPU and assemble arrays ----
            if return_output:
                topk_indices = topk_indices_t.cpu().numpy()   # [chunk, topk_k]
                topk_scores = topk_scores_t.cpu().numpy().astype(np.float32, copy=False)  # [chunk, topk_k]

                chunk_request_blocks = []
                chunk_item_blocks = []
                chunk_score_blocks = []
                chunk_label_blocks = []

                for i in range(chunk_len):
                    global_idx = chunk_start + i
                    req_id = request_ids_list[global_idx]
                    true_positives = ground_truths_list[global_idx]
                    user_topk_indices = topk_indices[i]
                    user_topk_scores = topk_scores[i]
                    user_topk_item_ids = item_ids_np[user_topk_indices]

                    if true_positives:
                        tp_item_ids = np.fromiter(
                            true_positives, dtype=item_ids_np.dtype, count=len(true_positives)
                        )
                    else:
                        tp_item_ids = np.empty(0, dtype=item_ids_np.dtype)

                    if len(tp_item_ids) > 0:
                        tp_scores = np.zeros(len(tp_item_ids), dtype=np.float32)
                        for tp_pos, tp_item_id in enumerate(tp_item_ids):
                            match_idx = np.flatnonzero(user_topk_item_ids == tp_item_id)
                            if match_idx.size > 0:
                                tp_scores[tp_pos] = user_topk_scores[match_idx[0]]
                    else:
                        tp_scores = np.empty(0, dtype=np.float32)

                    negatives_needed = max(self.top_k - len(tp_item_ids), 0)
                    if negatives_needed > 0:
                        if len(tp_item_ids) > 0:
                            negative_mask = ~np.isin(user_topk_item_ids, tp_item_ids, assume_unique=False)
                            neg_item_ids = user_topk_item_ids[negative_mask][:negatives_needed]
                            neg_scores = user_topk_scores[negative_mask][:negatives_needed]
                        else:
                            neg_item_ids = user_topk_item_ids[:negatives_needed]
                            neg_scores = user_topk_scores[:negatives_needed]
                    else:
                        neg_item_ids = np.empty(0, dtype=item_ids_np.dtype)
                        neg_scores = np.empty(0, dtype=np.float32)
                    neg_pop = np.empty(0, dtype=np.float32)
                    if negatives_needed > 0 and total_popularity_recall is not None:
                        if len(tp_item_ids) > 0:
                            negative_mask = ~np.isin(user_topk_item_ids, tp_item_ids, assume_unique=False)
                            neg_indices = user_topk_indices[negative_mask][:negatives_needed]
                        else:
                            neg_indices = user_topk_indices[:negatives_needed]
                        neg_pop = self.item_popularity[neg_indices]

                    total_candidates = len(tp_item_ids) + len(neg_item_ids)
                    if total_candidates == 0:
                        continue

                    if total_popularity_recall is not None:
                        if len(tp_item_ids) > 0:
                            tp_pop = np.fromiter(
                                (
                                    self.item_popularity[self.item_id_to_idx[tp_item_id]]
                                    if tp_item_id in self.item_id_to_idx else 0.0
                                    for tp_item_id in tp_item_ids
                                ),
                                dtype=np.float32,
                                count=len(tp_item_ids)
                            )
                        else:
                            tp_pop = np.empty(0, dtype=np.float32)

                        popularity_scores = np.concatenate([tp_pop, neg_pop])
                        popularity_labels = np.concatenate([
                            np.ones(len(tp_item_ids), dtype=np.int8),
                            np.zeros(len(neg_item_ids), dtype=np.int8),
                        ])
                        popularity_item_ids = np.concatenate([tp_item_ids, neg_item_ids])
                        # Tie-break by item_id to avoid favoring prepended positives.
                        popularity_order = np.lexsort((
                            popularity_item_ids.astype(np.float64, copy=False),
                            -popularity_scores
                        ))
                        best_positive_rank = np.flatnonzero(popularity_labels[popularity_order] == 1)
                        if best_positive_rank.size > 0:
                            best_positive_rank = int(best_positive_rank[0]) + 1
                            for k in popularity_metrics_k:
                                if best_positive_rank <= min(k, total_candidates):
                                    total_popularity_recall[k] += 1

                    chunk_request_blocks.append(np.full(total_candidates, req_id, dtype=request_id_dtype))
                    chunk_item_blocks.append(np.concatenate([tp_item_ids, neg_item_ids]))
                    chunk_score_blocks.append(np.concatenate([tp_scores, neg_scores]))
                    chunk_label_blocks.append(np.concatenate([
                        np.ones(len(tp_item_ids), dtype=np.int8),
                        np.zeros(len(neg_item_ids), dtype=np.int8),
                    ]))

                if chunk_request_blocks:
                    chunk_frame = pd.DataFrame({
                        'request_id': np.concatenate(chunk_request_blocks),
                        'item_id': np.concatenate(chunk_item_blocks),
                        'score': np.concatenate(chunk_score_blocks),
                        'label': np.concatenate(chunk_label_blocks),
                    })
                    candidate_frames.append(chunk_frame)

            del topk_scores_t, topk_indices_t, scores_t
            elapsed = (datetime.datetime.now() - chunk_begin_time).total_seconds()
            self.logger.info(f"Finished chunk {chunk_idx}/{num_chunks} in {elapsed:.2f}s")

        # Finalize metrics
        metrics = None
        if compute_metrics:
            metrics = {f"Recall@{k}": v / num_requests for k, v in total_recall.items()}
            if total_popularity_recall is not None:
                metrics.update({
                    f"PopularityRecall@{k}": v / num_requests
                    for k, v in total_popularity_recall.items()
                })
            self.logger.info(f"Metrics: {metrics}")

            # Save metrics to CSV
            metrics_path = os.path.join(self.output_dir, "eval_metrics.csv")
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric_name', 'value'])
                for name, value in sorted(metrics.items()):
                    writer.writerow([name, f"{value:.6f}"])

        # Finalize output using DataFrame-first API
        if return_output:
            if candidate_frames:
                candidates_df = pd.concat(candidate_frames, ignore_index=True)
            else:
                candidates_df = pd.DataFrame(columns=['request_id', 'item_id', 'score', 'label'])

            if user_features_list is not None and len(user_features_list) > 0:
                user_features_df = pd.DataFrame(user_features_list)
                user_features_df.insert(0, 'request_id', request_ids_arr)
            else:
                user_features_df = pd.DataFrame({'request_id': request_ids_arr})

            output = StageOutput.from_dataframes(
                stage_name=self.stage_name,
                candidates_df=candidates_df,
                user_features_df=user_features_df,
                metrics=metrics or {},
                metadata={}
            )
            self.logger.info(f"Generated {output.get_total_candidates()} candidates across {output.get_num_requests()} requests.")
            output.end_time = datetime.datetime.now().isoformat()

        return output, metrics

    def process(self,
                input_data: Any,
                item_data=None,
                **kwargs):
        """
        Process user queries to retrieve top-K candidates from the item pool.

        Optionally computes evaluation metrics during processing.

        Args:
            input_data: FuxiCTR DataGenerator with user queries
            item_data: FuxiCTR DataGenerator for item features (to build index)
            **kwargs: Additional parameters (chunk_size, etc.)

        Returns:
            StageOutput with candidate sets
        """
        self.logger.info(f"Starting retrieval process for candidate generation (top_k={self.top_k}).")

        compute_metrics = kwargs.pop('compute_metrics', True)
        metrics_k = kwargs.pop('metrics_k', self.metrics_k)

        output, metrics = self._retrieve_and_score(
            input_data=input_data,
            item_data=item_data,
            return_output=True,
            compute_metrics=compute_metrics,
            metrics_k=metrics_k,
            **kwargs
        )

        return output, metrics

    def evaluate(self, test_data, metrics_k=None, **kwargs) -> Dict[str, float]:
        """
        Compute Recall@K metrics against built item index.

        This is a lightweight version that only computes metrics without
        generating the full StageOutput.

        Args:
            test_data: FuxiCTR DataGenerator with test data
            metrics_k: List of K values for Recall@K
            **kwargs: Additional parameters

        Returns:
            Dictionary of metric names to values
        """
        self.logger.info("Evaluating...")

        _, metrics = self._retrieve_and_score(
            input_data=test_data,
            item_data=None,  # Assume index already built
            return_output=False,
            compute_metrics=True,
            metrics_k=metrics_k,
            **kwargs
        )

        return metrics or {}
