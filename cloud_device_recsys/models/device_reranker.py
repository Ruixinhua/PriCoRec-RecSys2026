# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Device Re-ranker Model

This module implements a lightweight re-ranking model for on-device inference.
It uses ALL feature groups (FG1 + FG2 + FG3) including private features.
"""

import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from .losses import DiversityLossMixin


class DeviceReranker(DiversityLossMixin, BaseModel):
    """
    Lightweight on-device re-ranking model.

    Features:
    - Uses ALL feature groups (FG1 + FG2 + FG3)
    - FG3 features provide personalization with private user data
    - Lightweight architecture for mobile/edge deployment
    - Support for knowledge distillation from teacher model
    - ONNX export for cross-platform deployment
    """

    def __init__(self,
                 feature_map,
                 model_id="DeviceReranker",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,  # Smaller for device
                 hidden_units=[64, 32],  # Compact architecture
                 hidden_activations="ReLU",
                 dropout_rates=0.0,  # Less dropout for smaller model
                 batch_norm=False,  # Avoid BN for easier mobile deployment
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_diversity_loss=False,
                 diversity_lambda=0.7,
                 diversity_theta=0.7,
                 **kwargs):
        """
        Initialize Device Re-ranker.

        Args:
            feature_map: FuxiCTR FeatureMap object
            model_id: Model identifier
            gpu: GPU device ID (-1 for CPU)
            learning_rate: Learning rate
            embedding_dim: Embedding dimension (smaller for efficiency)
            hidden_units: Hidden layer sizes (compact)
            hidden_activations: Activation function
            dropout_rates: Dropout rate
            batch_norm: Whether to use batch normalization
            use_diversity_loss: Whether to use diversity loss
            diversity_lambda: Weight of diversity loss
            diversity_theta: Weight between prediction sum and diversity term
        """
        # Initialize DiversityLossMixin first
        DiversityLossMixin.__init__(
            self,
            use_diversity_loss=use_diversity_loss,
            diversity_lambda=diversity_lambda,
            diversity_theta=diversity_theta,
            **kwargs
        )

        BaseModel.__init__(
            self,
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Feature embedding layer
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        # Calculate input dimension
        num_fields = len(feature_map.features)
        input_dim = embedding_dim * num_fields

        self.logger.info(f"DeviceReranker: {num_fields} fields (including FG3), "
                        f"embedding_dim={embedding_dim}, input_dim={input_dim}")

        # Compact MLP for scoring
        self.mlp = MLP_Block(
            input_dim=input_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        # Count and log model size
        self._log_model_size()

    def _log_model_size(self):
        """Log model size for deployment awareness"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Estimate memory size (float32 = 4 bytes)
        size_mb = (total_params * 4) / (1024 * 1024)

        self.logger.info(f"Model size: {total_params:,} params ({trainable_params:,} trainable)")
        self.logger.info(f"Estimated size: {size_mb:.2f} MB")

    def get_diversity_item_embeddings(self, feat_emb_dict):
        """Override to provide item embeddings for diversity loss."""
        # Use configured features if available
        emb = super().get_diversity_item_embeddings(feat_emb_dict)
        if emb is not None:
            return emb

        # Fallback: try to finding 'cand_item_id'
        if 'cand_item_id' in feat_emb_dict:
            return feat_emb_dict['cand_item_id']

        return None

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Input batch

        Returns:
            Dictionary with y_pred
        """
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)
        feat_emb = self.embedding_layer.dict2tensor(feat_emb_dict)

        # Flatten embeddings
        flat_emb = feat_emb.flatten(start_dim=1)

        # MLP prediction
        logits = self.mlp(flat_emb)
        y_pred = self.output_activation(logits)

        return_dict = {
            "y_pred": y_pred,
            "logit": logits
        }

        # Add embedding dict for diversity loss
        if self.use_diversity_loss:
            return_dict["feat_emb_dict"] = feat_emb_dict

        return return_dict

    def compute_loss(self, return_dict, y_true):
        """
        Compute loss with optional diversity regularization.

        Args:
            return_dict: Output from forward pass
            y_true: Ground truth labels

        Returns:
            Total loss
        """
        base_loss = super().compute_loss(return_dict, y_true)

        # Add Diversity Regularization
        diversity_loss = None
        if self.use_diversity_loss and "feat_emb_dict" in return_dict:
            diversity_loss = self.compute_diversity_regularization(
                feat_emb_dict=return_dict["feat_emb_dict"],
                y_pred=return_dict["y_pred"]
            )

        total_loss = self.add_diversity_to_loss(
            base_loss,
            diversity_loss
        )

        return total_loss

    def rerank(self,
               candidates_scores: np.ndarray,
               private_features: Optional[Dict[str, Any]] = None,
               top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank candidates using private (FG3) features.

        Args:
            candidates_scores: Pre-ranking scores [num_candidates]
            private_features: FG3 features for personalization
            top_k: Number of final recommendations

        Returns:
            Tuple of (selected_indices, final_scores)
        """
        # In real implementation, combine candidate features with FG3
        # This is simplified - would need proper feature tensor construction

        # For now, just select top-k
        k = min(top_k, len(candidates_scores))
        top_indices = np.argsort(-candidates_scores)[:k]
        top_scores = candidates_scores[top_indices]

        return top_indices, top_scores

    def export_onnx(self, export_path: str, sample_input: torch.Tensor) -> str:
        """
        Export model to ONNX format for device deployment.

        Args:
            export_path: Path to save ONNX model
            sample_input: Sample input tensor for tracing

        Returns:
            Path to exported model
        """
        self.eval()

        # Ensure directory exists
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.',
                   exist_ok=True)

        # Export to ONNX
        torch.onnx.export(
            self,
            (sample_input,),
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['predictions'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )

        self.logger.info(f"Exported model to ONNX: {export_path}")

        # Log file size
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        self.logger.info(f"ONNX model size: {size_mb:.2f} MB")

        return export_path

    def export_torchscript(self, export_path: str, sample_input: torch.Tensor) -> str:
        """
        Export model to TorchScript for mobile deployment.

        Args:
            export_path: Path to save TorchScript model
            sample_input: Sample input tensor for tracing

        Returns:
            Path to exported model
        """
        self.eval()

        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.',
                   exist_ok=True)

        # Trace the model
        traced_model = torch.jit.trace(self, sample_input)
        traced_model.save(export_path)

        self.logger.info(f"Exported model to TorchScript: {export_path}")

        # Log file size
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        self.logger.info(f"TorchScript model size: {size_mb:.2f} MB")

        return export_path
