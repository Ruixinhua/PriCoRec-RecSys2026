# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Dual Tower Retrieval Model

This module implements a dual-tower (two-tower) model for candidate retrieval.
Based on the DSSM architecture from FuxiCTR.
"""

import torch
from torch import nn
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from .losses import DiversityLossMixin


class DualTowerRetrieval(DiversityLossMixin, BaseModel):
    """
    Dual-tower model for candidate retrieval.

    Architecture:
    - User Tower: Processes FG1 + FG2 features related to user
    - Item Tower: Processes FG1 + FG2 features related to item
    - Similarity: Dot product of user and item embeddings

    Supports optional diversity loss via DiversityLossMixin.
    """

    def __init__(self,
                 feature_map,
                 model_id="DualTowerRetrieval",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 user_tower_layers=None,
                 item_tower_layers=None,
                 user_tower_activations="ReLU",
                 item_tower_activations="ReLU",
                 user_tower_dropout=0.1,
                 item_tower_dropout=0.1,
                 batch_norm=True,
                 use_l2_norm=True,
                 temperature=0.1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_diversity_loss=False,
                 diversity_lambda=0.7,
                 diversity_theta=0.7,
                 **kwargs):
        """
        Initialize Dual Tower Retrieval model.

        Args:
            feature_map: FuxiCTR FeatureMap object
            model_id: Model identifier
            gpu: GPU device ID (-1 for CPU)
            learning_rate: Learning rate
            embedding_dim: Embedding dimension for features
            user_tower_layers: Hidden units for user tower MLP
            item_tower_layers: Hidden units for item tower MLP
            user_tower_activations: Activation function for user tower
            item_tower_activations: Activation function for item tower
            user_tower_dropout: Dropout rate for user tower
            item_tower_dropout: Dropout rate for item tower
            batch_norm: Whether to use batch normalization
            use_l2_norm: Whether to L2 normalize embeddings
            temperature: Temperature for similarity computation
            use_diversity_loss: Whether to use diversity loss
            diversity_lambda: Weight of diversity loss in total loss
            diversity_theta: Weight between prediction sum and diversity term
        """
        # Initialize DiversityLossMixin first
        DiversityLossMixin.__init__(
            self,
            use_diversity_loss=use_diversity_loss,
            diversity_lambda=diversity_lambda,
            diversity_theta=diversity_theta,
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
        self.use_l2_norm = use_l2_norm
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.use_diversity_loss:
            self.logger.info(f"Using Diversity Loss: lambda={diversity_lambda}, theta={diversity_theta}")
        # Create embedding layer
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        # Count user and item fields
        user_fields = sum(1 if feature_spec.get("source") == "user" else 0
                         for _, feature_spec in feature_map.features.items())
        item_fields = sum(1 if feature_spec.get("source") == "item" else 0
                         for _, feature_spec in feature_map.features.items())

        # Fallback: if source not specified, use feature name patterns
        if user_fields == 0 or item_fields == 0:
            self.logger.warning("Feature source not configured, using name-based detection")
            user_fields = 0
            item_fields = 0
            for name, spec in feature_map.features.items():
                if any(p in name.lower() for p in ['user', 'his', 'seq', 'click', 'exp', 'ipv']):
                    user_fields += 1
                else:
                    item_fields += 1

        self.logger.info(f"User fields: {user_fields}, Item fields: {item_fields}")

        # Transformer for sequence encoding (optional)
        self.user_transformer_layers = kwargs.get("user_transformer_layers", 1)
        self.use_user_transformer = kwargs.get("use_user_transformer", False) and self.user_transformer_layers > 0
        if self.use_user_transformer:
            self.user_transformer_heads = kwargs.get("user_transformer_heads", 4)
            self.user_transformer_dim = embedding_dim * max(1, user_fields)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.user_transformer_dim,
                nhead=self.user_transformer_heads,
                dim_feedforward=self.user_transformer_dim * 4,
                dropout=kwargs.get("dropout", 0.1),
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.user_transformer_layers)
            self.logger.info(f"Initialized Transformer User Tower: heads={self.user_transformer_heads}, layers={self.user_transformer_layers}")

        # Transformer for Item Tower (Feature Interaction)
        self.item_transformer_layers = kwargs.get("item_transformer_layers", 1)
        self.use_item_transformer = kwargs.get("use_item_transformer", False) and self.item_transformer_layers > 0
        if self.use_item_transformer:
            self.item_transformer_heads = kwargs.get("item_transformer_heads", 4)
            # Input to transformer is (B, F, D), so d_model = embedding_dim

            item_encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=self.item_transformer_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=kwargs.get("dropout", 0.1),
                batch_first=True
            )
            self.item_transformer_encoder = nn.TransformerEncoder(item_encoder_layer, num_layers=self.item_transformer_layers)
            self.logger.info(f"Initialized Transformer Item Tower: heads={self.item_transformer_heads}, layers={self.item_transformer_layers}")

        # User tower
        # If transformer is used, input is flattened pooled sequence (same dim as before, just better processed)
        self.user_tower = MLP_Block(
            input_dim=embedding_dim * max(1, user_fields),
            output_dim=user_tower_layers[-1],
            hidden_units=user_tower_layers[:-1],
            hidden_activations=user_tower_activations,
            output_activation=None,
            dropout_rates=user_tower_dropout,
            batch_norm=batch_norm
        )

        # Item tower
        self.item_tower = MLP_Block(
            input_dim=embedding_dim * max(1, item_fields),
            output_dim=item_tower_layers[-1],
            hidden_units=item_tower_layers[:-1],
            hidden_activations=item_tower_activations,
            output_activation=None,
            dropout_rates=item_tower_dropout,
            batch_norm=batch_norm
        )

        # Store field counts for embedding separation
        self.user_fields = user_fields
        self.item_fields = item_fields

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Input batch

        Returns:
            Dictionary with y_pred and optionally user/item embeddings
        """
        # Flatten and pass through towers
        user_emb = self.get_user_embedding(inputs)
        item_emb = self.get_item_embedding(inputs)
        y_pred = self.cal_similarity(user_emb, item_emb)
        return_dict = {
            "y_pred": y_pred,
            "user_embedding": user_emb,
            "item_embedding": item_emb
        }
        return return_dict

    def get_user_embedding(self, inputs) -> torch.Tensor:
        """Get user embedding [batch_size, dim]"""
        X = self.get_inputs(inputs, feature_source="user")

        if self.use_user_transformer:
            # Handle sequence features with Transformer
            # 1. Get embeddings dict without flattening/stacking yet
            feat_emb_dict = self.embedding_layer(X)

            # 2. Extract and align sequences
            seq_embs = []
            max_len = 0

            # Collect all user features
            for name, emb in feat_emb_dict.items():
                if emb.dim() == 2: # (B, D) -> (B, 1, D)
                    emb = emb.unsqueeze(1)
                if emb.size(1) > max_len:
                    max_len = emb.size(1)
                seq_embs.append(emb)

            if not seq_embs:
                raise ValueError("No user features found for Transformer!")

            # Pad and Concatenate along feature dimension: (B, L, D_total)
            # We assume features are [feat1, feat2, ...] at each time step effectively.
            # But wait, different sequences (click vs exposure) might not align in time.
            # However, for a general 'User Representation' from sequences, concatenating channel-wise
            # and letting Transformer attend is a standard approach.

            aligned_embs = []
            for emb in seq_embs:
                if emb.size(1) < max_len:
                    # Pad on time dim (dim 1) with 0
                    pad_len = max_len - emb.size(1)
                    pad = torch.zeros(emb.size(0), pad_len, emb.size(2), device=emb.device)
                    emb = torch.cat([emb, pad], dim=1)
                aligned_embs.append(emb)

            # Concat features: (B, L, D1) + (B, L, D2) -> (B, L, D_total)
            # D_total should match embedding_dim * user_fields
            user_seq = torch.cat(aligned_embs, dim=2)

            # Transformer Encoding
            # user_seq: (B, L, D_model)
            transformed = self.transformer_encoder(user_seq)

            # Mean Pooling / Attention Pooling (using Mean for simplicity/robustness)
            # Masking padding would be ideal but 0-padding works okay with LayerNorm often.
            user_emb = transformed.mean(dim=1)

            # Pass through MLP
            user_emb = self.user_tower(user_emb)
        else:
            # Original Logic
            feat_emb_dict = self.embedding_layer(X)
            user_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="user")
            user_emb = self.user_tower(user_emb.view(user_emb.size(0), -1))

        # L2 normalize if requested
        if self.use_l2_norm:
            user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=-1)
        return user_emb

    def get_item_embedding(self, inputs) -> torch.Tensor:
        """Get item embedding [batch_size, dim]"""
        X = self.get_inputs(inputs, feature_source="item")
        feat_emb_dict = self.embedding_layer(X)

        if self.use_item_transformer:
            # Get stacked embeddings: (B, NumFields, EmbeddingDim)
            item_feats = self.embedding_layer.dict2tensor(feat_emb_dict, flatten_emb=False, feature_source="item")

            # Transformer Encoding
            # item_feats: (B, F, D)
            transformed = self.item_transformer_encoder(item_feats)

            # Flatten: (B, F, D) -> (B, F*D)
            # This allows the MLP to see all transformed field interactions
            item_emb_input = transformed.view(transformed.size(0), -1)

            item_emb = self.item_tower(item_emb_input)
        else:
            item_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="item")
            item_emb = self.item_tower(item_emb.view(item_emb.size(0), -1))

        # L2 normalize if requested
        if self.use_l2_norm:
            item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=-1)
        return item_emb

    def cal_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity scores between user and item embeddings.
        Applies output_activation (sigmoid) — use for BCE-based training only.

        Args:
            user_emb: User embeddings [batch_size, dim]
            item_emb: Item embeddings [batch_size, dim]
        Returns:
            Similarity scores [batch_size, 1]
        """
        # Compute similarity (dot product)
        similarity = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        # Scale by temperature
        similarity = similarity / self.temperature
        # Apply output activation (sigmoid for binary classification)
        return self.output_activation(similarity)

    def cal_similarity_raw(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Calculate raw similarity logits (no sigmoid). Use for pairwise ranking losses
        (BPR, margin, softmax) where the loss function handles the activation.

        Args:
            user_emb: User embeddings [batch_size, dim] or [batch_size, num_items, dim]
            item_emb: Item embeddings [batch_size, dim] or [batch_size, num_items, dim]
        Returns:
            Raw similarity logits [batch_size, 1] or [batch_size, num_items]
        """
        if user_emb.dim() == 3 or item_emb.dim() == 3:
            # Batched: user_emb [B, 1, D] x item_emb [B, N, D] -> [B, N]
            return (user_emb * item_emb).sum(dim=-1) / self.temperature
        similarity = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        return similarity / self.temperature

    def get_diversity_item_embeddings(self, feat_emb_dict):
        """
        Override mixin method to use item tower embeddings for diversity.

        For DualTowerRetrieval, we use the final item embeddings from the item tower
        instead of raw feature embeddings.
        """
        # Use stored item embedding from forward pass
        if hasattr(self, '_last_item_embedding'):
            return self._last_item_embedding
        return None

    def compute_loss(self, return_dict, y_true):
        """
        Compute loss with optional diversity regularization.

        Args:
            return_dict: Output from forward pass (contains y_pred, item_embedding)
            y_true: Ground truth labels

        Returns:
            Total loss
        """
        total_base = super().compute_loss(return_dict, y_true)

        # For diversity loss, use item embeddings from forward pass
        if self._use_diversity_loss and "item_embedding" in return_dict:
            from .losses import compute_diversity_loss
            diversity_loss = compute_diversity_loss(
                item_embeddings=return_dict["item_embedding"],
                y_pred=return_dict["y_pred"],
                theta=self._diversity_theta,
            )
        else:
            diversity_loss = None

        total_loss = self.add_diversity_to_loss(
            total_base,
            diversity_loss
        )

        return total_loss
