# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Deep Interest Network (DIN) for Pre-ranking

This module implements the DIN model for the pre-ranking stage,
leveraging user behavior sequences with an attention mechanism.
"""

import torch
import logging

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from .losses import DiversityLossMixin


class DINRanker(DiversityLossMixin, BaseModel):
    """
    Deep Interest Network (DIN) model for pre-ranking.

    Supports optional diversity loss via DiversityLossMixin.
    """

    def __init__(self,
                 feature_map,
                 model_id="DINRanker",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=32,
                 hidden_units=[128, 64],
                 hidden_activations="Dice",
                 dropout_rates=0.1,
                 batch_norm=True,
                 attention_hidden_units=[32, 16],
                 attention_dropout_rates=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_diversity_loss=False,
                 diversity_lambda=0.7,
                 diversity_theta=0.7,
                 **kwargs):
        # Initialize DiversityLossMixin first
        DiversityLossMixin.__init__(
            self,
            use_diversity_loss=use_diversity_loss,
            diversity_lambda=diversity_lambda,
            diversity_theta=diversity_theta,
        )
        # Initialize BaseModel
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
        if self.use_diversity_loss:
            self.logger.info(f"Using Diversity Loss: lambda={diversity_lambda}, theta={diversity_theta}")
        # --- Feature Embedding Layer ---
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        # --- Identify Sequence Features ---
        self.sequence_features = []
        self.target_item_features = []

        for feature_name, feature_spec in feature_map.features.items():
            if feature_spec.get('type') == 'sequence':
                target_name = feature_spec.get('share_embedding') or \
                              (feature_spec.get('share_embedding_map') and list(feature_spec['share_embedding_map'].values())[0])
                if target_name:
                    self.sequence_features.append(feature_name)
                    if target_name not in self.target_item_features:
                        self.target_item_features.append(target_name)

        if 'cand_item_id' not in self.target_item_features and 'cand_item_id' in feature_map.features:
            self.target_item_features.append('cand_item_id')
        self.logger.info(self.target_item_features)

        # Set item features for diversity loss (from mixin)
        self._diversity_item_features = self.target_item_features

        self.target_item_attention_key = self.target_item_features[0]

        # --- Attention Mechanism ---
        attention_input_dim = embedding_dim * 2
        self.attention_mlp = MLP_Block(
            input_dim=attention_input_dim,
            output_dim=1,
            hidden_units=attention_hidden_units,
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=attention_dropout_rates,
            batch_norm=False
        )

        # --- Final MLP ---
        # Identify all features that are CATEGORICAL and NOT SEQUENCES
        self.final_feature_names = []
        for name, spec in feature_map.features.items():
            if spec.get('type') == 'categorical' and name not in self.sequence_features:
                self.final_feature_names.append(name)

        # Dimension = (Number of categorical non-seq features) * embedding_dim + attended_interest (embedding_dim)
        final_mlp_input_dim = (len(self.final_feature_names) + 1) * embedding_dim

        self.mlp = MLP_Block(
            input_dim=final_mlp_input_dim,
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
        self.logger.info(f"DINRanker: Features for MLP: {self.final_feature_names}")
        self.logger.info(f"DINRanker: final_mlp_input_dim: {final_mlp_input_dim}")

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feat_emb_dict = self.embedding_layer(X)

        # Store feat_emb_dict for diversity loss computation
        self._last_feat_emb_dict = feat_emb_dict

        target_item_emb = feat_emb_dict[self.target_item_attention_key]

        # 3D ensure
        curr_seq_emb = feat_emb_dict[self.sequence_features[0]]
        if curr_seq_emb.dim() == 2:
            curr_seq_emb = curr_seq_emb.unsqueeze(1)

        target_item_emb_expanded = target_item_emb.unsqueeze(1)
        B, L, D = curr_seq_emb.size()
        target_emb_expanded = target_item_emb_expanded.expand(B, L, D)

        attention_input = torch.cat([target_emb_expanded, curr_seq_emb], dim=-1)
        attention_scores_logits = self.attention_mlp(attention_input)

        # Mask padding *before* softmax. Multiplying logits by zero leaves a
        # finite logit for padded tokens, so they still receive probability
        # mass. Fully padded histories need a finite fallback to avoid a
        # softmax over only -inf values; their final attention weights remain
        # exactly zero.
        actual_mask = X[self.sequence_features[0]] != 0
        if actual_mask.dim() == 1:
            actual_mask = actual_mask.unsqueeze(1)

        attention_logits = attention_scores_logits.squeeze(-1)
        masked_attention_scores_logits = attention_logits.masked_fill(~actual_mask, float("-inf"))
        fully_padded = ~actual_mask.any(dim=-1, keepdim=True)
        masked_attention_scores_logits = masked_attention_scores_logits.masked_fill(fully_padded, 0.0)
        attention_weights = torch.softmax(masked_attention_scores_logits, dim=-1)
        attention_weights = attention_weights * actual_mask.to(dtype=attention_weights.dtype)

        attended_user_interest = torch.sum(curr_seq_emb * attention_weights.unsqueeze(-1), dim=1)

        # Concatenate for Final MLP
        final_mlp_input_list = []
        for name in self.final_feature_names:
            final_mlp_input_list.append(feat_emb_dict[name])

        final_mlp_input_list.append(attended_user_interest)
        final_mlp_input = torch.cat(final_mlp_input_list, dim=-1)

        logits = self.mlp(final_mlp_input)
        y_pred = self.output_activation(logits)

        return_dict = {"y_pred": y_pred}

        # Add embedding dict for diversity loss (needed to avoid overwrite during negative sampling)
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
        # Base loss (BCE or other) + regularization
        base_loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        reg_loss = self.regularization_loss()
        total_base = base_loss + reg_loss

        # Compute diversity regularization using mixin method
        if hasattr(self, '_last_feat_emb_dict'):
            diversity_loss = self.compute_diversity_regularization(
                self._last_feat_emb_dict,
                return_dict["y_pred"]
            )
        else:
            diversity_loss = None

        total_loss = self.add_diversity_to_loss(
            total_base,
            diversity_loss
        )

        return total_loss
