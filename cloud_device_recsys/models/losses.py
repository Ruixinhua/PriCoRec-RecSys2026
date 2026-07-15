# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Loss Functions and Mixins for Recommendation Models.

This module provides:
1. Standalone loss functions (e.g., diversity_loss) for easy reuse
2. Mixin classes for adding loss capabilities to existing models
"""

import logging
import math
from typing import Dict, Optional, List

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Pairwise Ranking Loss Functions
# =============================================================================

def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) Loss.

    BPR loss encourages positive items to have higher scores than negative items:
        loss = -log(sigmoid(pos_score - neg_score))

    Args:
        pos_scores: Positive item scores, shape [batch_size, 1] or [batch_size]
        neg_scores: Negative item scores, shape [batch_size, num_negatives]
                   or [batch_size]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        BPR loss value

    Example:
        >>> pos_scores = model(pos_batch)["y_pred"]  # [B, 1]
        >>> neg_scores = model(neg_batch)["y_pred"]  # [B, num_neg]
        >>> loss = bpr_loss(pos_scores, neg_scores)
    """
    pos_scores = pos_scores.view(-1, 1)  # [B, 1]

    if neg_scores.dim() == 1:
        neg_scores = neg_scores.view(-1, 1)  # [B, 1]

    # Difference: pos - neg for each negative
    diff = pos_scores - neg_scores  # [B, num_neg]

    # BPR loss: -log(sigmoid(diff))
    loss = -torch.nn.functional.logsigmoid(diff)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Margin Ranking Loss (Hinge Loss).

    Margin loss enforces a minimum margin between positive and negative scores:
        loss = max(0, margin - pos_score + neg_score)

    Args:
        pos_scores: Positive item scores, shape [batch_size, 1] or [batch_size]
        neg_scores: Negative item scores, shape [batch_size, num_negatives]
        margin: Minimum margin between positive and negative scores (default: 1.0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Margin ranking loss value
    """
    pos_scores = pos_scores.view(-1, 1)  # [B, 1]

    if neg_scores.dim() == 1:
        neg_scores = neg_scores.view(-1, 1)  # [B, 1]

    # Hinge loss: max(0, margin - pos + neg)
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0.0)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def softmax_cross_entropy_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Softmax Cross-Entropy Loss for Pairwise Ranking.

    Treats ranking as classification: positive item should have highest score
    among all candidates (positive + negatives).

    Args:
        pos_scores: Positive item scores, shape [batch_size, 1]
        neg_scores: Negative item scores, shape [batch_size, num_negatives]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Cross-entropy loss value
    """
    pos_scores = pos_scores.view(-1, 1)  # [B, 1]

    if neg_scores.dim() == 1:
        neg_scores = neg_scores.view(-1, 1)  # [B, 1]

    # Concatenate: [pos, neg1, neg2, ...] -> [B, 1+num_neg]
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)

    # Target: positive is at index 0
    targets = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)

    loss = torch.nn.functional.cross_entropy(all_scores, targets, reduction=reduction)

    return loss


# =============================================================================
# Knowledge Distillation Loss Functions
# =============================================================================

def kd_mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    MSE-based knowledge distillation loss between student and teacher logits.

    Args:
        student_logits: Student model logits, shape [B, 1] or [B]
        teacher_logits: Teacher model logits (detached), shape [B, 1] or [B]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE loss value
    """
    return torch.nn.functional.mse_loss(
        student_logits.view(-1), teacher_logits.detach().view(-1), reduction=reduction
    )


def kd_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = 'batchmean',
) -> torch.Tensor:
    """
    KL-divergence knowledge distillation loss with temperature scaling.

    Softens both student and teacher distributions with temperature, then
    computes KL(teacher || student). Loss is scaled by T^2 to keep gradients
    comparable across different temperatures (Hinton et al., 2015).

    Args:
        student_logits: Student model logits, shape [B, 1] or [B]
        teacher_logits: Teacher model logits (detached), shape [B, 1] or [B]
        temperature: Temperature for softening distributions (default 1.0)
        reduction: KL-div reduction mode (default 'batchmean')

    Returns:
        KL-divergence loss scaled by T^2
    """
    # Compute Bernoulli KL through BCE-with-logits instead of materializing
    # [B, 2] probability tensors.  The calculation stays in float32 so
    # saturated FP16 probabilities cannot turn log(0) into NaN/Inf.
    student_scaled = student_logits.reshape(-1).float() / temperature
    teacher_scaled = teacher_logits.detach().reshape(-1).float() / temperature
    teacher_probs = torch.sigmoid(teacher_scaled)
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(
        student_scaled,
        teacher_probs,
        reduction="none",
    ) - torch.nn.functional.binary_cross_entropy_with_logits(
        teacher_scaled,
        teacher_probs,
        reduction="none",
    )
    # The two stable BCE terms can differ by a tiny negative rounding error
    # when student and teacher are nearly identical. KL is non-negative by
    # definition, so enforce that contract before reduction.
    per_example = per_example.clamp_min(0.0)

    if reduction in ("batchmean", "mean"):
        loss = per_example.mean()
    elif reduction == "sum":
        loss = per_example.sum()
    elif reduction == "none":
        loss = per_example
    else:
        raise ValueError(f"Unsupported KL reduction: {reduction}")
    return loss * (temperature ** 2)


def kd_cosine_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Cosine similarity knowledge distillation loss.

    Computes 1 - cosine_similarity(student, teacher) to encourage the student's
    logit vector to align directionally with the teacher's logit vector.
    This focuses on relative ranking agreement across the batch.

    Loss range: [0, 2], where 0 = identical direction, 2 = opposite direction.

    Args:
        student_logits: Student model logits, shape [B, 1] or [B]
        teacher_logits: Teacher model logits (detached), shape [B, 1] or [B]
        reduction: 'mean', 'sum', or 'none' (only 'mean' is used for scalar output)

    Returns:
        Cosine distance loss (scalar)
    """
    student_logits = student_logits.view(-1)
    teacher_logits = teacher_logits.detach().view(-1)

    # Cosine similarity between the two logit vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        student_logits.unsqueeze(0), teacher_logits.unsqueeze(0)
    )
    # Loss: 1 - cos_sim (0 when perfectly aligned, 2 when opposed)
    return 1.0 - cos_sim.squeeze()


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_loss_type: str = 'mse',
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Dispatcher for knowledge distillation loss functions.

    Args:
        student_logits: Student model logits [B, 1] or [B]
        teacher_logits: Teacher model logits (detached) [B, 1] or [B]
        kd_loss_type: 'mse', 'kl_div', or 'cosine'
        temperature: Temperature for KL-div softening (only used with 'kl_div')

    Returns:
        KD loss scalar
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    if student_logits.numel() != teacher_logits.numel():
        raise ValueError(
            "student and teacher logits must have the same shape after flattening; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )

    kd_loss_type = str(kd_loss_type).strip().lower()
    if kd_loss_type == "kl":
        kd_loss_type = "kl_div"

    if kd_loss_type == 'mse':
        return kd_mse_loss(student_logits, teacher_logits)
    elif kd_loss_type == 'kl_div':
        return kd_kl_div_loss(student_logits, teacher_logits, temperature=temperature)
    elif kd_loss_type == 'cosine':
        return kd_cosine_loss(student_logits, teacher_logits)
    else:
        raise ValueError(f"Unknown kd_loss_type: {kd_loss_type}. Choices: mse, kl_div, cosine")


def compute_diversity_loss(
    item_embeddings: torch.Tensor,
    y_pred: torch.Tensor,
    theta: float = 0.7,
    eps: float = 1e-6,
    kernel: str = 'gram',
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Compute diversity loss based on item embedding similarity and prediction scores.

    This loss encourages diversity in recommendations by penalizing similar items
    being predicted together. The formula is:
        diversity_loss = theta * sum(y_pred) + (1 - theta) * log_det(similarity_matrix)

    Args:
        item_embeddings: Item embedding tensor of shape (batch_size, embedding_dim)
                        or (batch_size, num_items, embedding_dim)
        y_pred: Prediction scores of shape (batch_size, 1) or (batch_size,)
        theta: Weight between prediction sum and diversity term (default: 0.7)
        eps: Small epsilon for numerical stability in logdet (default: 1e-6)

    Returns:
        Diversity loss scalar (higher value = more diverse)

    Example:
        >>> item_emb = model.get_item_embeddings(batch)  # (B, D)
        >>> y_pred = model(batch)["y_pred"]               # (B, 1)
        >>> div_loss = compute_diversity_loss(item_emb, y_pred, theta=0.7)
        >>> total_loss = base_loss - lambda_ * div_loss
    """
    # Normalize embeddings for cosine similarity or RBF kernel
    item_vectors_normalized = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)

    # Compute cosine similarity matrix
    if item_embeddings.dim() == 3:
        # (batch_size, num_items, embedding_dim) -> per-sample similarity
        cosine_similarity = torch.bmm(
            item_vectors_normalized,
            item_vectors_normalized.transpose(-1, -2)
        )
    else:
        # (batch_size, embedding_dim) -> global batch similarity
        cosine_similarity = torch.matmul(
            item_vectors_normalized,
            item_vectors_normalized.t()
        )

    if kernel == 'rbf':
        # RBF Kernel: exp(-gamma * ||x_i - x_j||^2)
        # For normalized embeddings, ||x_i - x_j||^2 = 2 - 2 * cos_sim
        dist_sq = 2.0 - 2.0 * cosine_similarity
        similarity_matrix = torch.exp(-gamma * dist_sq)
    elif kernel == 'gram':
        # Fast DPP Literature Gram Matrix: L_ij = r_i r_j <f_i, f_j>
        # Ensure y_pred is positive (as required by DPP item scores)
        # For pairwise ranking logics, this assumes y_pred is transformed to [0,1] or positive scaling
        r = torch.clamp(y_pred, min=eps)
        if r.dim() == 1:
            r = r.unsqueeze(-1) # [B, 1]

        if cosine_similarity.dim() == 3: # Per-user
            # r could be [B, num_items, 1]
            if r.dim() == 2:
                r = r.unsqueeze(-1)
            r_matrix = torch.bmm(r, r.transpose(-1, -2))
        else: # Global batch
            r_matrix = torch.matmul(r, r.t())

        similarity_matrix = r_matrix * cosine_similarity
    else:
        # Default: Map cosine similarity [-1, 1] to [0, 1]
        similarity_matrix = (1 + cosine_similarity) / 2

    # Add small identity matrix for numerical stability before logdet
    identity = torch.eye(
        similarity_matrix.size(-1),
        device=similarity_matrix.device
    ) * eps

    if similarity_matrix.dim() == 3:
        identity = identity.unsqueeze(0).expand(similarity_matrix.size(0), -1, -1)

    log_det_similarity = torch.logdet(similarity_matrix + identity)

    # Handle potential NaN from logdet (e.g., non-positive definite matrix)
    if similarity_matrix.dim() == 3:
        log_det_similarity = torch.where(
            torch.isnan(log_det_similarity) | torch.isinf(log_det_similarity),
            torch.zeros_like(log_det_similarity),
            log_det_similarity
        )
        log_det_similarity = log_det_similarity.mean()
    else:
        if torch.isnan(log_det_similarity) or torch.isinf(log_det_similarity):
            log_det_similarity = torch.tensor(0.0, device=item_embeddings.device)

    # Normalize log det by candidate size to prevent scale explosion
    group_size = similarity_matrix.size(-1)
    log_det_similarity = log_det_similarity / group_size

    # Mean of predicted scores instead of sum to prevent scale explosion
    r_ui_mean = torch.mean(y_pred)

    # Diversity Loss Calculation
    diversity_loss = theta * r_ui_mean + (1 - theta) * log_det_similarity

    return diversity_loss


def compute_item_similarity_matrix(
    item_embeddings: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute item-item similarity matrix from embeddings.

    Args:
        item_embeddings: Item embedding tensor of shape (batch_size, embedding_dim)
        normalize: Whether to L2-normalize embeddings before computing similarity

    Returns:
        Similarity matrix of shape (batch_size, batch_size) with values in [0, 1]
    """
    if normalize:
        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)

    cosine_similarity = torch.matmul(item_embeddings, item_embeddings.t())

    # Transform to [0, 1] range
    similarity_matrix = (1 + cosine_similarity) / 2

    return similarity_matrix


def compute_diversity_loss_per_user(
    model,
    pos_inputs: dict,
    neg_inputs: dict,
    pos_scores: torch.Tensor,
    neg_scores_flat: torch.Tensor,
    num_negatives: int,
    theta: float = 0.7,
    lambda_: float = 0.01,
    eps: float = 1e-6,
    kernel: str = 'gram',
    gamma: float = 1.0,
    scores_are_logits: bool = True,
) -> torch.Tensor:
    """
    Compute per-user diversity loss treating each sample's pos+neg as an impression.

    Instead of computing a single huge similarity matrix across the entire batch,
    this function reshapes the data into per-user groups of size (1 + num_negatives)
    and computes diversity within each user's impression.

    Args:
        model: The model instance (for extracting item embeddings)
        pos_inputs: Positive example batch dict (batch_size B)
        neg_inputs: Negative example batch dict (batch_size B * num_negatives)
        pos_scores: Positive prediction scores [B, 1]
        neg_scores_flat: Negative prediction scores [B * num_neg, 1]
        num_negatives: Number of negatives per positive
        theta: Weight between prediction sum and diversity term (default: 0.7)
        lambda_: Weight of diversity loss in total loss (default: 0.01)
        eps: Small epsilon for numerical stability (default: 1e-6)
        scores_are_logits: Whether pairwise scores are raw logits and need a
                            sigmoid before the prediction-mean term.

    Returns:
        Diversity loss delta to ADD to the base loss (negative value encourages diversity).
    """
    batch_size = pos_scores.size(0)
    group_size = 1 + num_negatives  # items per user

    # --- Extract item embeddings ---
    item_embeddings = _extract_item_embeddings_for_diversity(model, pos_inputs, neg_inputs, num_negatives)

    if item_embeddings is None:
        return torch.tensor(0.0, device=pos_scores.device)

    # item_embeddings shape: [B * group_size, emb_dim]
    emb_dim = item_embeddings.size(-1)

    # Reshape to per-user groups: [B, group_size, emb_dim]
    item_embeddings_grouped = item_embeddings.view(batch_size, group_size, emb_dim)

    # Normalize embeddings for similarity metric computation
    item_emb_norm = torch.nn.functional.normalize(item_embeddings_grouped, p=2, dim=-1)

    # Per-user cosine similarities: [B, group_size, group_size]
    cos_sim = torch.bmm(item_emb_norm, item_emb_norm.transpose(-1, -2))

    if kernel == 'rbf':
        # Fast DPP style RBF Kernel: exp(-gamma * ||x_i - x_j||^2)
        dist_sq = 2.0 - 2.0 * cos_sim
        sim_matrices = torch.exp(-gamma * dist_sq)
    elif kernel == 'gram':
        # Literature Fast DPP: L_ij = r_i r_j <f_i, f_j>
        # all_scores captures [pos_score, neg_score1, ...]
        neg_scores_grouped = neg_scores_flat.view(batch_size, num_negatives)  # [B, num_neg]
        all_scores = torch.cat([pos_scores, neg_scores_grouped], dim=1)  # [B, group_size]

        # Ensure scores are positive to act as scaling factors (r_i > 0)
        # Assuming logits are usually passed, we transform to [0, 1] via sigmoid
        r = torch.sigmoid(all_scores).unsqueeze(-1) # [B, group_size, 1]
        r_matrix = torch.bmm(r, r.transpose(-1, -2)) # [B, group_size, group_size]

        sim_matrices = r_matrix * cos_sim
    else:
        # Map to [0, 1]: (1 + cos_sim) / 2
        sim_matrices = (1 + cos_sim) / 2

    # Add small identity for numerical stability
    identity = torch.eye(group_size, device=sim_matrices.device) * eps
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

    # Per-user log-determinant: [B]
    log_det = torch.logdet(sim_matrices + identity)

    # Handle NaN/Inf (replace with 0)
    log_det = torch.where(
        torch.isnan(log_det) | torch.isinf(log_det),
        torch.zeros_like(log_det),
        log_det
    )

    # Normalize log det by candidate size to prevent gradient explosion
    log_det = log_det / group_size

    # Per-user prediction mean instead of sum
    # pos_scores: [B, 1], neg_scores_flat: [B * num_neg, 1]
    neg_scores_grouped = neg_scores_flat.view(batch_size, num_negatives)  # [B, num_neg]
    all_scores = torch.cat([pos_scores, neg_scores_grouped], dim=1)  # [B, group_size]
    # Pairwise training normally passes raw logits (``use_logit=True``), while
    # the prediction component of diversity is defined on probabilities.
    # Keep an explicit switch so callers that already pass probabilities do
    # not apply sigmoid twice.
    score_values = torch.sigmoid(all_scores) if scores_are_logits else all_scores
    r_ui_mean = score_values.mean(dim=1)  # [B]

    lambda_ = float(lambda_)
    theta = float(theta)

    # Per-user diversity loss: [B]
    per_user_div = theta * r_ui_mean + (1 - theta) * log_det

    # Average across users, then apply as regularization (negative = encourage diversity)
    diversity_loss = per_user_div.mean()

    return -lambda_ * diversity_loss


def _extract_item_embeddings_for_diversity(model, pos_inputs, neg_inputs, num_negatives):
    """
    Extract and concatenate item embeddings from positive and negative inputs.

    Supports both wrapper-based and mixin-based models.

    Args:
        model: The model instance
        pos_inputs: Positive batch dict (B samples)
        neg_inputs: Negative batch dict (B * num_neg samples)
        num_negatives: Number of negatives per positive

    Returns:
        Concatenated item embeddings [B * (1 + num_neg), emb_dim] or None
    """
    # Determine which layer and features to use
    emb_dict_layer = None
    item_features = []

    # Wrapper-based model
    if getattr(model, '_diversity_enabled', False):
        emb_dict_layer = getattr(model, '_diversity_emb_dict_layer', None)
        item_features = getattr(model, '_diversity_item_features', [])

    # Mixin-based model
    if not item_features and hasattr(model, '_use_diversity_loss'):
        item_features = getattr(model, '_diversity_item_features', [])
        if not item_features:
            item_features = _detect_item_features(model)
        emb_dict_layer = _find_embedding_dict_layer(model)

    if emb_dict_layer is None or not item_features:
        return None

    try:
        # Extract positive item embeddings
        X_pos = model.get_inputs(pos_inputs)
        feat_emb_pos = emb_dict_layer(X_pos)

        # Extract negative item embeddings
        X_neg = model.get_inputs(neg_inputs)
        feat_emb_neg = emb_dict_layer(X_neg)

        # Collect item feature embeddings
        pos_embs = []
        neg_embs = []
        for feat_name in item_features:
            if feat_name in feat_emb_pos:
                emb_p = feat_emb_pos[feat_name]
                emb_n = feat_emb_neg[feat_name]
                if emb_p.dim() == 3:
                    emb_p = emb_p.mean(dim=1)
                if emb_n.dim() == 3:
                    emb_n = emb_n.mean(dim=1)
                pos_embs.append(emb_p)
                neg_embs.append(emb_n)

        if not pos_embs:
            return None

        pos_item_emb = torch.cat(pos_embs, dim=-1)  # [B, D]
        neg_item_emb = torch.cat(neg_embs, dim=-1)  # [B * num_neg, D]

        batch_size = pos_item_emb.size(0)
        emb_dim = pos_item_emb.size(-1)

        # Reshape neg: [B * num_neg, D] -> [B, num_neg, D]
        neg_item_emb_grouped = neg_item_emb.view(batch_size, num_negatives, emb_dim)

        # Concat: [B, 1, D] + [B, num_neg, D] -> [B, 1+num_neg, D]
        combined = torch.cat([pos_item_emb.unsqueeze(1), neg_item_emb_grouped], dim=1)

        # Flatten back: [B * (1 + num_neg), D]
        return combined.view(-1, emb_dim)

    except Exception as e:
        logger.debug(f"Failed to extract item embeddings for per-user diversity: {e}")
        return None


class DiversityLossMixin:
    """
    Mixin class to add diversity loss capability to any model.

    This mixin can be added to any model inheriting from BaseModel to enable
    diversity-aware training. It provides:
    - Configuration parameters for diversity loss
    - Methods to compute item embeddings and similarity matrix
    - Override of compute_loss to include diversity regularization

    Usage:
        class MyModel(DiversityLossMixin, BaseModel):
            def __init__(self, ..., use_diversity_loss=False, **kwargs):
                # Initialize DiversityLossMixin first
                DiversityLossMixin.__init__(
                    self,
                    use_diversity_loss=use_diversity_loss,
                    diversity_lambda=kwargs.pop('diversity_lambda', 0.7),
                    diversity_theta=kwargs.pop('diversity_theta', 0.7),
                )
                # Then initialize BaseModel
                BaseModel.__init__(self, ...)

            def get_item_embeddings(self, feat_emb_dict):
                # Return item embeddings for diversity calculation
                return feat_emb_dict['item_id']
    """

    def __init__(
        self,
        use_diversity_loss: bool = False,
        diversity_lambda: float = 0.7,
        diversity_theta: float = 0.7,
        diversity_kernel: str = 'cosine',
        diversity_gamma: float = 1.0,
        diversity_item_features: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize diversity loss parameters.

        Args:
            use_diversity_loss: Whether to use diversity loss
            diversity_lambda: Weight of diversity loss in total loss (default: 0.7)
            diversity_theta: Weight between prediction sum and diversity term (default: 0.7)
            diversity_item_features: List of feature names to use for item embeddings.
                                    If None, subclass must implement get_diversity_item_embeddings()
        """
        self._use_diversity_loss = use_diversity_loss
        self._diversity_lambda = diversity_lambda
        self._diversity_theta = diversity_theta
        self._diversity_kernel = diversity_kernel
        self._diversity_gamma = diversity_gamma
        self._diversity_item_features = diversity_item_features or []
        self._diversity_logger = logging.getLogger(self.__class__.__name__)

    @property
    def use_diversity_loss(self) -> bool:
        return self._use_diversity_loss

    @use_diversity_loss.setter
    def use_diversity_loss(self, value: bool):
        self._use_diversity_loss = value

    @property
    def diversity_lambda(self) -> float:
        return self._diversity_lambda

    @property
    def diversity_theta(self) -> float:
        return self._diversity_theta

    def get_diversity_item_embeddings(
        self,
        feat_emb_dict: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Extract item embeddings for diversity calculation.

        Override this method if you need custom logic for extracting item embeddings.

        Args:
            feat_emb_dict: Dictionary of feature embeddings from embedding layer

        Returns:
            Concatenated item embeddings tensor or None if not available
        """
        if not self._diversity_item_features:
            return None

        item_feature_embs = []
        for feature_name in self._diversity_item_features:
            if feature_name in feat_emb_dict:
                item_feature_embs.append(feat_emb_dict[feature_name])

        if not item_feature_embs:
            return None

        return torch.cat(item_feature_embs, dim=-1)

    def compute_diversity_regularization(
        self,
        feat_emb_dict: Dict[str, torch.Tensor],
        y_pred: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute the diversity regularization term.

        Args:
            feat_emb_dict: Dictionary of feature embeddings
            y_pred: Prediction scores

        Returns:
            Diversity loss value or None if diversity loss is disabled/unavailable
        """
        if not self._use_diversity_loss:
            return None

        item_embeddings = self.get_diversity_item_embeddings(feat_emb_dict)
        if item_embeddings is None:
            return None

        return compute_diversity_loss(
            item_embeddings=item_embeddings,
            y_pred=y_pred,
            theta=self._diversity_theta,
            kernel=self._diversity_kernel,
            gamma=self._diversity_gamma,
        )

    def add_diversity_to_loss(
        self,
        base_loss: torch.Tensor,
        diversity_loss: Optional[torch.Tensor],
        log_losses: bool = False,
    ) -> torch.Tensor:
        """
        Combine base loss with diversity regularization.

        Args:
            base_loss: The base training loss (e.g., BCE)
            diversity_loss: The diversity loss term (can be None)
            log_losses: Whether to log loss components

        Returns:
            Total loss with diversity regularization
        """
        if diversity_loss is None:
            return base_loss

        total_loss = base_loss - self._diversity_lambda * diversity_loss

        if log_losses:
            self._diversity_logger.info(
                f"Base Loss: {base_loss.item():.6f}, "
                f"Diversity Loss: {diversity_loss.item():.6f}, "
                f"Total Loss: {total_loss.item():.6f}"
            )

        return total_loss


def _find_embedding_dict_layer(model):
    """
    Find the FeatureEmbeddingDict layer in a model.

    Model_zoo models use FeatureEmbedding which wraps FeatureEmbeddingDict:
        model.embedding_layer (FeatureEmbedding)
            .embedding_layer (FeatureEmbeddingDict)

    Cloud_device_recsys models use FeatureEmbeddingDict directly:
        model.embedding_layer (FeatureEmbeddingDict)

    Returns:
        The FeatureEmbeddingDict instance, or None if not found
    """
    from fuxictr.pytorch.layers import FeatureEmbeddingDict, FeatureEmbedding

    if hasattr(model, 'embedding_layer'):
        emb_layer = model.embedding_layer
        # Case 1: model_zoo pattern — FeatureEmbedding wrapping FeatureEmbeddingDict
        if isinstance(emb_layer, FeatureEmbedding) and hasattr(emb_layer, 'embedding_layer'):
            inner = emb_layer.embedding_layer
            if isinstance(inner, FeatureEmbeddingDict):
                return inner
        # Case 2: direct FeatureEmbeddingDict
        if isinstance(emb_layer, FeatureEmbeddingDict):
            return emb_layer

    # Case 3: search all submodules
    for module in model.modules():
        if isinstance(module, FeatureEmbeddingDict):
            return module

    return None


def _detect_item_features(model):
    """
    Auto-detect item feature names from feature_map.

    Detection priority:
    1. feature_cols with feature_group == 'FG1' from dataset_config
    2. dataset_config["item_id_col"] and its shared embeddings
    3. Fallback: features with 'item' in the name

    Returns:
        List of item feature names (only those present in feature_map.features)
    """
    feature_map = model.feature_map
    available_features = set(feature_map.features.keys())

    # --- Strategy 1: Use feature_group FG1 from dataset_config ---
    dataset_config = getattr(feature_map, 'dataset_config', None) or {}
    feature_cols = dataset_config.get('feature_cols', [])

    if feature_cols:
        fg1_features = []
        for feat_def in feature_cols:
            if feat_def.get('feature_group') != 'FG1':
                continue
            names = feat_def.get('name', [])
            if isinstance(names, str):
                names = [names]
            for name in names:
                if name in available_features:
                    fg1_features.append(name)

        if fg1_features:
            logger.info(f"Detected FG1 item features: {fg1_features}")
            return fg1_features

    # --- Strategy 2: Use item_id_col from dataset_config ---
    item_id_col = dataset_config.get("item_id_col")
    if item_id_col and item_id_col in available_features:
        item_features = [item_id_col]
        # Also include features that share embedding with item_id
        for name, spec in feature_map.features.items():
            if spec.get('share_embedding') == item_id_col and name != item_id_col:
                item_features.append(name)
        logger.info(f"Detected item features from item_id_col: {item_features}")
        return item_features

    # --- Strategy 3: Fallback heuristic ---
    item_features = []
    for name, spec in feature_map.features.items():
        if 'item' in name.lower() and spec.get('type') in ('categorical', 'sequence'):
            item_features.append(name)

    if item_features:
        logger.info(f"Detected item features by name heuristic: {item_features}")
    else:
        logger.warning("No item features detected by any strategy")

    return item_features



def compute_diversity_for_pairwise(model, inputs, y_pred):
    """
    Compute diversity loss for pairwise training loops.

    Works with both wrapper-based models (from wrap_model_with_diversity)
    and Mixin-based models (DiversityLossMixin).

    This is designed for custom training loops (negative sampling) that
    bypass model.compute_loss() and compute pairwise loss directly.

    Args:
        model: The model instance
        inputs: The positive example batch (for embedding extraction)
        y_pred: Positive prediction scores [B, 1]

    Returns:
        Diversity loss delta to ADD to the base loss (typically negative).
        Returns 0.0 (as tensor) if diversity is disabled or unavailable.
    """
    import torch

    # --- Path 1: Wrapper-based model ---
    if getattr(model, '_diversity_enabled', False):
        emb_dict_layer = getattr(model, '_diversity_emb_dict_layer', None)
        item_features = getattr(model, '_diversity_item_features', [])

        if emb_dict_layer is not None and item_features:
            try:
                X = model.get_inputs(inputs)
                feat_emb_dict = emb_dict_layer(X)

                item_embs = []
                for feat_name in item_features:
                    if feat_name in feat_emb_dict:
                        emb = feat_emb_dict[feat_name]
                        if emb.dim() == 3:
                            emb = emb.mean(dim=1)
                        item_embs.append(emb)

                if item_embs:
                    item_embeddings = torch.cat(item_embs, dim=-1)
                    div_loss = compute_diversity_loss(
                        item_embeddings=item_embeddings,
                        y_pred=y_pred,
                        theta=model._diversity_theta,
                        kernel=getattr(model, '_diversity_kernel', 'cosine'),
                        gamma=getattr(model, '_diversity_gamma', 1.0),
                    )
                    return -model._diversity_lambda * div_loss
            except Exception as e:
                logger.debug(f"Failed to compute diversity in pairwise: {e}")

        return torch.tensor(0.0, device=y_pred.device)

    # --- Path 2: Mixin-based model ---
    if hasattr(model, '_use_diversity_loss') and model._use_diversity_loss:
        # Need feat_emb_dict from the model's last forward pass
        feat_emb_dict = getattr(model, '_last_feat_emb_dict', None)

        # If not cached, try to get from forward output
        if feat_emb_dict is None:
            # Try to extract via the model's embedding layer
            try:
                X = model.get_inputs(inputs)
                if hasattr(model, 'embedding_layer'):
                    emb_layer = model.embedding_layer
                    if hasattr(emb_layer, 'dict_forward'):
                        feat_emb_dict = emb_layer.dict_forward(X)
                    elif hasattr(emb_layer, 'embedding_layer') and hasattr(emb_layer.embedding_layer, '__call__'):
                        feat_emb_dict = emb_layer.embedding_layer(X)
            except Exception:
                pass

        if feat_emb_dict is not None:
            div_loss = model.compute_diversity_regularization(feat_emb_dict, y_pred)
            if div_loss is not None:
                return model.add_diversity_to_loss(torch.tensor(0.0, device=y_pred.device), div_loss)

        return torch.tensor(0.0, device=y_pred.device)

    # No diversity loss configured
    return torch.tensor(0.0, device=y_pred.device)


def wrap_model_with_diversity(
    model,
    use_diversity_loss: bool = True,
    diversity_lambda: float = 0.7,
    diversity_theta: float = 0.7,
    diversity_kernel: str = 'cosine',
    diversity_gamma: float = 1.0,
    diversity_item_features: Optional[List[str]] = None,
):
    """
    Wrap any BaseModel to add diversity loss support.

    Works by overriding forward() and compute_loss() methods on the model
    instance (monkey-patching), without changing the model class.

    This allows any model_zoo model (DNN, DeepFM, DCN, etc.) to use
    diversity loss by simply adding configuration parameters.

    Args:
        model: A BaseModel instance (from model_zoo or cloud_device_recsys)
        use_diversity_loss: Whether to enable diversity loss
        diversity_lambda: Weight of diversity loss in total loss
        diversity_theta: Weight between prediction sum and diversity term
        diversity_item_features: List of feature names for item embeddings.
                                Auto-detected from feature_map if None.

    Returns:
        The same model instance, with patched methods

    Example:
        >>> model = DNN(feature_map, **params)
        >>> model = wrap_model_with_diversity(model, diversity_lambda=0.5)
    """
    if not use_diversity_loss:
        return model

    _logger = logging.getLogger("DiversityWrapper")

    # Find the FeatureEmbeddingDict layer
    emb_dict_layer = _find_embedding_dict_layer(model)
    if emb_dict_layer is None:
        _logger.warning(
            "Could not find FeatureEmbeddingDict in model. "
            "Diversity loss will be disabled."
        )
        return model

    # Detect or validate item features
    if diversity_item_features is None:
        diversity_item_features = _detect_item_features(model)

    if not diversity_item_features:
        _logger.warning(
            "No item features found for diversity loss. "
            "Please specify diversity_item_features explicitly."
        )
        return model

    _logger.info(
        f"Wrapping model with diversity loss: "
        f"lambda={diversity_lambda}, theta={diversity_theta}, "
        f"item_features={diversity_item_features}"
    )

    # Store diversity config on the model
    model._diversity_enabled = True
    model._diversity_lambda = diversity_lambda
    model._diversity_theta = diversity_theta
    model._diversity_kernel = diversity_kernel
    model._diversity_gamma = diversity_gamma
    model._diversity_item_features = diversity_item_features
    model._diversity_emb_dict_layer = emb_dict_layer
    model._diversity_item_embeddings = None  # populated during forward

    # Save original methods
    original_forward = model.forward
    original_compute_loss = model.compute_loss

    def patched_forward(inputs):
        """Forward pass with item embedding extraction for diversity loss."""
        # Call original forward
        return_dict = original_forward(inputs)

        # Optimization: Skip extraction during inference or when diversity is disabled
        if not getattr(model, '_diversity_enabled', True) or not model.training:
            model._diversity_item_embeddings = None
            return return_dict

        # Extract item embeddings via the FeatureEmbeddingDict layer
        # We need to get the input features (X) and pass through the dict layer
        try:
            X = model.get_inputs(inputs)
            feat_emb_dict = emb_dict_layer(X)

            # Collect item feature embeddings
            item_embs = []
            for feat_name in diversity_item_features:
                if feat_name in feat_emb_dict:
                    emb = feat_emb_dict[feat_name]
                    # Handle sequence features: use mean pooling
                    if emb.dim() == 3:
                        emb = emb.mean(dim=1)
                    item_embs.append(emb)

            if item_embs:
                model._diversity_item_embeddings = torch.cat(item_embs, dim=-1)
            else:
                model._diversity_item_embeddings = None
        except Exception as e:
            _logger.debug(f"Failed to extract item embeddings: {e}")
            model._diversity_item_embeddings = None

        return return_dict

    def patched_compute_loss(return_dict, y_true):
        """Compute loss with diversity regularization."""
        # Get base loss from original
        base_loss = original_compute_loss(return_dict, y_true)

        # Add diversity regularization
        item_embeddings = model._diversity_item_embeddings
        if item_embeddings is not None:
            y_pred = return_dict["y_pred"]
            div_loss = compute_diversity_loss(
                item_embeddings=item_embeddings,
                y_pred=y_pred,
                theta=diversity_theta,
                kernel=diversity_kernel,
                gamma=diversity_gamma,
            )
            total_loss = base_loss - diversity_lambda * div_loss
            return total_loss

        return base_loss

    # Apply patches using types.MethodType to properly bind methods
    import types
    model.forward = types.MethodType(lambda self, inputs: patched_forward(inputs), model)
    model.compute_loss = types.MethodType(
        lambda self, return_dict, y_true: patched_compute_loss(return_dict, y_true), model
    )

    return model
