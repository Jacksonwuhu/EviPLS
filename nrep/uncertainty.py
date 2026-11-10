"""Evidence uncertainty and uncertainty-guided hard-negative mining."""

from typing import Literal

import torch
import torch.nn.functional as F

EvidenceType = Literal["exp", "relu", "softplus"]


def evidence_from_logits(logits: torch.Tensor, kind: EvidenceType = "exp") -> torch.Tensor:
    """Convert logits to non-negative evidence using a numerically safe mapping."""
    if kind == "exp":
        return torch.exp(torch.clamp(logits, -10, 10))
    if kind == "relu":
        return F.relu(logits)
    if kind == "softplus":
        return F.softplus(logits)
    raise ValueError(f"Unsupported evidence function: {kind!r}")


def compute_uncertainty(
    logits: torch.Tensor,
    evidence_type: EvidenceType = "exp",
    *,
    num_classes: int = 2,
    prior_strength: float = 1.0,
) -> torch.Tensor:
    """Compute subjective-logic uncertainty along the final class dimension."""
    if logits.ndim == 0 or logits.shape[-1] != num_classes:
        raise ValueError(
            f"Expected logits[..., {num_classes}], got shape {tuple(logits.shape)}"
        )
    evidence = evidence_from_logits(logits, evidence_type)
    strength = evidence.sum(dim=-1) + prior_strength
    return num_classes / strength.clamp_min(torch.finfo(evidence.dtype).eps)


def hard_negative_mining(
    features: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    min_uncertainty: float = 0.01,
) -> torch.Tensor:
    """Select a similar, uncertain peer for every item in a batch.

    A singleton batch maps to itself. This makes evaluation and small final
    training batches safe instead of selecting from an all-``-inf`` matrix.
    """
    if features.ndim < 2:
        raise ValueError(f"Expected batched features, got shape {tuple(features.shape)}")
    batch_size = features.shape[0]
    if uncertainty.numel() != batch_size:
        raise ValueError(
            f"Expected one uncertainty score per sample ({batch_size}), "
            f"got {uncertainty.numel()}"
        )
    if batch_size < 2:
        return torch.arange(batch_size, device=features.device)

    pooled = features.mean(dim=tuple(range(1, features.ndim - 1))) if features.ndim > 2 else features
    normalized = F.normalize(pooled, p=2, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity.fill_diagonal_(-torch.inf)
    score = similarity * uncertainty.reshape(1, -1).clamp_min(min_uncertainty)
    return score.argmax(dim=1)
