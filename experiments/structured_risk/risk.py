"""Canonical binary-Dirichlet risk primitives for isolated experiments.

Provides risk calculations for training experiments and CPU numerical tests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


EvidenceKind = Literal["exp", "relu", "softplus"]


def evidence_from_logits(logits: Tensor, kind: EvidenceKind = "exp") -> Tensor:
    """Map final-axis binary logits to finite non-negative evidence."""

    if kind == "exp":
        return torch.exp(logits.clamp(-10.0, 10.0))
    if kind == "relu":
        return F.relu(logits)
    if kind == "softplus":
        return F.softplus(logits)
    raise ValueError(f"unsupported evidence kind: {kind!r}")


def _check_binary(evidence: Tensor) -> None:
    if evidence.ndim == 0 or evidence.shape[-1] != 2:
        raise ValueError(f"expected evidence[..., 2], got {tuple(evidence.shape)}")
    if not torch.isfinite(evidence).all():
        raise ValueError("evidence must be finite")
    if (evidence < 0).any():
        raise ValueError("evidence must be non-negative")


def beta_risk_from_evidence(
    evidence: Tensor,
    *,
    prior_strength: float = 1.0,
) -> dict[str, Tensor]:
    """Return standard binary-Dirichlet means and variance components.

    For ``alpha=e+lambda`` and ``S=sum(alpha)``, the predictive mean is
    ``mu=alpha_pos/S``.  ``epistemic_variance`` is the variance of the random
    categorical probability under the Dirichlet/Beta opinion; ``aleatoric``
    and ``total`` are its standard two-term decomposition.
    """

    _check_binary(evidence)
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    prior = torch.as_tensor(prior_strength, dtype=evidence.dtype, device=evidence.device)
    alpha = evidence + prior
    strength = alpha.sum(dim=-1)
    probability = alpha[..., 0] / strength
    total = probability * (1.0 - probability)
    epistemic = total / (strength + 1.0)
    aleatoric = strength / (strength + 1.0) * total
    vacuity = (2.0 * prior) / strength
    return {
        "alpha": alpha,
        "strength": strength,
        "probability": probability,
        "vacuity": vacuity,
        "total_variance": total,
        "epistemic_variance": epistemic,
        "aleatoric_variance": aleatoric,
    }


def beta_risk_from_logits(
    logits: Tensor,
    kind: EvidenceKind = "exp",
    *,
    prior_strength: float = 1.0,
) -> dict[str, Tensor]:
    """Compose evidence mapping and :func:`beta_risk_from_evidence`."""

    return beta_risk_from_evidence(
        evidence_from_logits(logits, kind), prior_strength=prior_strength
    )


def beta_total_variance_gate(
    evidence: Tensor,
    *,
    prior_strength: float = 1.0,
) -> Tensor:
    """Return a bounded ``[0, 1]`` risk gate from a strict Beta opinion.

    The predictive variance of a Bernoulli probability is
    ``mu * (1 - mu)`` and is maximized at ``0.25``.  Dividing by this
    maximum makes the quantity directly usable as an IPR risk gate while
    retaining the evidence-derived Beta mean.  This helper intentionally
    does *not* alter the pseudo-label probability used by the caller.
    """

    out = beta_risk_from_evidence(evidence, prior_strength=prior_strength)
    return (out["total_variance"] / 0.25).clamp(0.0, 1.0)


def empirical_cdf(values: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Return deterministic mid-rank ECDF values, preserving input container."""

    is_tensor = isinstance(values, Tensor)
    array = values.detach().cpu().numpy() if is_tensor else np.asarray(values)
    flat = array.reshape(-1)
    if flat.size == 0:
        result = np.empty_like(flat, dtype=np.float64).reshape(array.shape)
    else:
        order = np.argsort(flat, kind="mergesort")
        sorted_values = flat[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
        ends = np.r_[starts[1:], flat.size]
        ranks = np.empty(flat.size, dtype=np.float64)
        for start, end in zip(starts, ends):
            ranks[start:end] = (start + end + 1.0) / 2.0
        inverse = np.empty_like(order)
        inverse[order] = np.arange(flat.size)
        ranks = ranks[inverse] / flat.size
        result = ranks.reshape(array.shape)
    if is_tensor:
        return torch.as_tensor(result, dtype=values.dtype, device=values.device)
    return result


def rank_normalize(values: Tensor, *, dim: int | None = None) -> Tensor:
    """Return deterministic empirical-CDF ranks on a tensor device.

    ``dim=None`` ranks all entries jointly.  With ``dim`` set, ranks are
    computed independently along that axis (for example, ``dim=0`` ranks
    donors within a mini-batch for every time/class coordinate).  Ties use a
    deterministic ordinal rank; this is equivalent to the empirical CDF up to
    the conventional mid-rank tie correction and avoids a CPU round trip
    inside the training loop.
    """

    if not isinstance(values, Tensor) or values.ndim == 0:
        raise ValueError("rank_normalize expects a non-scalar torch.Tensor")
    if not torch.isfinite(values).all():
        raise ValueError("rank_normalize expects finite values")
    if dim is None:
        flat = values.detach().reshape(-1)
        if flat.numel() <= 1:
            return torch.full_like(values, 0.5)
        # The pinned NREP environment predates the ``stable`` keyword; its
        # default argsort is deterministic for the finite tensors used here.
        order = torch.argsort(flat)
        ranks = torch.empty_like(order)
        ranks.scatter_(0, order, torch.arange(flat.numel(), device=values.device))
        return (ranks.to(values.dtype) / float(flat.numel() - 1)).reshape_as(values)

    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise ValueError(f"invalid rank dimension {dim} for shape {tuple(values.shape)}")
    length = values.shape[dim]
    if length <= 1:
        return torch.full_like(values, 0.5)
    order = torch.argsort(values.detach(), dim=dim)
    ordinal = torch.arange(length, device=values.device, dtype=values.dtype)
    shape = [1] * values.ndim
    shape[dim] = length
    ordinal = ordinal.reshape(shape).expand_as(values)
    ranks = torch.empty_like(ordinal)
    ranks.scatter_(dim, order, ordinal)
    return ranks / float(length - 1)


def aggregate_class_risk(
    risk: Tensor,
    *,
    mode: str = "class_mean",
    presence: Tensor | None = None,
    top_fraction: float = 0.2,
) -> Tensor:
    """Aggregate ``(B,C)`` risk while keeping empty-presence fallbacks explicit."""

    if risk.ndim != 2:
        raise ValueError(f"expected risk [B,C], got {tuple(risk.shape)}")
    if mode == "class_mean":
        return risk.mean(dim=-1)
    if mode == "donor_positive":
        if presence is None or presence.shape != risk.shape:
            raise ValueError("donor_positive requires presence with risk shape")
        mask = presence.to(dtype=risk.dtype)
        count = mask.sum(dim=-1)
        weighted = (risk * mask).sum(dim=-1) / count.clamp_min(1.0)
        return torch.where(count > 0, weighted, risk.mean(dim=-1))
    if mode == "topk_risk":
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        k = max(1, int(np.ceil(risk.shape[-1] * top_fraction)))
        return risk.topk(k, dim=-1, largest=True).values.mean(dim=-1)
    raise ValueError(f"unsupported aggregation mode: {mode!r}")
