"""Run bounded, validation-only scheduler screens on the legacy paper pipeline.

This module intentionally leaves :mod:`main_ipr` unchanged.  It delegates the
historical random/similarity/U-HNM controls to the original trainer and only
patches the training step for the two new controls:

* ``no_mix``: original pseudo-label objective without the auxiliary exchange.
* ``risk_anchor``: high-risk visual anchors paired with compatible low-risk
  audio donors, while keeping the auxiliary batch size and loss coefficients
  identical to the historical exchange path.
* ``random_consistency``: historical random audio-track exchange plus a
  confidence-gated retained-visual prediction consistency objective.
* ``uncertainty_consistency``: the same random intervention, with Jensen--
  Shannon consistency restricted to uncertain-yet-learnable visual positions.
* ``xor_uhnm`` / ``xor_random``: supervise exchanged class positions only when
  donor audio and anchor visual presence disagree, retaining paired positive
  and complementary negative evidence.

The test-set evaluator is blocked by construction.  Model selection and final
comparison therefore use AVVP validation data only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main_ipr as legacy  # noqa: E402
from nets.criterion import EvidenceLoss, MutualLearningLoss  # noqa: E402
from nets.net_audiovisual import LabelSmoothingNCELoss  # noqa: E402
from nrep.uncertainty import compute_uncertainty, hard_negative_mining  # noqa: E402
from experiments.structured_risk.risk import (  # noqa: E402
    aggregate_class_risk,
    beta_risk_from_logits,
    empirical_cdf,
    rank_normalize,
)
from tqdm import tqdm  # noqa: E402


_SCHEDULER = "uhnm"
_MIX_FRACTION = 0.5
_CONSISTENCY_WEIGHT = 0.1
_DONOR_CAPACITY = 4
_TEMPORAL_TOP_K = 3
_BAND_MU = 0.78
_BAND_SIGMA = 0.18
_RISK_PROXY = "edl"
_MIX_RISK_SIDE = "high"
_DIAGNOSTICS_PATH: Path | None = None
_TEST_LABEL: Path | None = None
_TEST_ATTEMPTS = 0
_ORIGINAL_EVAL = legacy.eval
_ORIGINAL_TRAIN = legacy.train_with_pseudo_labels


def presence_absence_masks(
    donor_audio_presence: torch.Tensor,
    anchor_visual_presence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return class masks whose counterfactual supervision is well-defined."""
    if donor_audio_presence.shape != anchor_visual_presence.shape:
        raise ValueError(
            "PA routing expects matching [B,C] presence tensors, got "
            f"{tuple(donor_audio_presence.shape)} and "
            f"{tuple(anchor_visual_presence.shape)}"
        )
    if donor_audio_presence.ndim != 2:
        raise ValueError(
            "PA routing expects video-class presence tensors [B,C], got "
            f"{tuple(donor_audio_presence.shape)}"
        )
    donor_audio = (donor_audio_presence > 0.5).to(donor_audio_presence.dtype)
    anchor_visual = (anchor_visual_presence > 0.5).to(anchor_visual_presence.dtype)
    return donor_audio * (1.0 - anchor_visual), anchor_visual * (1.0 - donor_audio)


def presence_absence_xor_mask(
    donor_audio_presence: torch.Tensor,
    anchor_visual_presence: torch.Tensor,
) -> torch.Tensor:
    """Select disagreement classes with paired positive/negative supervision."""
    if donor_audio_presence.shape != anchor_visual_presence.shape:
        raise ValueError(
            "XOR routing expects matching [B,C] presence tensors, got "
            f"{tuple(donor_audio_presence.shape)} and "
            f"{tuple(anchor_visual_presence.shape)}"
        )
    if donor_audio_presence.ndim != 2:
        raise ValueError(
            "XOR routing expects video-class presence tensors [B,C], got "
            f"{tuple(donor_audio_presence.shape)}"
        )
    donor_audio = donor_audio_presence > 0.5
    anchor_visual = anchor_visual_presence > 0.5
    return torch.logical_xor(donor_audio, anchor_visual).to(
        donor_audio_presence.dtype
    )


def masked_evidence_bce(
    criterion: EvidenceLoss,
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Match legacy EvidenceLoss BCE while reducing over selected classes."""
    if criterion.loss_type != "bce" or criterion.uncertain:
        raise ValueError(
            "PA routing currently supports the reproduced legacy BCE contract only"
        )
    if logits.shape[:-1] != target.shape or target.shape != mask.shape:
        raise ValueError(
            "logits/target/mask contract mismatch: "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    probabilities = logits.softmax(dim=-1)
    binary_target = torch.stack((target, 1.0 - target), dim=-1)
    per_class = torch.nn.functional.binary_cross_entropy(
        probabilities, binary_target, reduction="none"
    ).mean(dim=-1)
    if criterion.brier_weight > 0:
        per_class = per_class + criterion.brier_weight * (
            probabilities[..., 0] - target
        ).square()
    weights = mask.to(per_class.dtype)
    return (per_class * weights).sum() / weights.sum().clamp_min(1.0)


def global_uhnm_assignment(
    features: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    min_uncertainty: float = 0.01,
    donor_capacity: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Maximize batch-level U-HNM score under bounded donor reuse.

    Historical U-HNM independently takes the best donor for every anchor, so
    many anchors can collapse onto the same item.  Here the same similarity ×
    uncertainty score is retained, but a capacitated linear assignment bounds
    each donor's reuse.  ``donor_capacity=1`` gives a one-to-one derangement;
    larger values trade coverage for harder donors.  No model, target, or
    mixed-loss semantics are changed.
    """
    if isinstance(donor_capacity, bool) or donor_capacity < 1:
        raise ValueError("donor_capacity must be a positive integer")
    if features.ndim < 2:
        raise ValueError(f"Expected batched features, got {tuple(features.shape)}")
    batch_size = features.shape[0]
    if uncertainty.numel() != batch_size:
        raise ValueError(
            f"Expected one uncertainty score per sample ({batch_size}), "
            f"got {uncertainty.numel()}"
        )
    if batch_size < 2:
        index = torch.arange(batch_size, device=features.device)
        return index, {
            "assignment_score": 0.0,
            "selected_similarity": 1.0 if batch_size else 0.0,
            "selected_uncertainty": (
                float(uncertainty.reshape(-1)[0].item()) if batch_size else 0.0
            ),
            "self_pair_rate": 1.0 if batch_size else 0.0,
            "unique_donor_fraction": 1.0 if batch_size else 0.0,
        }

    pooled = (
        features.mean(dim=tuple(range(1, features.ndim - 1)))
        if features.ndim > 2
        else features
    )
    normalized = torch.nn.functional.normalize(pooled.detach(), p=2, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    donor_uncertainty = uncertainty.detach().reshape(-1).clamp_min(min_uncertainty)
    score = similarity * donor_uncertainty.unsqueeze(0)
    score.fill_diagonal_(-torch.inf)
    donor_slots = torch.arange(batch_size, device=features.device).repeat_interleave(
        donor_capacity
    )
    expanded_score = score[:, donor_slots]
    rows, columns = linear_sum_assignment((-expanded_score).cpu().numpy())
    donor_idx = torch.empty(batch_size, dtype=torch.long, device=features.device)
    selected_slots = torch.as_tensor(columns, device=features.device)
    donor_idx[torch.as_tensor(rows, device=features.device)] = donor_slots[
        selected_slots
    ]
    anchors = torch.arange(batch_size, device=features.device)
    selected_similarity = similarity[anchors, donor_idx]
    selected_uncertainty = donor_uncertainty[donor_idx]
    selected_score = selected_similarity * selected_uncertainty
    donor_counts = torch.bincount(donor_idx, minlength=batch_size)
    diagnostics = {
        "assignment_score": float(selected_score.mean().item()),
        "selected_similarity": float(selected_similarity.mean().item()),
        "selected_uncertainty": float(selected_uncertainty.mean().item()),
        "self_pair_rate": float((anchors == donor_idx).float().mean().item()),
        "unique_donor_fraction": float(
            donor_idx.unique().numel() / max(batch_size, 1)
        ),
        "max_donor_reuse": float(donor_counts.max().item()),
        "donor_capacity": float(donor_capacity),
    }
    return donor_idx, diagnostics


def evidence_gated_visual_consistency(
    original_frame_logits: torch.Tensor,
    mixed_frame_logits: torch.Tensor,
    criterion_edl: EvidenceLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep retained visual predictions stable under random audio exchange.

    The original visual prediction is a stop-gradient teacher.  Its evidential
    certainty ``1-u`` continuously gates a foreground-probability MSE, so the
    experiment does not force uncertain predictions to become self-confirming.
    Both inputs must follow the NREP frame-logit contract ``[B,T,2,C,2]``.
    """
    expected = original_frame_logits.shape
    if original_frame_logits.ndim != 5 or expected[2] != 2 or expected[-1] != 2:
        raise ValueError(
            "Expected original_frame_logits [B,T,2,C,2], "
            f"got {tuple(expected)}"
        )
    if mixed_frame_logits.shape != expected:
        raise ValueError(
            "Mixed frame logits must match the original shape: "
            f"{tuple(mixed_frame_logits.shape)} != {tuple(expected)}"
        )

    original_visual = original_frame_logits[:, :, 1, :, :]
    mixed_visual = mixed_frame_logits[:, :, 1, :, :]
    original_probability, original_uncertainty = criterion_edl.get_predictions(
        original_visual
    )
    mixed_probability, _ = criterion_edl.get_predictions(mixed_visual)

    teacher_foreground = original_probability[..., 0].detach()
    student_foreground = mixed_probability[..., 0]
    certainty_gate = (1.0 - original_uncertainty.squeeze(-1).detach()).clamp(0.0, 1.0)
    squared_change = (student_foreground - teacher_foreground).square()
    weighted_loss = (certainty_gate * squared_change).sum() / certainty_gate.sum().clamp_min(1e-8)

    diagnostics = {
        "consistency_loss": float(weighted_loss.detach().item()),
        "raw_visual_sensitivity": float(squared_change.detach().mean().item()),
        "certainty_gate_mean": float(certainty_gate.mean().item()),
        "certainty_gate_gt_05": float((certainty_gate > 0.5).float().mean().item()),
    }
    return weighted_loss, diagnostics


def _within_sample_risk_rank(uncertainty: torch.Tensor) -> torch.Tensor:
    """Map each sample's segment-class uncertainty to deterministic [0, 1] ranks."""
    if uncertainty.ndim < 2:
        raise ValueError(
            "uncertainty must include a batch axis and at least one risk axis, "
            f"got {tuple(uncertainty.shape)}"
        )
    flat = uncertainty.detach().reshape(uncertainty.shape[0], -1)
    if flat.shape[1] == 1:
        return torch.full_like(uncertainty, 0.5)
    order = torch.argsort(flat, dim=1)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(flat.shape[1], device=flat.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return (ranks.to(flat.dtype) / float(flat.shape[1] - 1)).reshape_as(uncertainty)


def _rank_uhnm_pairs(
    features: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    min_uncertainty: float = 0.01,
    band_mu: float | None = None,
    band_sigma: float = 0.18,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select donors with a batch-empirical risk rank.

    This is the ordinal-risk version of historical U-HNM.  The donor score is
    ``cos(h_i,h_j) * r_j``; with ``band_mu`` set, ``r_j`` is replaced by a
    Gaussian band-pass gate centred on a learnable-but-not-extreme risk band.
    The helper keeps the original full-batch exchange contract and never
    changes the mixed-loss target semantics.
    """
    if features.ndim < 2:
        raise ValueError(f"Expected batched features, got {tuple(features.shape)}")
    batch_size = features.shape[0]
    if uncertainty.numel() != batch_size:
        raise ValueError(
            f"Expected one uncertainty score per sample ({batch_size}), "
            f"got {uncertainty.numel()}"
        )
    if band_mu is not None and not 0.0 <= band_mu <= 1.0:
        raise ValueError(f"band_mu must be in [0, 1], got {band_mu}")
    if band_sigma <= 0.0:
        raise ValueError(f"band_sigma must be positive, got {band_sigma}")
    if batch_size < 2:
        index = torch.arange(batch_size, device=features.device)
        return index, {
            "selected_similarity": 1.0 if batch_size else 0.0,
            "selected_risk_rank": 0.5 if batch_size else 0.0,
            "self_pair_rate": 1.0 if batch_size else 0.0,
            "unique_donor_fraction": 1.0 if batch_size else 0.0,
        }

    pooled = (
        features.mean(dim=tuple(range(1, features.ndim - 1)))
        if features.ndim > 2
        else features
    )
    normalized = torch.nn.functional.normalize(pooled.detach(), p=2, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    risk_rank = rank_normalize(uncertainty.detach().reshape(-1))
    if band_mu is None:
        risk_gate = risk_rank
    else:
        risk_gate = torch.exp(-0.5 * ((risk_rank - band_mu) / band_sigma).square())
    score = similarity * risk_gate.reshape(1, -1).clamp_min(min_uncertainty)
    score.fill_diagonal_(-torch.inf)
    donor_idx = score.argmax(dim=1)
    anchors = torch.arange(batch_size, device=features.device)
    diagnostics = {
        "selected_similarity": float(similarity[anchors, donor_idx].mean().item()),
        "selected_risk_rank": float(risk_rank[donor_idx].mean().item()),
        "selected_risk_gate": float(risk_gate[donor_idx].mean().item()),
        "risk_rank_mean": float(risk_rank.mean().item()),
        "self_pair_rate": float((anchors == donor_idx).float().mean().item()),
        "unique_donor_fraction": float(
            donor_idx.unique().numel() / max(batch_size, 1)
        ),
    }
    if band_mu is not None:
        diagnostics.update({"band_mu": float(band_mu), "band_sigma": float(band_sigma)})
    return donor_idx, diagnostics


def build_temporal_neighbor_map(
    segment_names: Iterable[str],
    *,
    top_k: int = 3,
    max_gap: float | None = None,
) -> dict[str, list[str]]:
    """Build same-video temporal-neighbor candidates from full segment names."""
    if isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    rows: list[tuple[str, str, float]] = []
    for raw in segment_names:
        name = str(raw)
        parts = name.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid segment identifier: {name!r}")
        try:
            start = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid segment start in {name!r}") from exc
        rows.append((name, name[:11], start))

    grouped: dict[str, list[tuple[str, float]]] = {}
    for name, video_id, start in rows:
        grouped.setdefault(video_id, []).append((name, start))
    result: dict[str, list[str]] = {}
    for name, video_id, start in rows:
        candidates = []
        for other, other_start in grouped[video_id]:
            if other == name:
                continue
            gap = abs(other_start - start)
            if max_gap is not None and gap > max_gap:
                continue
            candidates.append((gap, other))
        candidates.sort(key=lambda item: (item[0], item[1]))
        result[name] = [other for _, other in candidates[:top_k]]
    return result


def _build_global_memory_bank(
    model,
    train_loader,
    args,
) -> dict[str, object]:
    """Collect all train audio tracks, features, risks, labels and names."""
    was_training = model.training
    model.eval()
    audio_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    risk_rows: list[torch.Tensor] = []
    pa_rows: list[torch.Tensor] = []
    names: list[str] = []
    with torch.no_grad():
        for sample in train_loader:
            audio = sample["audio"].to("cuda")
            video = sample["video_s"].to("cuda")
            video_st = sample["video_st"].to("cuda")
            outputs = model(audio, video, video_st, with_ca=True, epoch_num=0)
            a_logits = outputs[1]
            uncertainty = compute_uncertainty(
                a_logits.detach(), args.evidence, prior_strength=1.0
            )
            audio_rows.append(audio.detach().cpu())
            feature_rows.append(audio.detach().mean(dim=1).cpu())
            risk_rows.append(
                uncertainty.detach().reshape(audio.shape[0], -1).mean(dim=1).cpu()
            )
            pa_rows.append(sample["Pa"].float().cpu())
            raw_names = sample.get("segment_name", sample.get("video_name"))
            if isinstance(raw_names, str):
                raw_names = [raw_names]
            names.extend(str(name) for name in raw_names)
    if not audio_rows:
        raise RuntimeError("Cannot build an empty global memory bank")
    bank_audio = torch.cat(audio_rows, dim=0).to("cuda", non_blocking=True)
    bank_features = torch.nn.functional.normalize(
        torch.cat(feature_rows, dim=0).to("cuda", non_blocking=True),
        p=2,
        dim=-1,
    )
    bank_risk = torch.cat(risk_rows, dim=0).to("cuda", non_blocking=True)
    bank_pa = torch.cat(pa_rows, dim=0).to("cuda", non_blocking=True)
    name_to_index = {name: index for index, name in enumerate(names)}
    temporal_map = build_temporal_neighbor_map(names, top_k=3)
    if was_training:
        model.train()
    return {
        "audio": bank_audio,
        "features": bank_features,
        "risk": bank_risk,
        "Pa": bank_pa,
        "names": names,
        "name_to_index": name_to_index,
        "temporal_map": temporal_map,
    }


def _global_memory_bank_pairs(
    audio: torch.Tensor,
    bank: dict[str, object],
    anchor_names: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Select globally similar, risk-weighted donors and return their tracks."""
    bank_features = bank["features"]
    bank_risk = bank["risk"]
    bank_audio = bank["audio"]
    bank_pa = bank["Pa"]
    names = bank["names"]
    if not isinstance(bank_features, torch.Tensor) or not isinstance(bank_risk, torch.Tensor):
        raise TypeError("memory bank tensors are missing")
    anchor_features = torch.nn.functional.normalize(audio.detach().mean(dim=1), p=2, dim=-1)
    similarity = anchor_features @ bank_features.transpose(0, 1)
    score = similarity * bank_risk.reshape(1, -1).clamp_min(0.01)
    anchor_names = [str(name) for name in anchor_names]
    for row, name in enumerate(anchor_names):
        own = bank["name_to_index"].get(name)
        if own is not None:
            score[row, int(own)] = -torch.inf
    donor_bank_idx = score.argmax(dim=1)
    selected_audio = bank_audio[donor_bank_idx]
    selected_pa = bank_pa[donor_bank_idx]
    selected_similarity = similarity[
        torch.arange(audio.shape[0], device=audio.device), donor_bank_idx
    ]
    diagnostics = {
        "memory_bank_size": float(len(names)),
        "selected_global_similarity": float(selected_similarity.mean().item()),
        "selected_global_risk": float(bank_risk[donor_bank_idx].mean().item()),
        "self_pair_rate": 0.0,
        "unique_donor_fraction": float(
            donor_bank_idx.unique().numel() / max(audio.shape[0], 1)
        ),
    }
    return selected_audio, selected_pa, donor_bank_idx, diagnostics


def _temporal_neighbor_pairs(
    audio: torch.Tensor,
    bank: dict[str, object],
    anchor_names: Iterable[str],
    *,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Select from top-k same-video temporal neighbors in the memory bank."""
    bank_features = bank["features"]
    bank_risk = bank["risk"]
    bank_audio = bank["audio"]
    bank_pa = bank["Pa"]
    temporal_map = bank["temporal_map"]
    name_to_index = bank["name_to_index"]
    anchor_names = [str(name) for name in anchor_names]
    anchor_features = torch.nn.functional.normalize(audio.detach().mean(dim=1), p=2, dim=-1)
    selected: list[int] = []
    covered = 0
    for row, name in enumerate(anchor_names):
        candidates = [
            name_to_index[candidate]
            for candidate in temporal_map.get(name, [])[:top_k]
            if candidate in name_to_index
        ]
        if candidates:
            covered += 1
            candidate_idx = torch.as_tensor(candidates, device=audio.device)
            similarity = anchor_features[row] @ bank_features[candidate_idx].transpose(0, 1)
            score = similarity * bank_risk[candidate_idx].clamp_min(0.01)
            selected.append(int(candidate_idx[score.argmax()].item()))
        else:
            selected.append(-1)
    fallback_audio, fallback_pa, fallback_idx, fallback_diag = _global_memory_bank_pairs(
        audio, bank, anchor_names
    )
    selected_idx = torch.as_tensor(selected, device=audio.device)
    valid = selected_idx >= 0
    selected_idx = torch.where(valid, selected_idx, fallback_idx)
    selected_audio = torch.where(
        valid[:, None, None], bank_audio[selected_idx], fallback_audio
    )
    selected_pa = torch.where(valid[:, None], bank_pa[selected_idx], fallback_pa)
    diagnostics = {
        **fallback_diag,
        "temporal_candidate_coverage": float(covered / max(len(anchor_names), 1)),
        "temporal_top_k": float(top_k),
        "temporal_fallback_fraction": float((~valid).float().mean().item()),
    }
    return selected_audio, selected_pa, selected_idx, diagnostics


def _class_aware_uhnm_pairs(
    audio: torch.Tensor,
    frame_logits: torch.Tensor,
    audio_pseudo_labels: torch.Tensor,
    visual_pseudo_labels: torch.Tensor,
    *,
    evidence_type: str = "exp",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select donors using anchor-conditioned, class-level ordinal risk.

    For anchor ``i`` and donor ``j`` the risk term is
    ``R_ij = sum_{t,c} w_{i,t,c} r_{j,t,c} / sum w_i``.  The active event
    positions of the anchor therefore determine which donor risks matter,
    retaining AVVP's multi-label/time structure instead of collapsing all
    class uncertainty to one sample scalar.
    """
    if audio.ndim != 3 or frame_logits.ndim != 5 or frame_logits.shape[2] != 2:
        raise ValueError(
            "class-aware U-HNM expects audio [B,T,D] and frame logits "
            f"[B,T,2,C,2], got {tuple(audio.shape)} and {tuple(frame_logits.shape)}"
        )
    expected = frame_logits.shape[:2] + (frame_logits.shape[3],)
    if audio_pseudo_labels.shape != expected or visual_pseudo_labels.shape != expected:
        raise ValueError(
            "pseudo-label shapes must match [B,T,C], got "
            f"{tuple(audio_pseudo_labels.shape)} and {tuple(visual_pseudo_labels.shape)}"
        )
    batch_size = audio.shape[0]
    if batch_size < 2:
        index = torch.arange(batch_size, device=audio.device)
        return index, {"self_pair_rate": 1.0 if batch_size else 0.0}

    audio_u = compute_uncertainty(
        frame_logits[:, :, 0, :, :].detach(), evidence_type, prior_strength=1.0
    )
    # Rank each time/class coordinate across the current batch.  This is the
    # exact ordinal information assumed by the scheduler and is invariant to
    # the absolute EDL scale.
    class_rank = rank_normalize(audio_u, dim=0)
    anchor_presence = (
        (audio_pseudo_labels.detach() > 0.5)
        | (visual_pseudo_labels.detach() > 0.5)
    ).to(class_rank.dtype)
    counts = anchor_presence.sum(dim=(1, 2))
    # [B_anchor, B_donor] after matching time/class coordinates.
    pair_risk = torch.einsum("itc,jtc->ij", anchor_presence, class_rank)
    # Fallback remains donor-specific when an anchor has no active event.
    fallback = class_rank.mean(dim=(1, 2))
    pair_risk = pair_risk / counts[:, None].clamp_min(1.0)
    pair_risk = torch.where(counts[:, None] > 0, pair_risk, fallback[None, :])

    pooled = torch.nn.functional.normalize(audio.detach().mean(dim=1), p=2, dim=-1)
    similarity = pooled @ pooled.transpose(0, 1)
    score = similarity * pair_risk.clamp_min(0.01)
    score.fill_diagonal_(-torch.inf)
    donor_idx = score.argmax(dim=1)
    anchors = torch.arange(batch_size, device=audio.device)
    diagnostics = {
        "selected_similarity": float(similarity[anchors, donor_idx].mean().item()),
        "selected_conditioned_risk": float(pair_risk[anchors, donor_idx].mean().item()),
        "batch_conditioned_risk": float(pair_risk.mean().item()),
        "anchor_active_fraction": float(anchor_presence.mean().item()),
        "self_pair_rate": float((anchors == donor_idx).float().mean().item()),
        "unique_donor_fraction": float(
            donor_idx.unique().numel() / max(batch_size, 1)
        ),
    }
    return donor_idx, diagnostics


def uncertain_learnable_visual_consistency(
    original_frame_logits: torch.Tensor,
    mixed_frame_logits: torch.Tensor,
    criterion_edl: EvidenceLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply retained-visual JS consistency only at mid-risk positions.

    The original prediction is a stop-gradient teacher. Within each sample,
    evidential uncertainty is rank-normalized and transformed by
    ``4 r (1-r)``. The gate is zero at the easiest and riskiest extremes and
    peaks at intermediate risk, targeting positions that are perturbed enough
    to be informative without self-confirming the least reliable predictions.
    """
    expected = original_frame_logits.shape
    if original_frame_logits.ndim != 5 or expected[2] != 2 or expected[-1] != 2:
        raise ValueError(
            "Expected original_frame_logits [B,T,2,C,2], "
            f"got {tuple(expected)}"
        )
    if mixed_frame_logits.shape != expected:
        raise ValueError(
            "Mixed frame logits must match the original shape: "
            f"{tuple(mixed_frame_logits.shape)} != {tuple(expected)}"
        )

    original_visual = original_frame_logits[:, :, 1, :, :]
    mixed_visual = mixed_frame_logits[:, :, 1, :, :]
    original_probability, original_uncertainty = criterion_edl.get_predictions(
        original_visual
    )
    mixed_probability, _ = criterion_edl.get_predictions(mixed_visual)

    # Legacy probabilities do not sum to one because their numerator omits the
    # Dirichlet prior. Renormalize only for categorical JS; uncertainty retains
    # the exact paper-checkpoint convention.
    eps = torch.finfo(original_probability.dtype).eps
    teacher = original_probability.detach().clamp_min(eps)
    student = mixed_probability.clamp_min(eps)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (teacher + student)
    js = 0.5 * (
        (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
        + (student * (student.log() - midpoint.log())).sum(dim=-1)
    )

    risk_rank = _within_sample_risk_rank(original_uncertainty.squeeze(-1))
    learnability_gate = 4.0 * risk_rank * (1.0 - risk_rank)
    weighted_loss = (learnability_gate * js).sum() / learnability_gate.sum().clamp_min(1e-8)

    diagnostics = {
        "consistency_loss": float(weighted_loss.detach().item()),
        "raw_visual_sensitivity": float(js.detach().mean().item()),
        "learnability_gate_mean": float(learnability_gate.mean().item()),
        "learnability_gate_gt_075": float(
            (learnability_gate > 0.75).float().mean().item()
        ),
        "risk_rank_mean": float(risk_rank.mean().item()),
    }
    return weighted_loss, diagnostics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return one mean per batch item, falling back to the unmasked mean."""
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).to(values.dtype)
    counts = flat_mask.sum(dim=1)
    selected = (flat_values * flat_mask).sum(dim=1) / counts.clamp_min(1.0)
    fallback = flat_values.mean(dim=1)
    return torch.where(counts > 0, selected, fallback)


def select_risk_anchor_pairs(
    frame_logits: torch.Tensor,
    audio_pseudo_labels: torch.Tensor,
    visual_pseudo_labels: torch.Tensor,
    *,
    evidence_type: str = "exp",
    mix_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build a full-size risk-anchor exchange batch.

    High-risk anchors are identified from frame-level evidence uncertainty on
    currently active pseudo-label positions.  Donors are sampled from the
    lower-risk half of the batch subject to non-self and class-disjointness
    constraints.  Constraints are relaxed only when a small batch has no valid
    candidate, and the relaxation rate is reported.
    """
    if frame_logits.ndim != 5 or frame_logits.shape[2] != 2 or frame_logits.shape[-1] != 2:
        raise ValueError(f"Expected frame_logits [B,T,2,C,2], got {tuple(frame_logits.shape)}")
    if not 0.0 < mix_fraction <= 1.0:
        raise ValueError(f"mix_fraction must be in (0, 1], got {mix_fraction}")

    batch_size = frame_logits.shape[0]
    device = frame_logits.device
    if batch_size == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, {
            "anchor_risk": math.nan,
            "donor_risk": math.nan,
            "class_overlap_rate": math.nan,
            "self_pair_rate": math.nan,
            "constraint_relaxation_rate": math.nan,
            "unique_anchor_fraction": math.nan,
        }

    # Match the historical U-HNM uncertainty convention exactly: legacy runs
    # use prior_strength=1.0 in main_ipr.py.
    audio_u = compute_uncertainty(
        frame_logits[:, :, 0, :, :].detach(), evidence_type, prior_strength=1.0
    )
    visual_u = compute_uncertainty(
        frame_logits[:, :, 1, :, :].detach(), evidence_type, prior_strength=1.0
    )
    active_audio = audio_pseudo_labels > 0.5
    active_visual = visual_pseudo_labels > 0.5
    active_union = active_audio | active_visual

    anchor_risk = 0.5 * (
        _masked_mean(audio_u, active_union) + _masked_mean(visual_u, active_union)
    )
    donor_risk = _masked_mean(audio_u, active_audio)

    pool_size = max(1, int(math.ceil(batch_size * mix_fraction)))
    anchor_pool = torch.topk(anchor_risk, k=pool_size, largest=True).indices
    # Keep the auxiliary pass at B items, as in random/similarity/U-HNM.
    anchor_idx = anchor_pool[torch.randint(pool_size, (batch_size,), device=device)]

    audio_presence = active_audio.any(dim=1)
    visual_presence = active_visual.any(dim=1)
    low_risk = donor_risk <= donor_risk.median()
    all_indices = torch.arange(batch_size, device=device)
    donors: list[torch.Tensor] = []
    relaxed = 0

    for anchor in anchor_idx:
        non_self = all_indices != anchor
        overlap = (audio_presence & visual_presence[anchor].unsqueeze(0)).any(dim=1)
        candidates = all_indices[non_self & ~overlap & low_risk]
        if candidates.numel() == 0:
            relaxed += 1
            candidates = all_indices[non_self & low_risk]
        if candidates.numel() == 0:
            candidates = all_indices[non_self]
        if candidates.numel() == 0:  # singleton-batch safety
            candidates = all_indices
        donors.append(candidates[torch.randint(candidates.numel(), (1,), device=device)])

    donor_idx = torch.cat(donors)
    overlap_after = (
        audio_presence[donor_idx] & visual_presence[anchor_idx]
    ).any(dim=1)
    diagnostics = {
        "anchor_risk": float(anchor_risk[anchor_idx].mean().item()),
        "donor_risk": float(donor_risk[donor_idx].mean().item()),
        "batch_anchor_risk": float(anchor_risk.mean().item()),
        "batch_donor_risk": float(donor_risk.mean().item()),
        "class_overlap_rate": float(overlap_after.float().mean().item()),
        "self_pair_rate": float((anchor_idx == donor_idx).float().mean().item()),
        "constraint_relaxation_rate": float(relaxed / batch_size),
        "unique_anchor_fraction": float(anchor_idx.unique().numel() / batch_size),
    }
    return anchor_idx, donor_idx, diagnostics


def select_structured_uhnm_pairs(
    audio: torch.Tensor,
    frame_logits: torch.Tensor,
    audio_pseudo_labels: torch.Tensor,
    visual_pseudo_labels: torch.Tensor,
    *,
    evidence_type: str = "exp",
    mix_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Select high-risk anchors and reliable, similar donors.

    This is an isolated scheduler replacement for historical U-HNM.  It keeps
    the exchange budget and the cross-modal similarity constraint, changing the
    risk definition to standard Beta/Dirichlet total variance.  Risk is used
    asymmetrically: the high-risk half of the batch supplies anchors (the
    positions that need intervention), while the lower-risk half supplies
    donors.  This avoids propagating uncertain pseudo-labels through the
    exchange operation.  The donor index is detached from the optimization
    graph.
    """
    if (
        audio.ndim != 3
        or frame_logits.ndim != 5
        or frame_logits.shape[2] != 2
        or frame_logits.shape[-1] != 2
    ):
        raise ValueError(
            "structured U-HNM expects audio [B,T,D] and frame logits "
            "[B,T,2,C,2], "
            f"got {tuple(audio.shape)} and {tuple(frame_logits.shape)}"
        )
    expected_pseudo_shape = frame_logits.shape[:2] + (frame_logits.shape[3],)
    if (
        audio.shape[0] != frame_logits.shape[0]
        or audio_pseudo_labels.shape != expected_pseudo_shape
        or visual_pseudo_labels.shape != expected_pseudo_shape
    ):
        raise ValueError(
            "audio, logits, and pseudo-label batch/time/class axes disagree: "
            f"audio={tuple(audio.shape)}, logits={tuple(frame_logits.shape)}, "
            f"audio_pseudo={tuple(audio_pseudo_labels.shape)}, "
            f"visual_pseudo={tuple(visual_pseudo_labels.shape)}"
        )
    batch_size = audio.shape[0]
    if batch_size == 0:
        empty = torch.empty(0, dtype=torch.long, device=audio.device)
        return empty, empty, {"selected": 0.0}

    audio_risk = beta_risk_from_logits(
        frame_logits[:, :, 0, :, :].detach(), evidence_type, prior_strength=1.0
    )["total_variance"]
    visual_risk = beta_risk_from_logits(
        frame_logits[:, :, 1, :, :].detach(), evidence_type, prior_strength=1.0
    )["total_variance"]
    flat_audio_risk = audio_risk.reshape(batch_size, -1)
    flat_visual_risk = visual_risk.reshape(batch_size, -1)
    flat_audio_presence = audio_pseudo_labels.detach().reshape(batch_size, -1) > 0.5
    flat_visual_presence = visual_pseudo_labels.detach().reshape(batch_size, -1) > 0.5
    active_union = flat_audio_presence | flat_visual_presence
    anchor_risk = 0.5 * (
        _masked_mean(flat_audio_risk, active_union)
        + _masked_mean(flat_visual_risk, active_union)
    )
    donor_risk = aggregate_class_risk(
        flat_audio_risk,
        mode="donor_positive",
        presence=flat_audio_presence,
    )
    pooled = torch.nn.functional.normalize(audio.detach().mean(dim=1), p=2, dim=-1)
    similarity = pooled @ pooled.transpose(0, 1)
    nonself_matrix = ~torch.eye(batch_size, dtype=torch.bool, device=audio.device)
    pool_size = max(1, int(math.ceil(batch_size * mix_fraction)))
    anchor_pool = torch.topk(anchor_risk, k=pool_size, largest=True).indices
    # Keep the auxiliary pass at B items, as in the historical exchange path.
    anchor_idx = anchor_pool[
        torch.randint(pool_size, (batch_size,), device=audio.device)
    ]
    low_risk = donor_risk <= donor_risk.median()
    all_indices = torch.arange(batch_size, device=audio.device)
    # Class-level presence is used for the disjointness constraint; a donor
    # sharing an active event class with the anchor is not a valid negative.
    audio_presence = audio_pseudo_labels.detach().bool().any(dim=1)
    visual_presence = visual_pseudo_labels.detach().bool().any(dim=1)
    donors: list[torch.Tensor] = []
    relaxed = 0
    overlap_values: list[torch.Tensor] = []
    for anchor in anchor_idx:
        non_self = all_indices != anchor
        overlap = (
            audio_presence & visual_presence[anchor].unsqueeze(0)
        ).any(dim=1)
        candidates = all_indices[non_self & ~overlap & low_risk]
        if candidates.numel() == 0:
            relaxed += 1
            candidates = all_indices[non_self & low_risk]
        if candidates.numel() == 0:
            candidates = all_indices[non_self]
        if candidates.numel() == 0:  # singleton-batch safety
            candidates = all_indices
        donor = candidates[similarity[anchor, candidates].argmax()]
        donors.append(donor.reshape(1))
        overlap_values.append(overlap[donor].reshape(1))
    donor_idx = torch.cat(donors)
    overlap_after = torch.cat(overlap_values)
    diagnostics = {
        "selected": float(batch_size),
        "anchor_risk": float(anchor_risk[anchor_idx].mean().item()),
        "donor_risk": float(donor_risk[donor_idx].mean().item()),
        "donor_rank": float(empirical_cdf(donor_risk)[donor_idx].mean().item()),
        "similarity": float(similarity[anchor_idx, donor_idx].mean().item()),
        "self_pair_rate": float((anchor_idx == donor_idx).float().mean().item()),
        "unique_donor_fraction": float(donor_idx.unique().numel() / batch_size),
        "unique_anchor_fraction": float(anchor_idx.unique().numel() / batch_size),
        "class_overlap_rate": float(overlap_after.float().mean().item()),
        "constraint_relaxation_rate": float(relaxed / batch_size),
        "negative_similarity_fraction": float(
            (similarity[nonself_matrix] < 0).float().mean().item()
        ),
    }
    return anchor_idx, donor_idx, diagnostics


def _append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _average_diagnostics(records: Iterable[dict[str, float]]) -> dict[str, float]:
    records = list(records)
    if not records:
        return {}
    keys = records[0].keys()
    return {
        key: sum(float(row[key]) for row in records) / len(records)
        for key in keys
        if isinstance(records[0][key], (int, float))
    }


def _proxy_sample_risk(logits: torch.Tensor, args) -> torch.Tensor:
    """Return one high-value-is-risk score per batch item."""
    raw = legacy.risk_proxy_from_logits(
        logits.detach(),
        _RISK_PROXY,
        evidence_type=args.evidence,
        prior_strength=1.0,
    )
    risk = raw.reshape(raw.shape[0], -1).mean(dim=1)
    return risk if _MIX_RISK_SIDE == "high" else 1.0 - risk


def train_validation_control(args, models, train_loader, optimizers, criterion, epoch, logger):
    """Legacy training step with only the scheduler intervention changed."""
    del criterion, logger  # retained in the signature for main_ipr compatibility
    legacy.exp_logger(f"begin train_with_pseudo_labels ({_SCHEDULER}).")
    for model in models:
        model.train()

    criterion2 = LabelSmoothingNCELoss(classes=10, smoothing=args.nce_smooth)
    criterion3 = MutualLearningLoss()
    criterion_edl = EvidenceLoss(
        num_classes=2,
        evidence=args.evidence,
        loss_type=args.loss_type,
        uncertain=False,
        annealing_step=args.epochs,
        brier_weight=args.brier_weight,
        temperature=args.edl_temperature,
        legacy_normalization=True,
    )
    epoch_diagnostics: list[dict[str, float]] = []
    memory_bank = None
    if _SCHEDULER in {"memory_uhnm", "temporal_uhnm"}:
        memory_bank = _build_global_memory_bank(
            models[0 if epoch <= args.epochs else 1],
            train_loader,
            args,
        )

    for batch_idx, sample in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Legacy {_SCHEDULER} epoch {epoch}",
    ):
        audio = sample["audio"].to("cuda")
        video = sample["video_s"].to("cuda")
        video_st = sample["video_st"].to("cuda")
        target = sample["label"].float().to("cuda")
        Pa = sample["Pa"].float().to("cuda")
        Pv = sample["Pv"].float().to("cuda")
        audio_pseudo = sample["audio_pseudo_labels"].float().to("cuda")
        visual_pseudo = sample["visual_pseudo_labels"].float().to("cuda")

        for optimizer in optimizers:
            optimizer.zero_grad()

        model_index = 0 if epoch <= args.epochs else 1
        outputs = models[model_index](
            audio, video, video_st, with_ca=True, epoch_num=epoch - 1
        )
        (
            global_logits,
            a_logits,
            v_logits,
            frame_logits,
            temporal_logits,
            frame_att,
            temporal_att,
            sims_after,
            mask_after,
        ) = outputs

        a_frame_logits = frame_logits[:, :, 0, :, :]
        v_frame_logits = frame_logits[:, :, 1, :, :]
        fbcl_loss = args.lamda_1 * criterion2(sims_after, mask_after) + args.lamda_2 * (
            criterion3(temporal_att[..., 0], frame_att[..., 0].max(dim=-2)[0])
            + criterion3(temporal_att[..., 1], frame_att[..., 1].max(dim=-2)[0])
        )
        frame_target = (
            audio_pseudo
            * visual_pseudo
            * target.unsqueeze(1).expand(Pa.shape[0], 10, 25)
        )
        tel_loss = criterion_edl(global_logits, target, epoch_num=epoch) + criterion_edl(
            temporal_logits, frame_target, epoch_num=epoch
        )
        ps_loss = args.audio_factor * criterion_edl(
            a_frame_logits, audio_pseudo, fore_only=True, epoch_num=epoch
        ) + criterion_edl(
            v_frame_logits, visual_pseudo, fore_only=True, epoch_num=epoch
        )
        audio_sl_loss = criterion_edl(a_logits, Pa, epoch_num=epoch)
        video_sl_loss = criterion_edl(v_logits, Pv, epoch_num=epoch)
        if model_index == 0:
            mel_loss = args.audio_factor * audio_sl_loss + video_sl_loss
        else:
            mel_loss = audio_sl_loss + args.video_factor * video_sl_loss
        loss_original = (
            ps_loss + args.lamda_3 * mel_loss + args.lamda_4 * tel_loss + fbcl_loss
        )

        if _SCHEDULER == "no_mix":
            loss = loss_original
        elif _SCHEDULER in {
            "risk_anchor",
            "structured_uhnm",
            "rank_uhnm",
            "class_aware_uhnm",
            "bandpass_uhnm",
            "memory_uhnm",
            "temporal_uhnm",
            "random_consistency",
            "uncertainty_consistency",
            "pa_uhnm",
            "pa_random",
            "xor_uhnm",
            "xor_random",
            "balanced_uhnm",
            "proxy_uhnm",
        }:
            if _SCHEDULER == "risk_anchor":
                anchor_idx, donor_idx, diagnostics = select_risk_anchor_pairs(
                    frame_logits,
                    audio_pseudo,
                    visual_pseudo,
                    evidence_type=args.evidence,
                    mix_fraction=_MIX_FRACTION,
                )
            elif _SCHEDULER == "structured_uhnm":
                anchor_idx, donor_idx, diagnostics = select_structured_uhnm_pairs(
                    audio,
                    frame_logits,
                    audio_pseudo,
                    visual_pseudo,
                    evidence_type=args.evidence,
                    mix_fraction=_MIX_FRACTION,
                )
            elif _SCHEDULER == "class_aware_uhnm":
                batch_size = audio.shape[0]
                anchor_idx = torch.arange(batch_size, device=audio.device)
                donor_idx, diagnostics = _class_aware_uhnm_pairs(
                    audio,
                    frame_logits,
                    audio_pseudo,
                    visual_pseudo,
                    evidence_type=args.evidence,
                )
            elif _SCHEDULER in {"memory_uhnm", "temporal_uhnm"}:
                if memory_bank is None:
                    raise RuntimeError("memory scheduler requires a built memory bank")
                batch_size = audio.shape[0]
                anchor_idx = torch.arange(batch_size, device=audio.device)
                anchor_names = sample.get("segment_name", sample.get("video_name"))
                if isinstance(anchor_names, str):
                    anchor_names = [anchor_names]
                if _SCHEDULER == "temporal_uhnm":
                    mix_audio, donor_pa, donor_idx, diagnostics = _temporal_neighbor_pairs(
                        audio,
                        memory_bank,
                        anchor_names,
                        top_k=_TEMPORAL_TOP_K,
                    )
                else:
                    mix_audio, donor_pa, donor_idx, diagnostics = _global_memory_bank_pairs(
                        audio,
                        memory_bank,
                        anchor_names,
                    )
            elif _SCHEDULER in {"random_consistency", "uncertainty_consistency"}:
                batch_size = audio.shape[0]
                anchor_idx = torch.arange(batch_size, device=audio.device)
                donor_idx = torch.randperm(batch_size, device=audio.device)
                diagnostics = {}
            else:
                batch_size = audio.shape[0]
                anchor_idx = torch.arange(batch_size, device=audio.device)
                diagnostics = {}
                if _SCHEDULER == "balanced_uhnm":
                    audio_uncertainty = compute_uncertainty(
                        a_logits.detach(), args.evidence, prior_strength=1.0
                    )
                    donor_idx, diagnostics = global_uhnm_assignment(
                        audio,
                        audio_uncertainty.mean(dim=-1),
                        donor_capacity=_DONOR_CAPACITY,
                    )
                elif _SCHEDULER == "proxy_uhnm":
                    proxy_risk = _proxy_sample_risk(a_logits, args)
                    donor_idx = hard_negative_mining(audio, proxy_risk)
                    diagnostics.update({
                        "selected_proxy_risk": float(proxy_risk[donor_idx].mean().item()),
                        "batch_proxy_risk": float(proxy_risk.mean().item()),
                    })
                elif _SCHEDULER in {"rank_uhnm", "bandpass_uhnm"}:
                    audio_uncertainty = compute_uncertainty(
                        a_logits.detach(), args.evidence, prior_strength=1.0
                    ).reshape(batch_size, -1).mean(dim=1)
                    donor_idx, diagnostics = _rank_uhnm_pairs(
                        audio,
                        audio_uncertainty,
                        band_mu=_BAND_MU if _SCHEDULER == "bandpass_uhnm" else None,
                        band_sigma=_BAND_SIGMA,
                    )
                elif _SCHEDULER in {"pa_uhnm", "xor_uhnm"}:
                    audio_uncertainty = compute_uncertainty(
                        a_logits.detach(), args.evidence, prior_strength=1.0
                    )
                    donor_idx = hard_negative_mining(
                        audio, audio_uncertainty.mean(dim=-1)
                    )
                else:
                    donor_idx = torch.randperm(batch_size, device=audio.device)
                diagnostics.update({
                    "self_pair_rate": float(
                        (anchor_idx == donor_idx).float().mean().item()
                    ),
                    "unique_donor_fraction": float(
                        donor_idx.unique().numel() / max(batch_size, 1)
                    ),
                })
            if _SCHEDULER not in {"memory_uhnm", "temporal_uhnm"}:
                mix_audio = audio[donor_idx]
                donor_pa = Pa[donor_idx]
            mixed_outputs = models[model_index](
                mix_audio,
                video[anchor_idx],
                video_st[anchor_idx],
                with_ca=True,
                epoch_num=epoch - 1,
            )
            _, a_mix, v_mix, mixed_frame_logits, _, _, _, _, _ = mixed_outputs
            if _SCHEDULER in {
                "pa_uhnm",
                "pa_random",
                "xor_uhnm",
                "xor_random",
            }:
                if _SCHEDULER in {"xor_uhnm", "xor_random"}:
                    xor_mask = presence_absence_xor_mask(
                        Pa[donor_idx], Pv[anchor_idx]
                    )
                    audio_pa_mask = xor_mask
                    visual_pa_mask = xor_mask
                else:
                    audio_pa_mask, visual_pa_mask = presence_absence_masks(
                        Pa[donor_idx], Pv[anchor_idx]
                    )
                loss_audio_mix = masked_evidence_bce(
                    criterion_edl, a_mix, Pa[donor_idx], audio_pa_mask
                )
                loss_video_mix = masked_evidence_bce(
                    criterion_edl, v_mix, Pv[anchor_idx], visual_pa_mask
                )
                diagnostics.update(
                    {
                        "audio_mask_count": float(audio_pa_mask.sum().item()),
                        "visual_mask_count": float(visual_pa_mask.sum().item()),
                        "audio_active_samples": float(
                            (audio_pa_mask.sum(dim=1) > 0).sum().item()
                        ),
                        "visual_active_samples": float(
                            (visual_pa_mask.sum(dim=1) > 0).sum().item()
                        ),
                        "audio_mix_loss": float(loss_audio_mix.detach().item()),
                        "visual_mix_loss": float(loss_video_mix.detach().item()),
                        "audio_mask_positive_fraction": float(
                            (Pa[donor_idx] * audio_pa_mask).sum().item()
                            / max(audio_pa_mask.sum().item(), 1.0)
                        ),
                        "visual_mask_positive_fraction": float(
                            (Pv[anchor_idx] * visual_pa_mask).sum().item()
                            / max(visual_pa_mask.sum().item(), 1.0)
                        ),
                    }
                )
            else:
                loss_audio_mix = criterion_edl(a_mix, donor_pa, epoch_num=epoch)
                loss_video_mix = criterion_edl(v_mix, Pv[anchor_idx], epoch_num=epoch)
            if model_index == 0:
                loss = loss_original + 0.5 * loss_video_mix + 0.0 * loss_audio_mix
            else:
                loss = loss_original + 0.5 * (
                    loss_audio_mix + args.video_factor * loss_video_mix
                )
            if _SCHEDULER == "random_consistency":
                consistency_loss, consistency_diagnostics = (
                    evidence_gated_visual_consistency(
                        frame_logits,
                        mixed_frame_logits,
                        criterion_edl,
                    )
                )
                diagnostics.update(consistency_diagnostics)
                diagnostics["consistency_weight"] = _CONSISTENCY_WEIGHT
                loss = loss + _CONSISTENCY_WEIGHT * consistency_loss
            elif _SCHEDULER == "uncertainty_consistency":
                consistency_loss, consistency_diagnostics = (
                    uncertain_learnable_visual_consistency(
                        frame_logits,
                        mixed_frame_logits,
                        criterion_edl,
                    )
                )
                diagnostics.update(consistency_diagnostics)
                diagnostics["consistency_weight"] = _CONSISTENCY_WEIGHT
                loss = loss + _CONSISTENCY_WEIGHT * consistency_loss
            epoch_diagnostics.append(diagnostics)
        else:
            raise ValueError(f"Unsupported patched scheduler: {_SCHEDULER!r}")

        loss.backward()
        optimizers[model_index].step()
        ema = args.ema_audio if model_index == 0 else args.ema_video
        if ema is not None:
            ema.update()

        if batch_idx % args.log_interval == 0:
            legacy.exp_logger(
                f"Train {'Audio' if model_index == 0 else 'Video'} Model "
                f"Epoch: {epoch} \t Loss: {loss.item():.3f} ({_SCHEDULER})"
            )

    if epoch_diagnostics:
        record = {
            "epoch": epoch,
            "phase": "audio" if epoch <= args.epochs else "video",
            "scheduler": _SCHEDULER,
            **_average_diagnostics(epoch_diagnostics),
        }
        _append_jsonl(_DIAGNOSTICS_PATH, record)
        legacy.exp_logger("LEGACY_SCHEDULER_DIAGNOSTIC " + json.dumps(record, sort_keys=True))


def validation_only_eval(args, model, loader, set_path, epoch=0, dual_eval=False):
    """Hard guard that refuses every evaluation request for the test CSV."""
    global _TEST_ATTEMPTS
    requested = Path(set_path).expanduser().resolve()
    if _TEST_LABEL is not None and requested == _TEST_LABEL:
        _TEST_ATTEMPTS += 1
        legacy.exp_logger(
            "TEST EVALUATION BLOCKED: validation-only scheduler screening "
            f"({requested})"
        )
        return (float("nan"),) * 10
    return _ORIGINAL_EVAL(args, model, loader, set_path, epoch, dual_eval)


def _option_value(arguments: list[str], option: str, default: str) -> str:
    try:
        return arguments[arguments.index(option) + 1]
    except (ValueError, IndexError):
        return default


def _remove_option(arguments: list[str], option: str) -> list[str]:
    result: list[str] = []
    skip = False
    for item in arguments:
        if skip:
            skip = False
            continue
        if item == option:
            skip = True
            continue
        result.append(item)
    return result


def main() -> None:
    global _SCHEDULER, _MIX_FRACTION, _CONSISTENCY_WEIGHT, _DONOR_CAPACITY
    global _TEMPORAL_TOP_K
    global _BAND_MU, _BAND_SIGMA
    global _RISK_PROXY
    global _MIX_RISK_SIDE
    global _DIAGNOSTICS_PATH, _TEST_LABEL
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--scheduler",
        required=True,
        choices=[
            "no_mix",
            "random",
            "similarity",
            "uhnm",
            "risk_anchor",
            "structured_uhnm",
            "rank_uhnm",
            "class_aware_uhnm",
            "bandpass_uhnm",
            "memory_uhnm",
            "temporal_uhnm",
            "random_consistency",
            "uncertainty_consistency",
            "pa_uhnm",
            "pa_random",
            "xor_uhnm",
            "xor_random",
            "balanced_uhnm",
            "proxy_uhnm",
        ],
    )
    parser.add_argument("--mix_fraction", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--donor_capacity", type=int, default=4)
    parser.add_argument("--temporal_top_k", type=int, default=3)
    parser.add_argument(
        "--band_mu", type=float, default=0.78,
        help="Peak empirical risk percentile for bandpass U-HNM.",
    )
    parser.add_argument(
        "--band_sigma", type=float, default=0.18,
        help="Width of the empirical risk band for bandpass U-HNM.",
    )
    parser.add_argument(
        "--risk_proxy", choices=["edl", "entropy", "margin", "random"],
        default="edl",
        help="Risk proxy used by proxy_uhnm and --ipr_risk proxy.",
    )
    parser.add_argument(
        "--mix_risk_side", choices=["high", "low"], default="high",
        help="Whether proxy_uhnm selects high- or low-risk donors.",
    )
    parser.add_argument("--diagnostics_path", type=Path)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument(
        "--allow_ipr",
        action="store_true",
        help="Permit a validation-only Round-2 IPR run for a delegated scheduler.",
    )
    custom, remaining = parser.parse_known_args()

    _SCHEDULER = custom.scheduler
    _MIX_FRACTION = custom.mix_fraction
    _CONSISTENCY_WEIGHT = custom.consistency_weight
    _DONOR_CAPACITY = custom.donor_capacity
    _TEMPORAL_TOP_K = custom.temporal_top_k
    _BAND_MU = custom.band_mu
    _BAND_SIGMA = custom.band_sigma
    _RISK_PROXY = custom.risk_proxy
    _MIX_RISK_SIDE = custom.mix_risk_side
    if _DONOR_CAPACITY < 1:
        raise ValueError("--donor_capacity must be >= 1")
    if _TEMPORAL_TOP_K < 1:
        raise ValueError("--temporal_top_k must be >= 1")
    if (
        _SCHEDULER in {"random_consistency", "uncertainty_consistency"}
        and _CONSISTENCY_WEIGHT <= 0.0
    ):
        raise ValueError(f"{_SCHEDULER} requires --consistency_weight > 0")
    _DIAGNOSTICS_PATH = custom.diagnostics_path
    label_test = _option_value(remaining, "--label_test", "data/AVVP_test_pd.csv")
    _TEST_LABEL = Path(label_test).expanduser().resolve()

    requested_semantics = _option_value(remaining, "--loss_semantics", "legacy")
    requested_rounds = int(_option_value(remaining, "--ipr_rounds", "1"))
    ipr_gate = _option_value(remaining, "--ipr_gate", "low")
    if requested_semantics != "legacy":
        raise ValueError("This runner only permits --loss_semantics legacy")
    if requested_rounds != 1:
        if not (
            custom.allow_ipr
            and requested_rounds == 2
            and custom.scheduler in {
                "no_mix",
                "random",
                "similarity",
                "uhnm",
                "structured_uhnm",
                "rank_uhnm",
                "class_aware_uhnm",
                "bandpass_uhnm",
                "memory_uhnm",
                "temporal_uhnm",
                "pa_uhnm",
                "pa_random",
                "xor_uhnm",
                "xor_random",
                "balanced_uhnm",
                "proxy_uhnm",
            }
        ):
            raise ValueError(
                "Round-2 IPR requires --allow_ipr, --ipr_rounds 2, and a "
                "delegated no-mix/random/similarity/uhnm scheduler"
            )

    remaining = _remove_option(remaining, "--loss_semantics")
    remaining += ["--loss_semantics", "legacy"]
    remaining = _remove_option(remaining, "--ipr_rounds")
    remaining += ["--ipr_rounds", str(requested_rounds)]
    remaining = _remove_option(remaining, "--mixing")
    remaining = _remove_option(remaining, "--temporal_top_k")
    remaining = _remove_option(remaining, "--risk_proxy")
    remaining += ["--risk_proxy", _RISK_PROXY]
    remaining = _remove_option(remaining, "--mix_risk_side")
    # ``main_ipr.py`` validates ``--mixing`` against its historical enum.  The
    # structured scheduler is implemented by this wrapper, so it uses the
    # random value only as a parser-compatible placeholder and never delegates
    # the actual Round-1 exchange to the legacy branch.
    delegated_mixing = (
        custom.scheduler
        if custom.scheduler in {"random", "similarity", "uhnm"}
        else "random"
    )
    remaining += ["--mixing", delegated_mixing]

    if custom.scheduler in {
        "no_mix",
        "risk_anchor",
        "structured_uhnm",
        "random_consistency",
        "uncertainty_consistency",
        "pa_uhnm",
        "pa_random",
        "xor_uhnm",
        "xor_random",
        "balanced_uhnm",
        "proxy_uhnm",
        "rank_uhnm",
        "class_aware_uhnm",
        "bandpass_uhnm",
        "memory_uhnm",
        "temporal_uhnm",
    }:
        legacy.train_with_pseudo_labels = train_validation_control
    else:
        legacy.train_with_pseudo_labels = _ORIGINAL_TRAIN
    legacy.eval = validation_only_eval

    started = time.time()
    status = "failed"
    error: str | None = None
    try:
        sys.argv = [str(Path(legacy.__file__).resolve()), *remaining]
        legacy.main()
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata = {
            "status": status,
            "error": error,
            "scheduler": custom.scheduler,
            "delegated_mixing": delegated_mixing,
            "mix_fraction": custom.mix_fraction,
            "consistency_weight": custom.consistency_weight,
            "donor_capacity": custom.donor_capacity,
            "temporal_top_k": custom.temporal_top_k,
            "band_mu": custom.band_mu,
            "band_sigma": custom.band_sigma,
            "risk_proxy": custom.risk_proxy,
            "mix_risk_side": custom.mix_risk_side,
            "ipr_gate": ipr_gate,
            "loss_semantics": "legacy",
            "ipr_rounds": requested_rounds,
            "selection_split": "validation",
            "test_evaluated": False,
            "blocked_test_attempts": _TEST_ATTEMPTS,
            "test_label": str(_TEST_LABEL),
            "elapsed_seconds": time.time() - started,
            "argv": remaining,
        }
        custom.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        custom.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
