import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

# 尝试导入 BaseWeightedLoss
try:
    from .base import BaseWeightedLoss
except ImportError:
    class BaseWeightedLoss(nn.Module):
        def __init__(self): super().__init__()

# --- 辅助函数 ---
def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

# ==========================================
# 1. EvidenceLoss (终极修正版：强制展平)
# ==========================================
class EvidenceLoss(nn.Module):
    def __init__(self, num_classes=2, evidence='exp', loss_type='log', uncertain=False, annealing_step=10,
                 daedl_mode=False, u_reweight=False,
                 brier_weight=0.0, temperature=1.0,
                 legacy_normalization=True,
                 risk_weighting='none', risk_weight_floor=0.5,
                 risk_weight_gamma=1.0, risk_weight_warmup=3,
                 risk_smooth_max=0.2):
        """
        daedl_mode: remove +1 bias (S = evidence instead of S = evidence + 1) - EXPERIMENTAL
        u_reweight: uncertainty-aware loss reweighting - EXPERIMENTAL, may cause collapse
        brier_weight: Brier score regularization strength (0 = disabled). Prevents overconfidence.
        temperature: evidence logit temperature scaling (>1 = softer, <1 = sharper)
        """
        super(EvidenceLoss, self).__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.loss_type = loss_type
        self.uncertain = uncertain
        self.epoch_num = 0
        self.annealing_step = annealing_step
        self.daedl_mode = daedl_mode
        self.u_reweight = u_reweight
        self.brier_weight = brier_weight
        self.temperature = temperature
        self.legacy_normalization = legacy_normalization
        if risk_weighting not in {'none', 'global', 'class_mass', 'smooth'}:
            raise ValueError(f"Unsupported risk weighting: {risk_weighting!r}")
        if not 0.0 < risk_weight_floor <= 1.0:
            raise ValueError("risk_weight_floor must be in (0, 1]")
        if risk_weight_gamma <= 0.0:
            raise ValueError("risk_weight_gamma must be > 0")
        if risk_weight_warmup < 0:
            raise ValueError("risk_weight_warmup must be >= 0")
        if not 0.0 <= risk_smooth_max < 1.0:
            raise ValueError("risk_smooth_max must be in [0, 1)")
        self.risk_weighting = risk_weighting
        self.risk_weight_floor = risk_weight_floor
        self.risk_weight_gamma = risk_weight_gamma
        self.risk_weight_warmup = risk_weight_warmup
        self.risk_smooth_max = risk_smooth_max
        self.last_risk_weight_stats = None

    @staticmethod
    def _empirical_percentile(values):
        """Return deterministic empirical percentiles in [0, 1].

        The operation is deliberately detached: evidential risk controls the
        optimization schedule but is not itself optimized through the weight.
        """
        flat = values.detach().reshape(-1)
        if flat.numel() <= 1:
            return torch.zeros_like(flat)
        # The project is pinned to an older PyTorch release without the
        # ``stable=`` keyword.  Evidential scores are continuous in practice;
        # deterministic index order is sufficient for the vanishingly rare ties.
        order = torch.argsort(flat, dim=0)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        return rank.to(flat.dtype) / float(order.numel() - 1)

    def _ordinal_reliability_weights(self, output, target, epoch_num):
        logits = output.reshape(-1, self.num_classes)
        with torch.no_grad():
            evidence = self.get_evidence(logits)
            # The prior strength does not affect the ordering.  Using +1 here
            # preserves the risk interface of the locked paper pipeline.
            uncertainty = self.num_classes / (evidence.sum(dim=-1) + 1.0).clamp_min(1e-8)
            risk = self._empirical_percentile(uncertainty)
            base = self.risk_weight_floor + (1.0 - self.risk_weight_floor) * (
                1.0 - risk
            ).pow(self.risk_weight_gamma)
            if self.risk_weight_warmup > 0:
                progress = min(1.0, float(epoch_num) / float(self.risk_weight_warmup))
            else:
                progress = 1.0
            weight = 1.0 - progress * (1.0 - base)

            if self.risk_weighting == 'class_mass':
                # Preserve the original positive/negative loss mass.  Risk
                # changes which examples matter within each stratum, without
                # silently changing the task's implicit class prior.
                target_flat = target.detach().reshape(-1)
                foreground = target_flat >= 0.5
                for mask in (foreground, ~foreground):
                    if mask.any():
                        weight[mask] = weight[mask] / weight[mask].mean().clamp_min(1e-8)
            else:
                weight = weight / weight.mean().clamp_min(1e-8)

            self.last_risk_weight_stats = {
                'risk_mean': float(risk.mean()),
                'weight_mean': float(weight.mean()),
                'weight_min': float(weight.min()),
                'weight_max': float(weight.max()),
                'warmup_progress': progress,
            }
        return weight

    def _ordinal_smoothed_target(self, output, target, epoch_num):
        """Smooth uncertain binary targets without removing their gradient mass."""
        logits = output.reshape(-1, self.num_classes)
        with torch.no_grad():
            evidence = self.get_evidence(logits)
            uncertainty = self.num_classes / (evidence.sum(dim=-1) + 1.0).clamp_min(1e-8)
            risk = self._empirical_percentile(uncertainty)
            if self.risk_weight_warmup > 0:
                progress = min(1.0, float(epoch_num) / float(self.risk_weight_warmup))
            else:
                progress = 1.0
            smoothing = self.risk_smooth_max * progress * risk
            target_flat = target.detach().reshape(-1)
            smoothed = (1.0 - smoothing) * target_flat + 0.5 * smoothing
            self.last_risk_weight_stats = {
                'risk_mean': float(risk.mean()),
                'smoothing_mean': float(smoothing.mean()),
                'smoothing_max': float(smoothing.max()),
                'warmup_progress': progress,
            }
        return smoothed

    def get_evidence(self, y):
        # Temperature scaling: soften/sharpen logits before evidence computation
        y_scaled = y / self.temperature
        if self.evidence == 'relu':
            return F.relu(y_scaled)
        elif self.evidence == 'exp':
            return torch.exp(torch.clamp(y_scaled, -10, 10))
        elif self.evidence == 'softplus':
            return F.softplus(y_scaled)
        return torch.exp(torch.clamp(y_scaled, -10, 10))

    def get_predictions(self, x):
        evidence = self.get_evidence(x)
        if self.daedl_mode:
            alpha = evidence  # DAEDL: no +1 bias (EXPERIMENTAL)
        elif self.legacy_normalization:
            # Compatibility with the implementation that produced the paper's
            # reported checkpoints and fixed evaluation thresholds.
            denominator = torch.sum(evidence + 1.0, dim=-1, keepdim=True)
            p = evidence / denominator
            u = self.num_classes / denominator
            return p, u
        else:
            alpha = evidence + 1.0
        denominator = torch.sum(alpha, dim=-1, keepdim=True)
        p = alpha / denominator
        u = self.num_classes / denominator
        return p, u

    def kl_divergence(self, alpha, num_classes):
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        return first_term + second_term

    def loglikelihood_loss(self, y, alpha):
        y = y.to(alpha.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        return torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    def forward(self, output, target, fore_only=False, epoch_num=0, **kwargs):
        # 更新 epoch
        self.epoch_num = epoch_num
        
        # 1. 处理 Alpha (Evidence -> Alpha)
        if self.uncertain:
            evidence = self.get_evidence(output)
            alpha = evidence + 1
        else:
            alpha = output.softmax(dim=-1)

        # 2. 处理 Target (Indices -> OneHot/Binary)
        # target input shape: [Batch, Time, Class] or [Batch, Class]
        y = target
        n_y = 1 - target
        gt_y = torch.stack((y, n_y), dim=-1) # shape: [..., 2]
        
        # 3. ������ 强制展平 (Force Flatten)
        # 无论输入是 [B, T, C, 2] 还是 [B, C, 2]，统一拉直成 [N, 2]
        # 这样就能彻底解决 ValueError: Target size different to input size 的问题
        alpha_flat = alpha.reshape(-1, 2)
        gt_flat = gt_y.reshape(-1, 2)

        # 4. 计算 Loss
        if self.loss_type == 'log':
            loss_cls = torch.mean(self.loglikelihood_loss(gt_flat, alpha_flat))
            loss_kl = torch.mean(self.kl_divergence(alpha_flat, self.num_classes))
            
            # KL Annealing
            steps = self.annealing_step if self.annealing_step > 0 else 1
            annealing_coef = min(1.0, self.epoch_num / steps)
            
            return loss_cls + annealing_coef * 0.1 * loss_kl
            
        elif self.loss_type == 'bce':
            # p = alpha / S
            S = torch.sum(alpha_flat, dim=1, keepdim=True)
            p = alpha_flat / S

            if self.risk_weighting == 'none':
                loss_bce = F.binary_cross_entropy(p, gt_flat)
            elif self.risk_weighting == 'smooth':
                smoothed_target = self._ordinal_smoothed_target(output, target, epoch_num)
                loss_bce = F.binary_cross_entropy(p[:, 0], smoothed_target)
            else:
                # In the binary complementary formulation the two BCE terms
                # are identical, so the foreground component exactly recovers
                # the unweighted legacy loss while allowing one scalar weight
                # per time/class pseudo target.
                point_loss = F.binary_cross_entropy(
                    p[:, 0], gt_flat[:, 0], reduction='none'
                )
                weights = self._ordinal_reliability_weights(output, target, epoch_num)
                loss_bce = (weights * point_loss).mean()

            # Brier Score regularization: penalize overconfident wrong predictions
            # Brier = (p_fg - y)^2, robust against Dirichlet collapse
            if self.brier_weight > 0:
                brier = ((p[:, 0] - gt_flat[:, 0]) ** 2).mean()
                return loss_bce + self.brier_weight * brier
            else:
                return loss_bce
            
        else:
            raise NotImplementedError

# ==========================================
# 2. MutualLearningLoss
# ==========================================
class MutualLearningLoss(nn.Module):
    def __init__(self, weight=1.0, lamb=1.0):
        super(MutualLearningLoss, self).__init__()
        self.weight = weight
        self.lamb = lamb

    def get_positive_expectation(self, p_samples, measure='JSD', average=True):
        log_2 = math.log(2.)
        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'H2':
            Ep = torch.ones_like(p_samples) - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, measure='JSD', average=True):
        log_2 = math.log(2.)
        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Eq.mean()
        else:
            return Eq

    def forward(self, X, Y, measure='JSD'):
        E_pos = self.get_positive_expectation(X, measure, average=False)
        E_neg = self.get_negative_expectation(Y, measure, average=False)
        E = E_neg - E_pos
        return torch.mean(E)
