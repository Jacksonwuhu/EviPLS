import torch
import unittest

from nets.criterion import EvidenceLoss


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


def _criterion(**kwargs):
    return EvidenceLoss(
        num_classes=2,
        evidence='exp',
        loss_type='bce',
        uncertain=False,
        legacy_normalization=True,
        **kwargs,
    )


def test_disabled_mode_is_exact_legacy_bce():
    torch.manual_seed(7)
    logits = torch.randn(3, 4, 5, 2)
    target = torch.randint(0, 2, (3, 4, 5)).float()
    loss = _criterion(risk_weighting='none')(logits, target, epoch_num=3)
    probability = logits.softmax(dim=-1)
    expected = torch.nn.functional.binary_cross_entropy(
        probability, torch.stack((target, 1 - target), dim=-1)
    )
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-7)


def test_empirical_percentile_is_ordinal_and_bounded():
    values = torch.tensor([4.0, -2.0, 1.0, 9.0])
    rank = EvidenceLoss._empirical_percentile(values)
    torch.testing.assert_close(rank, torch.tensor([2 / 3, 0.0, 1 / 3, 1.0]))


def test_global_weights_preserve_loss_scale_and_stop_gradient():
    torch.manual_seed(11)
    logits = torch.randn(4, 3, 2, requires_grad=True)
    target = torch.randint(0, 2, (4, 3)).float()
    criterion = _criterion(
        risk_weighting='global', risk_weight_floor=0.5,
        risk_weight_gamma=1.0, risk_weight_warmup=0,
    )
    loss = criterion(logits, target, epoch_num=5)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    stats = criterion.last_risk_weight_stats
    assert abs(stats['weight_mean'] - 1.0) < 1e-6
    assert stats['weight_min'] > 0.0


def test_class_mass_mode_preserves_each_target_stratum():
    logits = torch.tensor(
        [[2.0, -1.0], [0.1, 0.0], [-2.0, 1.0], [1.0, -0.5]],
        dtype=torch.float32,
    )
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    criterion = _criterion(
        risk_weighting='class_mass', risk_weight_floor=0.5,
        risk_weight_gamma=1.0, risk_weight_warmup=0,
    )
    weights = criterion._ordinal_reliability_weights(logits, target, epoch_num=5)
    assert abs(float(weights[target == 1].mean()) - 1.0) < 1e-6
    assert abs(float(weights[target == 0].mean()) - 1.0) < 1e-6


def test_warmup_zero_effect_at_epoch_zero():
    logits = torch.randn(8, 2)
    target = torch.randint(0, 2, (8,)).float()
    criterion = _criterion(
        risk_weighting='global', risk_weight_floor=0.5,
        risk_weight_gamma=2.0, risk_weight_warmup=3,
    )
    weights = criterion._ordinal_reliability_weights(logits, target, epoch_num=0)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_ordinal_smoothing_preserves_bounds_and_gradient_mass():
    logits = torch.tensor([[3.0, -1.0], [0.0, 0.0], [-2.0, 2.0]], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0])
    criterion = _criterion(
        risk_weighting='smooth', risk_weight_warmup=0, risk_smooth_max=0.2,
    )
    smoothed = criterion._ordinal_smoothed_target(logits, target, epoch_num=5)
    assert torch.all(smoothed >= 0.0)
    assert torch.all(smoothed <= 1.0)
    assert torch.any((smoothed > 0.0) & (smoothed < 1.0))
    loss = criterion(logits, target, epoch_num=5)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
