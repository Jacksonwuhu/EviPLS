"""Small deterministic checks for the legacy scheduler selector and guard."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.legacy_scheduler import runner
from nets.criterion import EvidenceLoss


def _legacy_edl() -> EvidenceLoss:
    return EvidenceLoss(
        num_classes=2,
        evidence="exp",
        loss_type="bce",
        legacy_normalization=True,
    )


class EvidenceGatedConsistencyTest(unittest.TestCase):
    def test_identical_logits_have_zero_loss(self):
        logits = torch.zeros(2, 3, 2, 4, 2)
        loss, diagnostics = runner.evidence_gated_visual_consistency(
            logits, logits.clone(), _legacy_edl()
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diagnostics["raw_visual_sensitivity"], 0.0)
        self.assertEqual(diagnostics["consistency_loss"], 0.0)

    def test_teacher_is_detached_and_gradient_reaches_mixed_visual_branch(self):
        original = torch.zeros(2, 3, 2, 4, 2, requires_grad=True)
        mixed = torch.zeros(2, 3, 2, 4, 2, requires_grad=True)
        with torch.no_grad():
            mixed[:, :, 1, :, 0] = 1.0
        loss, _ = runner.evidence_gated_visual_consistency(
            original, mixed, _legacy_edl()
        )
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNone(original.grad)
        self.assertIsNotNone(mixed.grad)
        self.assertGreater(mixed.grad[:, :, 1, :, :].abs().sum().item(), 0.0)
        self.assertEqual(mixed.grad[:, :, 0, :, :].abs().sum().item(), 0.0)

    def test_stronger_evidence_produces_higher_certainty_gate(self):
        mixed = torch.zeros(1, 2, 2, 3, 2)
        weak = torch.zeros_like(mixed)
        strong = torch.zeros_like(mixed)
        strong[:, :, 1, :, :] = 4.0
        _, weak_diagnostics = runner.evidence_gated_visual_consistency(
            weak, mixed, _legacy_edl()
        )
        _, strong_diagnostics = runner.evidence_gated_visual_consistency(
            strong, mixed, _legacy_edl()
        )
        self.assertGreater(
            strong_diagnostics["certainty_gate_mean"],
            weak_diagnostics["certainty_gate_mean"],
        )

    def test_shape_mismatch_fails(self):
        original = torch.zeros(2, 3, 2, 4, 2)
        mixed = torch.zeros(2, 3, 2, 5, 2)
        with self.assertRaises(ValueError):
            runner.evidence_gated_visual_consistency(
                original, mixed, _legacy_edl()
            )


class RiskProxyTest(unittest.TestCase):
    def test_proxy_shapes_and_bounds(self):
        logits = torch.tensor(
            [[[2.0, -2.0], [0.0, 0.0]], [[-1.0, 1.0], [0.5, 0.5]]]
        )
        for proxy in ("edl", "entropy", "margin", "random"):
            values = runner.legacy.risk_proxy_from_logits(logits, proxy)
            self.assertEqual(values.shape, logits.shape[:-1])
            self.assertTrue(torch.isfinite(values).all())
            self.assertGreaterEqual(values.min().item(), 0.0)
            self.assertLessEqual(values.max().item(), 1.0)

    def test_entropy_and_margin_peak_at_ambiguous_logits(self):
        logits = torch.tensor([[[0.0, 0.0], [4.0, -4.0]]])
        entropy = runner.legacy.risk_proxy_from_logits(logits, "entropy")
        margin = runner.legacy.risk_proxy_from_logits(logits, "margin")
        self.assertGreater(entropy[0, 0].item(), entropy[0, 1].item())
        self.assertGreater(margin[0, 0].item(), margin[0, 1].item())


class UncertainLearnableConsistencyTest(unittest.TestCase):
    def test_identical_logits_have_zero_loss(self):
        logits = torch.randn(2, 3, 2, 4, 2)
        loss, diagnostics = runner.uncertain_learnable_visual_consistency(
            logits, logits.clone(), _legacy_edl()
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertAlmostEqual(diagnostics["raw_visual_sensitivity"], 0.0, places=7)

    def test_teacher_is_detached_and_gradient_reaches_only_mixed_visual_branch(self):
        original = torch.randn(2, 3, 2, 4, 2, requires_grad=True)
        mixed = original.detach().clone().requires_grad_(True)
        with torch.no_grad():
            mixed[:, :, 1, :, 0] += 0.7
        loss, diagnostics = runner.uncertain_learnable_visual_consistency(
            original, mixed, _legacy_edl()
        )
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNone(original.grad)
        self.assertIsNotNone(mixed.grad)
        self.assertGreater(mixed.grad[:, :, 1, :, :].abs().sum().item(), 0.0)
        self.assertEqual(mixed.grad[:, :, 0, :, :].abs().sum().item(), 0.0)
        self.assertAlmostEqual(diagnostics["risk_rank_mean"], 0.5, places=6)

    def test_mid_risk_gate_suppresses_both_rank_extremes(self):
        uncertainty = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        rank = runner._within_sample_risk_rank(uncertainty)
        gate = 4.0 * rank * (1.0 - rank)
        self.assertEqual(gate[0, 0].item(), 0.0)
        self.assertEqual(gate[0, -1].item(), 0.0)
        self.assertEqual(gate[0, 2].item(), 1.0)

    def test_shape_mismatch_fails(self):
        original = torch.zeros(2, 3, 2, 4, 2)
        mixed = torch.zeros(2, 3, 2, 5, 2)
        with self.assertRaises(ValueError):
            runner.uncertain_learnable_visual_consistency(
                original, mixed, _legacy_edl()
            )


class OrdinalSchedulerTest(unittest.TestCase):
    def test_rank_uhnm_is_nonself_and_uses_batch_percentiles(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]]
        )
        uncertainty = torch.tensor([0.1, 0.4, 0.8, 0.9])
        donors, diagnostics = runner._rank_uhnm_pairs(features, uncertainty)
        self.assertFalse(torch.any(donors == torch.arange(4)))
        self.assertGreaterEqual(diagnostics["selected_risk_rank"], 0.0)
        self.assertLessEqual(diagnostics["selected_risk_rank"], 1.0)

    def test_bandpass_peaks_at_configured_risk(self):
        features = torch.eye(5)
        uncertainty = torch.linspace(0.1, 0.9, 5)
        donors, diagnostics = runner._rank_uhnm_pairs(
            features, uncertainty, band_mu=0.75, band_sigma=0.2
        )
        self.assertFalse(torch.any(donors == torch.arange(5)))
        expected_peak = torch.exp(torch.tensor(-0.5 * ((0.75 - 0.75) / 0.2) ** 2))
        self.assertLessEqual(diagnostics["selected_risk_gate"], expected_peak.item())
        self.assertGreater(diagnostics["selected_risk_gate"], 0.0)

    def test_class_aware_selector_respects_time_class_contract(self):
        torch.manual_seed(5)
        audio = torch.randn(4, 2, 3)
        logits = torch.zeros(4, 2, 2, 3, 2)
        logits[:, :, 0, :, 0] = torch.randn(4, 2, 3)
        audio_labels = torch.zeros(4, 2, 3)
        visual_labels = torch.zeros(4, 2, 3)
        audio_labels[:, :, 0] = 1.0
        visual_labels[0, :, 1] = 1.0
        donors, diagnostics = runner._class_aware_uhnm_pairs(
            audio, logits, audio_labels, visual_labels
        )
        self.assertEqual(tuple(donors.shape), (4,))
        self.assertFalse(torch.any(donors == torch.arange(4)))
        self.assertGreaterEqual(diagnostics["selected_conditioned_risk"], 0.0)
        self.assertLessEqual(diagnostics["selected_conditioned_risk"], 1.0)


class RiskAnchorSelectorTest(unittest.TestCase):
    def test_pairs_are_full_size_nonself_and_disjoint_when_feasible(self):
        torch.manual_seed(3)
        logits = torch.zeros(4, 10, 2, 25, 2)
        audio_labels = torch.zeros(4, 10, 25)
        visual_labels = torch.zeros(4, 10, 25)
        for index in range(4):
            audio_labels[index, :, index] = 1
            visual_labels[index, :, index] = 1
        anchors, donors, diagnostic = runner.select_risk_anchor_pairs(
            logits, audio_labels, visual_labels, mix_fraction=0.5
        )
        self.assertEqual(tuple(anchors.shape), (4,))
        self.assertEqual(tuple(donors.shape), (4,))
        self.assertTrue(torch.all(anchors != donors))
        self.assertEqual(diagnostic["class_overlap_rate"], 0.0)
        self.assertEqual(diagnostic["self_pair_rate"], 0.0)

    def test_singleton_batch_is_safe(self):
        logits = torch.zeros(1, 10, 2, 25, 2)
        labels = torch.zeros(1, 10, 25)
        anchors, donors, diagnostic = runner.select_risk_anchor_pairs(logits, labels, labels)
        self.assertEqual(anchors.item(), 0)
        self.assertEqual(donors.item(), 0)
        self.assertEqual(diagnostic["self_pair_rate"], 1.0)

    def test_invalid_fraction_fails(self):
        logits = torch.zeros(2, 10, 2, 25, 2)
        labels = torch.zeros(2, 10, 25)
        with self.assertRaises(ValueError):
            runner.select_risk_anchor_pairs(logits, labels, labels, mix_fraction=0.0)


class TestGuardTest(unittest.TestCase):
    def test_test_path_is_blocked_without_calling_real_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.csv").resolve()
            runner._TEST_LABEL = path
            runner._TEST_ATTEMPTS = 0
            result = runner.validation_only_eval(None, None, None, str(path))
            self.assertEqual(len(result), 10)
            self.assertTrue(all(torch.isnan(torch.tensor(value)) for value in result))
            self.assertEqual(runner._TEST_ATTEMPTS, 1)


class GlobalUHNMTest(unittest.TestCase):
    def test_assignment_is_optimal_unique_derangement(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        uncertainty = torch.tensor([0.2, 0.9, 0.4, 0.7])
        donor_idx, diagnostics = runner.global_uhnm_assignment(
            features, uncertainty
        )
        self.assertEqual(donor_idx.unique().numel(), features.shape[0])
        self.assertFalse(torch.any(donor_idx == torch.arange(features.shape[0])))
        self.assertEqual(diagnostics["unique_donor_fraction"], 1.0)
        self.assertEqual(diagnostics["self_pair_rate"], 0.0)

        normalized = torch.nn.functional.normalize(features, dim=-1)
        score = normalized @ normalized.T * uncertainty.unsqueeze(0)
        actual = score[torch.arange(4), donor_idx].sum().item()
        feasible = [
            permutation
            for permutation in itertools.permutations(range(4))
            if all(row != column for row, column in enumerate(permutation))
        ]
        optimum = max(
            score[torch.arange(4), torch.tensor(permutation)].sum().item()
            for permutation in feasible
        )
        self.assertAlmostEqual(actual, optimum, places=6)

    def test_singleton_and_contract_validation(self):
        donor_idx, diagnostics = runner.global_uhnm_assignment(
            torch.ones(1, 3), torch.tensor([0.4])
        )
        torch.testing.assert_close(donor_idx, torch.tensor([0]))
        self.assertEqual(diagnostics["self_pair_rate"], 1.0)
        with self.assertRaises(ValueError):
            runner.global_uhnm_assignment(torch.ones(3, 2), torch.ones(2))

    def test_capacity_bounds_reuse_without_self_pairs(self):
        torch.manual_seed(12)
        features = torch.randn(9, 5)
        uncertainty = torch.linspace(0.1, 0.9, 9)
        donor_idx, diagnostics = runner.global_uhnm_assignment(
            features, uncertainty, donor_capacity=2
        )
        counts = torch.bincount(donor_idx, minlength=9)
        self.assertLessEqual(counts.max().item(), 2)
        self.assertFalse(torch.any(donor_idx == torch.arange(9)))
        self.assertEqual(diagnostics["max_donor_reuse"], 2.0)
        with self.assertRaises(ValueError):
            runner.global_uhnm_assignment(
                features, uncertainty, donor_capacity=0
            )


class MemoryAndTemporalSelectorTest(unittest.TestCase):
    def test_temporal_map_orders_same_video_by_time_gap(self):
        names = [
            "abcdefghijk_20_30",
            "abcdefghijk_0_10",
            "abcdefghijk_40_50",
            "zyxwvutsrqp_0_10",
        ]
        mapping = runner.build_temporal_neighbor_map(names, top_k=2)
        self.assertEqual(
            mapping["abcdefghijk_20_30"],
            ["abcdefghijk_0_10", "abcdefghijk_40_50"],
        )
        self.assertEqual(mapping["zyxwvutsrqp_0_10"], [])

    def test_global_memory_selector_excludes_anchor_names(self):
        audio = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]]
        )
        bank_audio = torch.cat([audio, torch.tensor([[[1.0, 1.0]]])], dim=0)
        bank_features = torch.nn.functional.normalize(bank_audio[:, 0], dim=-1)
        bank = {
            "audio": bank_audio,
            "features": bank_features,
            "risk": torch.tensor([0.1, 0.2, 0.9]),
            "Pa": torch.zeros(3, 25),
            "names": ["a", "b", "c"],
            "name_to_index": {"a": 0, "b": 1, "c": 2},
            "temporal_map": {"a": [], "b": [], "c": []},
        }
        selected_audio, selected_pa, selected, diagnostics = (
            runner._global_memory_bank_pairs(audio, bank, ["a", "b"])
        )
        self.assertEqual(tuple(selected_audio.shape), (2, 1, 2))
        self.assertEqual(tuple(selected_pa.shape), (2, 25))
        self.assertFalse(torch.any(selected == torch.tensor([0, 1])))
        self.assertEqual(diagnostics["self_pair_rate"], 0.0)

    def test_temporal_selector_reports_zero_coverage_and_falls_back(self):
        audio = torch.eye(2).reshape(2, 1, 2)
        bank = {
            "audio": audio,
            "features": torch.nn.functional.normalize(audio[:, 0], dim=-1),
            "risk": torch.tensor([0.2, 0.8]),
            "Pa": torch.zeros(2, 25),
            "names": ["a", "b"],
            "name_to_index": {"a": 0, "b": 1},
            "temporal_map": {"a": [], "b": []},
        }
        _, _, selected, diagnostics = runner._temporal_neighbor_pairs(
            audio, bank, ["a", "b"]
        )
        self.assertEqual(diagnostics["temporal_candidate_coverage"], 0.0)
        self.assertEqual(diagnostics["temporal_fallback_fraction"], 1.0)
        self.assertFalse(torch.any(selected == torch.tensor([0, 1])))


class PresenceAbsenceRoutingTest(unittest.TestCase):
    def test_masks_select_only_one_track_presence(self):
        donor_audio = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        anchor_visual = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        audio_mask, visual_mask = runner.presence_absence_masks(
            donor_audio, anchor_visual
        )
        torch.testing.assert_close(
            audio_mask, torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            visual_mask, torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        )

    def test_mask_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            runner.presence_absence_masks(torch.zeros(2, 4), torch.zeros(2, 3))

    def test_xor_mask_selects_both_directions(self):
        donor_audio = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        anchor_visual = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        mask = runner.presence_absence_xor_mask(donor_audio, anchor_visual)
        torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        selected_audio = donor_audio[mask.bool()]
        selected_visual = anchor_visual[mask.bool()]
        torch.testing.assert_close(
            selected_audio + selected_visual, torch.ones_like(selected_audio)
        )

    def test_xor_mask_shape_and_rank_validation(self):
        with self.assertRaises(ValueError):
            runner.presence_absence_xor_mask(
                torch.zeros(2, 4), torch.zeros(2, 3)
            )
        with self.assertRaises(ValueError):
            runner.presence_absence_xor_mask(
                torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)
            )

    def test_all_mask_matches_legacy_bce(self):
        torch.manual_seed(7)
        criterion = _legacy_edl()
        logits = torch.randn(3, 5, 2, requires_grad=True)
        target = torch.randint(0, 2, (3, 5)).float()
        expected = criterion(logits, target)
        actual = runner.masked_evidence_bce(
            criterion, logits, target, torch.ones_like(target)
        )
        # The selected reduction sums per-class values before dividing, while
        # PyTorch's fused global mean may accumulate float32 values in a
        # different order.  The observed gap is one float32 ULP.
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-7)

    def test_unselected_positions_have_zero_gradient(self):
        criterion = _legacy_edl()
        logits = torch.randn(2, 4, 2, requires_grad=True)
        target = torch.randint(0, 2, (2, 4)).float()
        mask = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        )
        loss = runner.masked_evidence_bce(criterion, logits, target, mask)
        loss.backward()
        self.assertEqual(logits.grad[mask == 0].abs().max().item(), 0.0)
        self.assertGreater(logits.grad[mask == 1].abs().sum().item(), 0.0)

    def test_empty_mask_is_finite_differentiable_zero(self):
        criterion = _legacy_edl()
        logits = torch.randn(2, 4, 2, requires_grad=True)
        target = torch.zeros(2, 4)
        loss = runner.masked_evidence_bce(
            criterion, logits, target, torch.zeros_like(target)
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(logits.grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
