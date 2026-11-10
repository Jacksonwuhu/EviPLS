import unittest

import torch

from nets.criterion import EvidenceLoss
from nrep.runtime import capture_random_state, seed_everything
from nrep.uncertainty import compute_uncertainty, hard_negative_mining


class UncertaintyTests(unittest.TestCase):
    def test_zero_logits_match_subjective_logic(self):
        logits = torch.zeros(3, 4, 2)
        actual = compute_uncertainty(logits)
        torch.testing.assert_close(actual, torch.full((3, 4), 2.0 / 3.0))

    def test_zero_logits_match_strict_edl_prior(self):
        logits = torch.zeros(3, 4, 2)
        actual = compute_uncertainty(logits, prior_strength=2.0)
        torch.testing.assert_close(actual, torch.full((3, 4), 0.5))

    def test_singleton_hard_negative_is_safe(self):
        features = torch.randn(1, 10, 8)
        selected = hard_negative_mining(features, torch.tensor([0.5]))
        self.assertEqual(selected.tolist(), [0])

    def test_constant_risk_reduces_hnm_to_similarity_matching(self):
        # With equal risk, the multiplicative score is a positive constant
        # times cosine similarity. Item 0 and 1 are closest; item 2 is
        # orthogonal to both.
        features = torch.tensor([
            [[1.0, 0.0]],
            [[0.9, 0.1]],
            [[0.0, 1.0]],
        ])
        selected = hard_negative_mining(features, torch.ones(3))
        self.assertEqual(selected.tolist(), [1, 0, 1])

    def test_random_state_is_cpu_safe(self):
        seed_everything(7)
        state = capture_random_state()
        self.assertIn("cuda_rng", state)

    def test_invalid_evidence_name_fails_clearly(self):
        with self.assertRaises(ValueError):
            compute_uncertainty(torch.zeros(1, 2), "unknown")

    def test_paper_probability_is_normalized(self):
        loss = EvidenceLoss(legacy_normalization=False)
        probability, uncertainty = loss.get_predictions(torch.zeros(2, 2))
        torch.testing.assert_close(probability.sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(uncertainty, torch.full((2, 1), 0.5))

    def test_legacy_probability_remains_reproducible(self):
        loss = EvidenceLoss(legacy_normalization=True)
        probability, _ = loss.get_predictions(torch.zeros(1, 2))
        torch.testing.assert_close(probability, torch.full((1, 2), 0.25))


if __name__ == "__main__":
    unittest.main()
