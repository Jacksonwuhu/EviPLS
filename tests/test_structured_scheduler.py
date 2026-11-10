"""Unittest coverage for the isolated structured U-HNM selector."""

import unittest

import torch

from experiments.legacy_scheduler.runner import select_structured_uhnm_pairs


class StructuredSchedulerTest(unittest.TestCase):
    def _batch(self):
        batch, time, classes, dim = 4, 2, 4, 8
        audio = torch.randn(batch, time, dim)
        logits = torch.zeros(batch, time, 2, classes, 2)
        audio_pseudo = torch.zeros(batch, time, classes)
        visual_pseudo = torch.zeros(batch, time, classes)
        for index in range(batch):
            audio_pseudo[index, :, index] = 1.0
            visual_pseudo[index, :, index] = 1.0
        return audio, logits, audio_pseudo, visual_pseudo

    def test_shape_and_constraints(self):
        result = select_structured_uhnm_pairs(*self._batch())
        anchors, donors, diagnostics = result
        self.assertEqual(tuple(anchors.shape), (4,))
        self.assertEqual(tuple(donors.shape), (4,))
        self.assertEqual(diagnostics["self_pair_rate"], 0.0)
        self.assertEqual(diagnostics["class_overlap_rate"], 0.0)
        self.assertEqual(diagnostics["constraint_relaxation_rate"], 0.0)
        self.assertGreaterEqual(diagnostics["anchor_risk"], diagnostics["donor_risk"])

    def test_rejects_segment_logits_without_time_axis(self):
        audio, logits, audio_pseudo, visual_pseudo = self._batch()
        with self.assertRaises(ValueError):
            select_structured_uhnm_pairs(
                audio,
                logits[:, 0],
                audio_pseudo[:, 0],
                visual_pseudo[:, 0],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
