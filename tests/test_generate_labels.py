"""Small CPU-only teacher label adapter tests; no checkpoint is loaded."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("generate_labels", Path(__file__).resolve().parents[1] / "generate_labels.py")
generate_labels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_labels)


class TeacherLabelTests(unittest.TestCase):
    def test_threshold_is_strict_and_weak_mask_is_respected(self):
        logits = np.ones((10, 25), dtype=np.float32)
        thresholds = np.ones(25, dtype=np.float32)
        weak = np.ones(25, dtype=bool)
        logits[:, 0] = 2
        logits[:, 1] = 2
        weak[1] = False
        labels = generate_labels.threshold_labels(logits, thresholds, weak)
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(int(labels.sum()), 10)
        self.assertTrue(np.all(labels[:, 0] == 1))

    def test_negative_threshold_does_not_restore_absent_class(self):
        labels = generate_labels.threshold_labels(np.ones((10, 25)), np.full(25, -np.inf), np.zeros(25))
        self.assertEqual(float(labels.sum()), 0)

    def test_bad_shape_and_nan_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_labels.threshold_labels(np.zeros((1, 25)), np.zeros(25), np.ones(25))
        with self.assertRaises(ValueError):
            generate_labels.threshold_labels(np.full((10, 25), np.nan), np.zeros(25), np.ones(25))

    def test_outputs_are_package_relative_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(generate_labels.resolve_output(Path("new_labels"), root), root / "new_labels")
            (root / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                generate_labels.resolve_output(Path("existing"), root)
            with self.assertRaises(ValueError):
                generate_labels.resolve_output(Path("../escape"), root)

    def test_limit_and_split_order_use_weak_labels_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/AVVP_train.csv").write_text("filename\tevent_labels\n12345678901_0_10\tSpeech,Dog\n", encoding="utf-8")
            # limit=1 must not attempt to open absent validation/test/dense GT.
            records = generate_labels.select_records("all", 1, root)
            self.assertEqual(records, [{"split": "train", "video": "12345678901", "filename": "12345678901_0_10", "weak_labels": ["Speech", "Dog"]}])
            with self.assertRaises(ValueError):
                generate_labels.select_records("all", 0, root)

    def test_comparison_reports_differences_without_changing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.npy"
            labels = np.zeros((10, 25), dtype=np.float32)
            self.assertFalse(generate_labels.frozen_comparison(labels, path)["available"])
            np.save(path, labels)
            self.assertTrue(generate_labels.frozen_comparison(labels, path)["exact_equal"])
            labels[0, 0] = 1
            result = generate_labels.frozen_comparison(labels, path)
            self.assertFalse(result["exact_equal"])
            self.assertEqual(result["differing_positions"], 1)


if __name__ == "__main__":
    unittest.main()
