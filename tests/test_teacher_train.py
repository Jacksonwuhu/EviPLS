"""Small, one-thread CPU tests; never instantiate the full five-layer teacher."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import teacher_train as teacher
from pretrain.dataloader import UnAV_dataset, build_dataloader
from pretrain.model.TemporalTransformer import TemporalTransformer


def make_fixture(root):
    rng = np.random.RandomState(7)
    for relative in teacher.INPUT_LAYOUT.values():
        path = root / relative
        (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
    for name in ("audio", "rgb", "flow"):
        (root / name).mkdir()
    database = {}
    for index, (name, subset, duration, length) in enumerate((
        ("train_short", "train", 2.2, 5), ("train_long", "train", 4, 7),
        ("validation_one", "validation", 3, 4), ("test_not_loaded", "test", 3, 3),
    )):
        database[name] = dict(subset=subset, duration=duration,
                              annotations=[dict(label=teacher.UNAV_CATEGORIES[index], label_id=index,
                                                segment=[0.2, 1.2])])
        if subset == "test":
            continue  # No TEST feature is needed or inspected during training.
        for key, dim in (("clap_segments", 512), ("clip_segments", 768)):
            dtype = np.float64 if name == "train_short" else np.float32
            np.save(root / teacher.INPUT_LAYOUT[key] / (name + ".npy"),
                    rng.randn(length, dim).astype(dtype))
        for directory, suffix in (("audio", "_vggish.npy"), ("rgb", "_rgb.npy"), ("flow", "_flow.npy")):
            np.save(root / directory / (name + suffix), rng.randn(length, 2).astype(np.float32))
    (root / teacher.INPUT_LAYOUT["annotations"]).write_text(json.dumps(dict(database=database)))
    for key, dim in (("clap_events", 512), ("clip_events", 768)):
        np.save(root / teacher.INPUT_LAYOUT[key], (rng.randn(100, dim) * 0.01).astype(np.float32))


class TeacherTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "UnAV"
        make_fixture(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def args(self, *extra):
        return teacher.prepare_args(teacher.parser().parse_args([
            "--data-root", str(self.root), "--output", str(Path(self.directory.name) / "run"),
            "--device", "cpu", *extra]))

    def tiny_model(self):
        # Same forward and loss code; deliberately tiny temporal backbone.
        return TemporalTransformer(input_v_2d_dim=768, input_v_3d_dim=1024, input_a_dim=512,
                                   hidden_dim=8, nhead=2, ff_dim=16, num_layers=1, dropout=0,
                                   clap_event_feats=torch.from_numpy(np.load(self.root / teacher.INPUT_LAYOUT["clap_events"])),
                                   clip_event_feats=torch.from_numpy(np.load(self.root / teacher.INPUT_LAYOUT["clip_events"])))

    def test_original_recipe_defaults(self):
        args = self.args()
        self.assertEqual((args.seed, args.epochs, args.batch_size, args.num_layers), (1000, 80, 64, 5))
        self.assertEqual((args.hidden_dim, args.nhead, args.ff_dim, args.dropout), (1024, 16, 2048, 0.3))
        self.assertEqual((args.lr, args.lr_min, args.warm_up_epoch, args.grad_norm), (1e-4, 1e-5, 10, 2.0))
        self.assertEqual((args.pos_weight, args.neg_weight, args.beta1, args.weight_decay), (5, 1, 0.5, 1e-3))
        self.assertEqual(args.num_workers, 0)

    def test_old_and_feature_only_data_and_collation_are_identical(self):
        consumed = ("clap_feat", "clip_feat", "weak_label", "gt_label", "attn_mask", "valid_mask")
        for split in ("train", "val"):
            old = UnAV_dataset(split, str(self.root / teacher.INPUT_LAYOUT["annotations"]),
                               str(self.root / "audio"), str(self.root / "rgb"), str(self.root / "flow"),
                               str(self.root / teacher.INPUT_LAYOUT["clap_segments"]),
                               str(self.root / teacher.INPUT_LAYOUT["clip_segments"]))
            new = teacher.FeatureOnlyUnAVDataset(split, self.root)
            self.assertEqual(old.data_list, new.data_list)
            for index in range(len(new)):
                previous, current = old[index], new[index]
                for key in consumed:
                    self.assertTrue(torch.equal(previous[key], current[key]), (split, index, key))
                for key in ("duration", "video_name"):
                    self.assertEqual(previous[key], current[key])
                self.assertNotIn("audio", current)
            args = self.args()
            torch.manual_seed(1000)
            previous = next(iter(build_dataloader(args, old, split)))
            torch.manual_seed(1000)
            current = next(iter(build_dataloader(args, new, split)))
            for key in consumed + ("duration",):
                self.assertTrue(torch.equal(previous[key], current[key]), (split, key))
            self.assertEqual(previous["video_name"], current["video_name"])

    def test_preflight_checks_actual_inputs_only_and_no_test_features(self):
        report = teacher.preflight(self.root)
        self.assertEqual(report["split_counts"], dict(train=2, validation=1))
        self.assertEqual(report["max_quantized_duration"], dict(train=4, validation=3))
        self.assertEqual(report["feature_files_checked"], 6)
        self.assertFalse(report["test_features_read"])
        self.assertEqual(report["shapes"]["clap_events"], [100, 512])

    def test_marked_diagnostic_subset_is_not_formal_training_input(self):
        path = self.root / teacher.INPUT_LAYOUT["annotations"]
        document = json.loads(path.read_text())
        document["verification_only_subset"] = True
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(teacher.PreflightError, "diagnostic subset"):
            teacher.preflight(self.root)

    def test_missing_unav_is_explicit_and_writes_nothing(self):
        missing = Path(self.directory.name) / "missing"
        output = Path(self.directory.name) / "unused_output"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = teacher.main(["--data-root", str(missing), "--output", str(output), "--check-only"])
        self.assertEqual(result, 2)
        self.assertIn("Missing UnAV teacher inputs", stderr.getvalue())
        self.assertIn("unav100_annotations.json", stderr.getvalue())
        self.assertIn("feats_CLAP/event_feats/all_event_feats.npy", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_bad_feature_shape_rejected(self):
        np.save(self.root / teacher.INPUT_LAYOUT["clap_segments"] / "train_short.npy", np.zeros((2, 128), np.float32))
        with self.assertRaisesRegex(teacher.PreflightError, "512"):
            teacher.preflight(self.root)

    def test_bad_event_embedding_contract_rejected(self):
        np.save(self.root / teacher.INPUT_LAYOUT["clip_events"], np.zeros((25, 768), np.float32))
        with self.assertRaisesRegex(teacher.PreflightError, "100, 768"):
            teacher.preflight(self.root)

    def test_check_only_does_not_construct_model_or_write_output(self):
        output = Path(self.directory.name) / "never_created"
        with mock.patch.object(teacher, "build_model", side_effect=AssertionError("must not construct model")):
            with contextlib.redirect_stdout(io.StringIO()):
                result = teacher.main(["--data-root", str(self.root), "--output", str(output), "--check-only"])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())

    def test_refuses_existing_output_and_input_descendant(self):
        with self.assertRaisesRegex(teacher.PreflightError, "existing output"):
            self.args("--output", self.directory.name)
        with self.assertRaisesRegex(teacher.PreflightError, "input directory"):
            self.args("--output", str(self.root / "run"))
        link = Path(self.directory.name) / "dangling_output"
        link.symlink_to(Path(self.directory.name) / "missing_target")
        with self.assertRaisesRegex(teacher.PreflightError, "symlink output"):
            self.args("--output", str(link))

    def test_original_loss_matches_explicit_formula_and_ignores_padding(self):
        args, model = self.args(), self.tiny_model()
        labels = torch.zeros(2, 4, 100)
        labels[0, :2, 0] = 1
        valid = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.]])
        audio, visual = torch.randn(2, 4, 100), torch.randn(2, 4, 100)
        loss, _ = model.calculate_loss(args, (audio, visual), labels, valid)
        weights = torch.where(labels.sum(1, keepdim=True) > 0, 5, 1)
        expected = (F.binary_cross_entropy(audio.sigmoid() * visual.sigmoid(), labels, reduction="none") *
                    weights * valid.unsqueeze(-1)).sum() / (valid.sum() * 100)
        self.assertTrue(torch.equal(loss, expected))
        audio[0, 3] = 20
        visual[0, 3] = -20
        padded_loss, _ = model.calculate_loss(args, (audio, visual), labels, valid)
        self.assertTrue(torch.equal(loss, padded_loss))

    def test_one_real_tiny_cpu_optimizer_step_and_validation(self):
        args = self.args()
        model = self.tiny_model()
        optimizer, scheduler = teacher.make_optimizer_scheduler(args, model)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        original = model.proj_a.layers[0].weight.detach().clone()
        loader = build_dataloader(args, teacher.FeatureOnlyUnAVDataset("train", self.root), "train")
        loss = teacher.train_epoch(args, model, loader, optimizer, torch.device("cpu"))
        self.assertTrue(np.isfinite(loss["loss_all"]))
        self.assertFalse(torch.equal(original, model.proj_a.layers[0].weight))
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-5)
        validation = build_dataloader(args, teacher.FeatureOnlyUnAVDataset("val", self.root), "val")
        scores, loss = teacher.evaluate_validation(args, model, validation, torch.device("cpu"))
        self.assertTrue(np.isfinite(scores["Seg-av"]))
        self.assertTrue(np.isfinite(loss["loss_all"]))
        with self.assertRaisesRegex(ValueError, "validation split"):
            teacher.evaluate_validation(args, model, loader, torch.device("cpu"))

    def test_best_checkpoint_uses_strict_validation_improvement(self):
        args = self.args("--epochs", "3", "--warm-up-epoch", "1")
        report = teacher.preflight(self.root)
        values = [(dict(**{"Seg-av": score}), dict(loss_all=0.1)) for score in (0., 55., 55.)]
        with mock.patch.object(teacher, "build_model", return_value=torch.nn.Linear(1, 1)):
            with mock.patch.object(teacher, "train_epoch", return_value=dict(loss_all=0.1)):
                with mock.patch.object(teacher, "evaluate_validation", side_effect=values):
                    state = teacher.run_training(args, report, lambda: None)
        self.assertEqual(state["best_epoch"], 2)
        self.assertEqual(state["best_seg_av"], 55.)
        self.assertFalse(state["test_used"])
        best = args.output / "checkpoints/checkpoint_best.pt"
        self.assertEqual(set(torch.load(best, map_location="cpu")), {"weight", "bias"})
        self.assertEqual(len((args.output / "metrics.jsonl").read_text().splitlines()), 3)

    def test_timeout_check_is_bounded(self):
        with mock.patch.object(teacher.time, "monotonic", side_effect=[0., 2.]):
            with teacher.bounded_runtime(1) as check:
                with self.assertRaises(TimeoutError):
                    check()


if __name__ == "__main__":
    unittest.main()
