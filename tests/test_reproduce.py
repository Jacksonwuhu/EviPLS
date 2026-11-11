"""CPU tests for command construction and output directory validation."""

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def make_args(action, *extra):
    return reproduce.parser().parse_args([action, "--run-id", "unit_s17", "--seed", "17", *extra])


class ReproductionTests(unittest.TestCase):
    def plan(self, action, *extra):
        with tempfile.TemporaryDirectory() as directory:
            return reproduce.build_plan(make_args(action, *extra), Path(directory))

    def test_llp_preserves_legacy_and_two_rounds(self):
        plan = self.plan("train-llp")
        command = plan["commands"][-1]
        config = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(command[0], sys.executable)
        for key, value in {"--seed": "17", "--loss_semantics": "legacy", "--epochs": "10",
                           "--ipr_rounds": "2", "--ipr_blend": "0.3", "--ipr_lr_decay": "0.5",
                           "--mixing": "uhnm", "--risk_loss": "none", "--ipr_nc_gamma": "0.0"}.items():
            self.assertEqual(config[key], value)
        self.assertNotIn("--uncertain", command)
        self.assertNotIn("--use_ema", command)
        self.assertEqual(plan["commands"][0][1:], ["-m", "nrep.validation"])

    def test_llp_evaluation_reuses_frozen_decoder(self):
        plan = self.plan("evaluate-llp", "--checkpoint", "models/frozen.best")
        command = plan["commands"][0]
        self.assertEqual(command[1], "experiments/legacy_scheduler/evaluate_test_checkpoint.py")
        self.assertEqual(plan["checkpoint"], str(Path(plan["cwd"]) / "models/frozen.best"))



    def test_rejects_traversal(self):
        args = make_args("train-llp")
        args.run_id = "../original"
        with self.assertRaises(ValueError):
            reproduce.build_plan(args)

    def test_rejects_each_existing_output(self):
        for parent in ("artifacts", "models", "logger"):
            with self.subTest(parent=parent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / parent / "unit_s17").mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    reproduce.build_plan(make_args("train-llp"), root)

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            (root / "artifacts").symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                reproduce.build_plan(make_args("train-llp"), root)

    def test_dry_run_does_not_execute_or_create_output(self):
        with mock.patch.object(reproduce, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as stream:
            reproduce.main(["evaluate-llp", "--run-id", "dry_test_unique", "--checkpoint", "missing.best", "--dry-run"])
        execute.assert_not_called()
        record = json.loads(stream.getvalue())
        self.assertFalse(Path(record["output_directories"][0]).exists())

    def test_invalid_budget_fails(self):
        args = make_args("train-llp", "--timeout-hours", "0")
        with self.assertRaises(ValueError):
            reproduce.build_plan(args)

    def test_every_generated_flag_is_declared_by_its_entry_point(self):
        for action in ("train-llp", "evaluate-llp",
                       'train-teacher', 'calibrate-teacher', 'generate-labels'):
            extra = ("--checkpoint", "missing.pt") if action.startswith("evaluate") or action == 'calibrate-teacher' else ()
            if action == 'train-teacher':
                extra = ('--data-root', 'data/UnAV')
            command = self.plan(action, *extra)["commands"][-1]
            tree = ast.parse((reproduce.ROOT / command[1]).read_text(encoding="utf-8"))
            declared = {arg.value for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument"
                        for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            with self.subTest(action=action):
                self.assertFalse({arg for arg in command[2:] if arg.startswith("--")} - declared)

    def test_execution_records_failure_without_reusing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frozen.best").write_bytes(b"not loaded in this mocked test")
            plan = reproduce.build_plan(make_args("evaluate-llp", "--checkpoint", "frozen.best"), root)
            with mock.patch.object(reproduce, "run_bounded", side_effect=subprocess.CalledProcessError(2, ["mock"])) as run:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                    reproduce.execute(plan)
            self.assertEqual(run.call_args.kwargs["cwd"], str(root))
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONHASHSEED"], "17")
            record = json.loads((root / "artifacts/unit_s17/run.json").read_text())
            self.assertEqual(record["state"], "failed")
            with self.assertRaises(FileExistsError):
                reproduce.build_plan(make_args("evaluate-llp", "--checkpoint", "frozen.best"), root)


if __name__ == "__main__":
    unittest.main()
