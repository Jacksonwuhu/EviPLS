#!/usr/bin/env python3
"""Portable, non-overwriting launchers for the frozen main_v3 protocols.

Dry runs only print commands. Numerical implementations are imported/executed
with the original default numerical recipe. LLP training retains its historical
end-of-round test evaluations, which must not be used to tune the recipe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def options(values: dict) -> list[str]:
    return [part for key, value in values.items() for part in ("--" + key, str(value))]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    for action in ("train-teacher", "calibrate-teacher", "generate-labels",
                   "train-llp", "evaluate-llp"):
        command = commands.add_parser(action)
        command.add_argument("--run-id", required=True, help="new output name; never reused")
        command.add_argument("--seed", type=int, default=1000 if action == 'train-teacher' else 6)
        command.add_argument("--dry-run", action="store_true", help="print only; no imports or writes")
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--timeout-hours", type=float, default=24.0 if action == 'train-teacher' else 4.0)
        if action.startswith("evaluate") or action in ('calibrate-teacher', 'generate-labels'):
            command.add_argument("--checkpoint", type=Path, required=action != 'generate-labels',
                                 help="absolute path, or relative to this package")
        if action.endswith('llp'):
            command.add_argument('--label-root', type=Path, help='generated labels directory containing audio/ and visual/ for ALL LLP splits')
        if action in ('train-teacher', 'calibrate-teacher', 'generate-labels', 'train-llp'):
            command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if action == 'train-teacher':
            command.add_argument('--data-root', type=Path, default=Path("data/UnAV"), help='UnAV feature/annotation root prepared by prepare_data.py')
            command.add_argument('--check-only', action='store_true')
        if action == 'train-llp':
            command.add_argument('--epochs', type=int, default=10)
            command.add_argument('--batch-size', type=int, default=128)
            command.add_argument('--workers', type=int, default=5)
        if action == 'generate-labels':
            command.add_argument('--threshold-dir', type=Path)
            command.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='all')
            command.add_argument('--limit', type=int, help='generate the first N videos for an inference check; full training requires all videos')
    return result


def build_plan(args: argparse.Namespace, root: Path = ROOT) -> dict:
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run ID must begin with a letter/digit and contain only letters, digits, _, - or .")
    if args.threads < 1 or not 0 < args.timeout_hours <= 168:
        raise ValueError("threads must be positive; timeout-hours must be in (0, 168]")
    if args.action == "train-llp" and (args.epochs < 1 or args.batch_size < 2 or args.workers < 0):
        raise ValueError("epochs must be positive, batch-size >= 2, workers >= 0")
    output = root / "artifacts" / args.run_id
    outputs = [output]
    prefix = [sys.executable]
    data = dict(audio_dir="data/CLAP/features", video_dir="data/CLIP/features",
                st_dir="pretrain/data/LLP/feats/r2plus1d_18",
                a_pseudo_data_dir="data/CLAP/a_segmentwise_pseudo_labels",
                v_pseudo_data_dir="data/CLIP/v_segmentwise_pseudo_labels")
    checkpoint = None
    if getattr(args, 'label_root', None):
        label_root = args.label_root.expanduser()
        label_root = (root / label_root).resolve() if not label_root.is_absolute() else label_root.resolve()
        data.update(a_pseudo_data_dir=str(label_root / 'audio'), v_pseudo_data_dir=str(label_root / 'visual'))
    if getattr(args, 'checkpoint', None):
        checkpoint = args.checkpoint.expanduser()
        checkpoint = (root / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
    if args.action == 'train-teacher':
        unav = args.data_root.expanduser()
        unav = (root / unav).resolve() if not unav.is_absolute() else unav.resolve()
        config = {'data-root': unav, 'output': output / 'teacher', 'seed': args.seed,
                  'device': args.device, 'threads': args.threads, 'timeout-seconds': args.timeout_hours * 3600}
        commands = [prefix + ['teacher_train.py'] + options(config) + (['--check-only'] if args.check_only else [])]
    elif args.action in ('calibrate-teacher', 'generate-labels'):
        config = {'output': output / ('thresholds' if args.action == 'calibrate-teacher' else 'labels'),
                  'device': args.device, 'threads': args.threads, 'timeout-seconds': args.timeout_hours * 3600}
        if checkpoint:
            config['checkpoint'] = checkpoint
        if args.action == 'generate-labels':
            config['split'] = args.split
            if args.limit is not None:
                config['limit'] = args.limit
            if args.threshold_dir:
                config['threshold-dir'] = args.threshold_dir
        script = 'calibrate_teacher.py' if args.action == 'calibrate-teacher' else 'generate_labels.py'
        commands = [prefix + [script] + options(config)]
    elif args.action == "train-llp":
        outputs += [root / "models" / args.run_id, root / "logger" / args.run_id]
        config = dict(mode="train_with_pseudo_labels", checkpoint=args.run_id,
                      model_save_dir="models", logger=f"logger/{args.run_id}/",
                      epochs=args.epochs, batch_size=args.batch_size, refine_batch_size=args.batch_size,
                      device=args.device, num_workers=args.workers, eval_num_workers=min(1, args.workers),
                      refined_output_root=str(output / "pseudo_labels"), **data,
                      label_train="data/AVVP_train.csv", label_val="data/AVVP_val_pd.csv",
                      label_test="data/AVVP_test_pd.csv", lr=5e-4, lr_step_size=6,
                      lr_gamma=0.25, seed=args.seed, num_layers=1, temperature=0.2,
                      lamda_1=1.0, lamda_2=0.01, lamda_3=1.0, lamda_4=0.01,
                      audio_factor=2.0, video_factor=3.0, evidence="exp", loss_type="bce",
                      loss_semantics="legacy", brier_weight=0.0, edl_temperature=1.0,
                      risk_loss="none", mixing="uhnm", ipr_rounds=2, ipr_blend=0.3,
                      ipr_temperature=1.0, ipr_lr_decay=0.5, ipr_mode="risk_aware",
                      ipr_gate="low", ipr_risk="vacuity", risk_proxy="edl",
                      ipr_sparsity=0.0, ipr_nc_gamma=0.0,
                      best_audio_metric="audio_seg", best_video_metric="avg_event_eve",
                      log_interval=700)
        preflight = prefix + ['-m', 'nrep.validation']
        if getattr(args, 'label_root', None):
            preflight += ['--label-root', str(label_root)]
        commands = [preflight, prefix + ["main_ipr.py"] + options(config)]
    elif args.action == "evaluate-llp":
        config = dict(checkpoint=checkpoint, output=output / "metrics.json", **data,
                      label_test="data/AVVP_test_pd.csv", num_layers=1, temperature=0.2, num_workers=0)
        commands = [prefix + ["experiments/legacy_scheduler/evaluate_test_checkpoint.py"] + options(config)]
    for path in outputs:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"output escapes package through a symlink: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}; choose a new --run-id")
    return dict(action=args.action, run_id=args.run_id, seed=args.seed, cwd=str(root),
                commands=commands, output_directories=[str(p) for p in outputs],
                checkpoint=str(checkpoint) if checkpoint else None,
                device=getattr(args, 'device', 'cuda' if args.action != 'evaluate-llp' else 'auto'),
                check_only=getattr(args, 'check_only', False),
                label_root=str(label_root) if getattr(args, 'label_root', None) else None,
                timeout_seconds=args.timeout_hours * 3600,
                environment={"OMP_NUM_THREADS": str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
                             'OPENBLAS_NUM_THREADS': str(args.threads), "PYTHONHASHSEED": str(args.seed)},
                protocol="EviPLS LLP legacy; U-HNM + IPR eta=0.3; two rounds",
                note="LLP training evaluates test after each round; model selection uses validation only.")


def run_bounded(command, *, cwd, env, stdout, timeout):
    """Stop the entire process group, including loader workers, on timeout."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        # A parent can exit before its worker processes; always clean the group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def execute(plan: dict) -> None:
    if plan["checkpoint"] and not Path(plan["checkpoint"]).is_file():
        raise FileNotFoundError(f"checkpoint not found: {plan['checkpoint']}")
    if plan['check_only']:
        subprocess.run(plan['commands'][0], cwd=plan['cwd'], env=dict(os.environ, **plan['environment']),
                       check=True, timeout=plan['timeout_seconds'])
        return
    if plan['label_root']:
        import csv
        import numpy as np
        label_root = Path(plan['label_root'])
        generation_manifest = label_root / 'manifest.json'
        if generation_manifest.is_file():
            record = json.loads(generation_manifest.read_text())
            if record.get('state') != 'complete' or record.get('split') != 'all' or record.get('limit') is not None:
                raise ValueError('--label-root generation must be complete, all-split and unlimited')
        for split in ('AVVP_train.csv', 'AVVP_val_pd.csv', 'AVVP_test_pd.csv'):
            with (Path(plan['cwd']) / 'data' / split).open(newline='') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    for modality in ('audio', 'visual'):
                        path = label_root / modality / (row['filename'][:11] + '.npy')
                        if not path.is_file():
                            raise FileNotFoundError(f'--label-root requires all LLP splits; missing {path}')
                        array = np.load(path, allow_pickle=False)
                        if array.shape != (10, 25) or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
                            raise ValueError(f'invalid custom pseudo-label: {path}')
    if plan['device'] == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select --device cpu for a bounded diagnostic or use a GPU host")
    output = Path(plan["output_directories"][0])
    output.mkdir(parents=True, exist_ok=False)
    record = dict(plan, started_utc=datetime.now(timezone.utc).isoformat(), state="running")
    manifest = output / "run.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output}\nLog: {output / 'launcher.log'}", flush=True)
    started = time.monotonic()
    try:
        with (output / "launcher.log").open("w", encoding="utf-8") as log:
            for command in plan["commands"]:
                remaining = plan["timeout_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("run time budget exhausted")
                run_bounded(command, cwd=plan["cwd"], env=dict(os.environ, **plan["environment"]),
                            stdout=log, timeout=remaining)
        record["state"] = "complete"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        argument_parser.exit(1, f"reproduce: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
