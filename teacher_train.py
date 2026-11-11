#!/usr/bin/env python3
"""Train the UnAV100 teacher from CLAP and CLIP features.

Uses temporal annotations, variable-length batching and a TemporalTransformer.
Checkpoint selection uses UnAV validation scores; run outputs are stored locally.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch

from pretrain.categories import UNAV_CATEGORIES
from pretrain.dataloader import UnAV_dataset, build_dataloader
from pretrain.model import build_model
from pretrain.utils.eval_metrics import event_level, segment_level
from pretrain.utils.eval_utils import AVVPEvaluator
from pretrain.utils.train_utils import AverageMeter, WarmUpCosineAnnealingLR

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = dict(model="TemporalTransformer", input_a_dim=512, input_v_2d_dim=768,
                    input_v_3d_dim=1024, hidden_dim=1024, nhead=16, ff_dim=2048,
                    num_layers=5, dropout=0.3, pre_norm=False, pos_weight=5,
                    neg_weight=1, beta1=0.5, beta2=0.999, eps=1e-8,
                    dataset="UnAV", num_classes=100)
INPUT_LAYOUT = {
    "annotations": "unav100_annotations.json",
    "clap_segments": "feats_CLAP/segment_feats",
    "clip_segments": "feats_CLIP/segment_feats",
    "clap_events": "feats_CLAP/event_feats/all_event_feats.npy",
    "clip_events": "feats_CLIP/event_feats/all_event_feats.npy",
}


class PreflightError(ValueError):
    pass


class FeatureOnlyUnAVDataset(UnAV_dataset):
    """Original UnAV dataset with only the model-consumed feature reads."""
    def __init__(self, mode, data_root):
        root = Path(data_root)
        super().__init__(mode, str(root / INPUT_LAYOUT["annotations"]),
                         None, None, None, str(root / INPUT_LAYOUT["clap_segments"]),
                         str(root / INPUT_LAYOUT["clip_segments"]))

    def __getitem__(self, idx):
        item = self.data_list[idx]
        duration = item["quantized duration"]
        features = {}
        for key, directory in (("clap_feat", self.clap_feat_dir), ("clip_feat", self.clip_feat_dir)):
            array = self.load_numpy_file(os.path.join(directory, item["id"] + ".npy")).astype(np.float32)
            tensor = self.avgpool_feature(torch.from_numpy(array), duration)
            features[key] = self.pad_feature(tensor, target_len=self.max_quantized_duration)
        attn_mask, valid_mask, gt_label = self.label_manager.get_labels(item["id"])
        features.update(weak_label=torch.where(gt_label.sum(dim=0) > 0, 1, 0).float(),
                        video_name=item["id"], duration=duration, attn_mask=attn_mask,
                        valid_mask=valid_mask, gt_label=gt_label)
        return features


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output", type=Path, help="New run directory; never reuse an existing path")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--seed", type=int, default=1000)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--lr-min", type=float, default=1e-5)
    result.add_argument("--warm-up-epoch", type=int, default=10)
    result.add_argument("--weight-decay", type=float, default=1e-3)
    result.add_argument("--grad-norm", type=float, default=2.0)
    result.add_argument("--save-interval", type=int, default=20)
    result.add_argument("--num-workers", type=int, default=0,
                        help="Operational default 0, original script default 8; no random data augmentation")
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--timeout-seconds", type=float, default=86400)
    result.add_argument("--check-only", action="store_true", help="Validate inputs without constructing a model or writing files")
    return result


def root_path(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def prepare_args(args):
    args.data_root = root_path(args.data_root)
    if args.output is not None:
        raw_output = args.output.expanduser()
        raw_output = raw_output if raw_output.is_absolute() else ROOT / raw_output
        if raw_output.is_symlink():
            raise PreflightError(f"Refusing symlink output: {raw_output}")
        args.output = root_path(args.output)
        if args.output.exists():
            raise PreflightError(f"Refusing existing output: {args.output}")
        if args.output.is_relative_to(args.data_root):
            raise PreflightError("Training output must not be inside the UnAV input directory")
    elif not args.check_only:
        raise PreflightError("--output is required unless --check-only is used")
    for name in ("epochs", "batch_size", "warm_up_epoch", "save_interval", "threads"):
        if getattr(args, name) <= 0:
            raise PreflightError(f"--{name.replace('_', '-')} must be positive")
    if args.epochs <= args.warm_up_epoch:
        raise PreflightError("--epochs must exceed --warm-up-epoch for the original cosine scheduler")
    if args.num_workers < 0 or args.grad_norm < 0 or args.weight_decay < 0:
        raise PreflightError("workers, gradient clip and weight decay cannot be negative")
    if not (0 < args.lr_min <= args.lr and math.isfinite(args.lr)):
        raise PreflightError("Learning rates must be finite and satisfy 0 < lr-min <= lr")
    if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
        raise PreflightError("--timeout-seconds must be finite and positive")
    for name, value in MODEL_CONFIG.items():
        setattr(args, name, value)
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(data_root, check_budget=lambda: None):
    """Check all train/validation feature values; never inspect TEST features."""
    root = Path(data_root)
    paths = {key: root / relative for key, relative in INPUT_LAYOUT.items()}
    missing = [str(path) for key, path in paths.items()
               if not (path.is_dir() if key.endswith("segments") else path.is_file())]
    if missing:
        raise PreflightError("Missing UnAV teacher inputs (not included in the LLP student data):\n  " +
                             "\n  ".join(missing))
    for key, dim in (("clap_events", 512), ("clip_events", 768)):
        array = np.load(paths[key], allow_pickle=False, mmap_mode="r")
        if array.shape != (100, dim) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise PreflightError(f"{paths[key]} must be finite float32 (100, {dim}); got {array.shape}/{array.dtype}")
    with paths["annotations"].open(encoding="utf-8") as source:
        document = json.load(source)
    if document.get("verification_only_subset"):
        raise PreflightError("diagnostic subset cannot be used as formal UnAV training input")
    database = document.get("database")
    if not isinstance(database, dict) or not database:
        raise PreflightError("unav100_annotations.json must contain a nonempty database mapping")
    counts = dict(train=0, validation=0)
    max_duration = dict(train=0, validation=0)
    failures, failure_count, checked_files = [], 0, 0
    for video_id, item in database.items():
        check_budget()
        if not isinstance(video_id, str) or video_id in ("", ".", "..") or any(c in video_id for c in "/\\"):
            raise PreflightError(f"Unsafe annotation video ID: {video_id!r}")
        annotations = item.get("annotations")
        if not isinstance(annotations, list) or any("label" not in a or "label_id" not in a for a in annotations):
            raise PreflightError(f"{video_id}: original loader requires annotations with label and label_id")
        subset = item.get("subset", "").lower()
        if not subset:
            raise PreflightError(f"{video_id}: subset must be nonempty (train/validation/test)")
        # Deliberately preserve the original loader's subset membership rule.
        modes = [mode for mode in counts if subset and subset in mode]
        if not modes:
            continue
        duration = item.get("duration")
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
            raise PreflightError(f"{video_id}: duration must be a positive number of seconds")
        quantized = math.ceil(duration)
        if not annotations:
            raise PreflightError(f"{video_id}: selected video has no annotations; original loader cannot consume it")
        for annotation in annotations:
            label_id = annotation["label_id"]
            segment = annotation.get("segment", [])
            if type(label_id) is not int or not 0 <= label_id < 100:
                raise PreflightError(f"{video_id}: label_id must index the fixed 100-class category order")
            if type(label_id) is int and 0 <= label_id < 100 and annotation["label"] != UNAV_CATEGORIES[label_id]:
                raise PreflightError(f"{video_id}: label name/order mismatch")
            if (len(segment) != 2 or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in segment)
                    or not 0 <= segment[0] < segment[1] <= quantized):
                raise PreflightError(f"{video_id}: invalid annotation segment {segment!r}")
        for mode in modes:
            counts[mode] += 1
            max_duration[mode] = max(max_duration[mode], quantized)
        for key, dim in (("clap_segments", 512), ("clip_segments", 768)):
            path = paths[key] / (video_id + ".npy")
            try:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dim or array.dtype.kind not in "fiu":
                    raise ValueError(f"expected numeric (L>=1, {dim}), got {array.shape}/{array.dtype}")
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite segment features")
                checked_files += 1
            except (OSError, ValueError, EOFError) as error:
                failure_count += 1
                if len(failures) < 20:
                    failures.append(f"{path}: {error}")
    if any(count == 0 for count in counts.values()):
        raise PreflightError(f"Need nonempty train and validation subsets under original split matching: {counts}")
    if failure_count:
        raise PreflightError(f"{failure_count} missing/invalid segment features (first 20):\n  " + "\n  ".join(failures))
    return dict(status="passed", data_root=str(root), split_counts=counts,
                max_quantized_duration=max_duration, feature_files_checked=checked_files,
                feature_content_finiteness="all train/validation values scanned; finite",
                shapes=dict(clap_events=[100, 512], clip_events=[100, 768], clap_segments=["L>=1", 512],
                            clip_segments=["L>=1", 768]), category_order=UNAV_CATEGORIES,
                fixed_inputs={key: dict(path=str(paths[key]), sha256=sha256(paths[key]))
                              for key in ("annotations", "clap_events", "clip_events")},
                test_features_read=False, unused_feature_reads_omitted=["VGGish", "I3D RGB", "I3D flow"])


def make_optimizer_scheduler(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(args.beta1, args.beta2), eps=args.eps)
    scheduler = WarmUpCosineAnnealingLR(optimizer, args.warm_up_epoch, args.epochs, args.lr_min, args.lr)
    return optimizer, scheduler


def train_epoch(args, model, loader, optimizer, device, check_budget=lambda: None):
    model.train()
    meter = AverageMeter()
    for batch in loader:
        check_budget()
        clip, clap = batch["clip_feat"].to(device), batch["clap_feat"].to(device)
        valid, attention = batch["valid_mask"].to(device), batch["attn_mask"].to(device)
        labels = batch["gt_label"].float().to(device)
        optimizer.zero_grad()
        outputs = model(clap, clip, valid_mask=valid, attn_mask=attention)
        loss, values = model.calculate_loss(args, outputs, labels, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite teacher training loss; run stopped without changing inputs")
        meter.update(values, clip.size(0))
        loss.backward()
        if args.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
    return dict(meter.average())


@torch.no_grad()
def evaluate_validation(args, model, loader, device, check_budget=lambda: None):
    if getattr(loader.dataset, "mode", None) != "validation":
        raise ValueError("Teacher checkpoint selection accepts only the UnAV validation split")
    model.eval()
    meter, evaluator = AverageMeter(), AVVPEvaluator()
    for batch in loader:
        check_budget()
        clip, clap = batch["clip_feat"].to(device), batch["clap_feat"].to(device)
        valid, attention = batch["valid_mask"].to(device), batch["attn_mask"].to(device)
        labels = batch["gt_label"].float().to(device)
        outputs = model(clap, clip, valid_mask=valid, attn_mask=attention)
        loss, values = model.calculate_loss(args, outputs, labels, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite teacher validation loss")
        meter.update(values, clip.size(0))
        audio, visual = (value.squeeze(0) for value in outputs)
        prediction = ((audio.sigmoid() * visual.sigmoid()) >= 0.5).to(torch.int32)
        duration = int(batch["duration"][0].item())
        prediction = prediction.permute(1, 0).cpu().numpy()[:, :duration]
        target = batch["gt_label"].float().squeeze(0).permute(1, 0).numpy()[:, :duration]
        empty_pred, empty_gt = np.zeros_like(prediction), np.zeros_like(target)
        evaluator.update("segment", *segment_level(empty_pred, empty_pred, prediction, empty_gt, empty_gt, target))
        evaluator.update("event", *event_level(empty_pred, empty_pred, prediction, empty_gt, empty_gt, target))
    return evaluator.output_result(), dict(meter.average())


@contextmanager
def bounded_runtime(seconds):
    deadline = time.monotonic() + seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Teacher run exceeded {seconds:g} seconds")
    def alarm(signum, frame):
        raise TimeoutError(f"Teacher run exceeded {seconds:g} seconds")
    previous = signal.signal(signal.SIGALRM, alarm) if hasattr(signal, "setitimer") else None
    if previous is not None:
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield check
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def run_training(args, input_report, check_budget):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    log = logging.getLogger("teacher_train")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "train.log", mode="x")
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler())
    state = dict(state="running", config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 preflight=input_report, selection="UnAV validation Seg-av strictly greater than prior best (initial 0)",
                 test_used=False, wandb_used=False, best_epoch=0, best_seg_av=0.0,
                 runtime=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]))
    state_path = output / "run.json"
    def save_state():
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_state()
    try:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(args.device)
        clap_events = torch.from_numpy(np.load(args.data_root / INPUT_LAYOUT["clap_events"], allow_pickle=False)).to(device)
        clip_events = torch.from_numpy(np.load(args.data_root / INPUT_LAYOUT["clip_events"], allow_pickle=False)).to(device)
        model = build_model(args, args.model, clip_events, clap_events).to(device)
        train_data = FeatureOnlyUnAVDataset("train", args.data_root)
        validation_data = FeatureOnlyUnAVDataset("val", args.data_root)
        train_loader = build_dataloader(args, train_data, "train")
        validation_loader = build_dataloader(args, validation_data, "val")
        optimizer, scheduler = make_optimizer_scheduler(args, model)
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
            for epoch in range(1, args.epochs + 1):
                check_budget()
                learning_rate = optimizer.param_groups[0]["lr"]
                train_loss = train_epoch(args, model, train_loader, optimizer, device, check_budget)
                scheduler.step()  # Original order: after training, before validation.
                scores, val_loss = evaluate_validation(args, model, validation_loader, device, check_budget)
                if epoch % args.save_interval == 0:
                    torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
                if scores["Seg-av"] > state["best_seg_av"]:
                    state.update(best_seg_av=scores["Seg-av"], best_epoch=epoch, best_metrics=scores)
                    torch.save(model.state_dict(), checkpoint_dir / "checkpoint_best.pt")
                row = dict(epoch=epoch, lr=learning_rate, train_loss=train_loss, validation_loss=val_loss,
                           validation=scores, best_epoch=state["best_epoch"])
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                state["last_epoch"] = epoch
                save_state()
                log.info("Epoch %d/%d lr=%.8g train=%.6f val=%.6f Seg-av=%.6f", epoch, args.epochs,
                         learning_rate, train_loss["loss_all"], val_loss["loss_all"], scores["Seg-av"])
        state["state"] = "completed"
        best = checkpoint_dir / "checkpoint_best.pt"
        state["selected_checkpoint"] = str(best) if best.exists() else None
        if best.exists():
            state["checkpoint_sha256"] = sha256(best)
        else:
            log.warning("No positive validation Seg-av: original strict selection produced no best checkpoint")
    except BaseException as error:
        state.update(state="failed", error_type=type(error).__name__, error=str(error))
        log.exception("Teacher run stopped")
        raise
    finally:
        save_state()
        for current in list(log.handlers):
            current.close()
            log.removeHandler(current)
    return state


def main(argv=None):
    try:
        args = prepare_args(parser().parse_args(argv))
        torch.set_num_threads(args.threads)
        with bounded_runtime(args.timeout_seconds) as check_budget:
            report = preflight(args.data_root, check_budget)
            if args.check_only:
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0
            if args.device == "cuda" and not torch.cuda.is_available():
                raise PreflightError("--device cuda requested but CUDA is unavailable; check-only does not require a GPU")
            run_training(args, report, check_budget)
        return 0
    except (PreflightError, OSError, ValueError, TimeoutError, RuntimeError, FloatingPointError) as error:
        print(f"Teacher training stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
