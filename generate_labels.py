#!/usr/bin/env python3
"""Regenerate LLP teacher labels into a NEW directory using frozen thresholds.

Uses the released five-layer checkpoint and existing (not event_feats_v2) text
embeddings. Does not recalibrate thresholds, normalize projected features,
load dense ground truth, or require the unused VGGish/Res152 teacher inputs.
New teacher checkpoints require their own validation-calibrated thresholds.
AVE regeneration is deliberately not supported (teacher audio inputs missing).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pretrain.categories import AVVP_CATEGORIES

ROOT = Path(__file__).resolve().parent
SPLITS = {"train": "AVVP_train.csv", "val": "AVVP_val_pd.csv", "test": "AVVP_test_pd.csv"}


def resolve_output(output: Path, root: Path = ROOT) -> Path:
    root = root.resolve()
    output = output.expanduser()
    output = root / output if not output.is_absolute() else output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output = output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must stay inside this package; symlink escapes are refused")
    return output


def select_records(split: str, limit: int | None, root: Path = ROOT) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    result, seen = [], set()
    for name in SPLITS if split == "all" else (split,):
        with (root / "data" / SPLITS[name]).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                key = row["filename"][:11]
                if key in seen:
                    raise ValueError(f"duplicate feature key: {key}")
                seen.add(key)
                # The original loader uses the last TSV column for weak labels.
                labels = row[reader.fieldnames[-1]].split(",")
                unknown = set(labels) - set(AVVP_CATEGORIES)
                if unknown:
                    raise ValueError(f"unknown weak labels in {key}: {unknown}")
                result.append(dict(split=name, video=key, filename=row['filename'], weak_labels=labels))
                if limit is not None and len(result) >= limit:
                    return result
    return result


def threshold_labels(logits: np.ndarray, thresholds: np.ndarray, weak: np.ndarray) -> np.ndarray:
    if logits.shape != (10, 25) or thresholds.shape != (25,) or weak.shape != (25,):
        raise ValueError("expected logits (10,25), thresholds (25,), weak labels (25,)")
    if not np.isfinite(logits).all() or np.isnan(thresholds).any():
        raise ValueError("nonfinite logits or NaN thresholds")
    masked = np.where(weak.astype(bool)[None, :], logits, -np.inf)
    return (masked > thresholds[None, :]).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_comparison(labels: np.ndarray, path: Path) -> dict:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    frozen = np.load(path, allow_pickle=False)
    same_shape = frozen.shape == labels.shape
    return dict(available=True, exact_equal=bool(np.array_equal(frozen, labels)),
                shape_equal=same_shape,
                differing_positions=int(np.count_nonzero(frozen != labels)) if same_shape else None)


def input_path(path: Path, root: Path = ROOT) -> Path:
    path = Path(path).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def teacher_paths(root: Path, checkpoint=None, thresholds=None) -> dict:
    return {
        'checkpoint': input_path(checkpoint or 'pretrain/checkpoints/teacher_unav100_best.pt', root),
        'audio_events': root / 'pretrain/data/LLP/feats_CLAP/event_feats/all_event_feats.npy',
        'visual_events': root / 'pretrain/data/LLP/feats_CLIP/event_feats/all_event_feats.npy',
        'audio_thresholds': input_path(thresholds or 'pretrain/thresholds/llp', root) / 'a_classwise_threhsolds.npy',
        'visual_thresholds': input_path(thresholds or 'pretrain/thresholds/llp', root) / 'v_classwise_threhsolds.npy',
    }


def materialize_state_dict(model, state):
    """Bind CPU checkpoint tensors to a meta-initialized model without a second copy.

    Explicitly enforce the same key/shape/dtype contract as the released model.
    The normal training constructor and initialization are unchanged.
    """
    import torch
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError("teacher checkpoint keys do not match the model")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"teacher checkpoint shape/dtype mismatch: {name}")
        module = model
        parts = name.split('.')
        for part in parts[:-1]:
            module = getattr(module, part)
        key = parts[-1]
        if key in module._parameters:
            module._parameters[key] = torch.nn.Parameter(tensor, requires_grad=module._parameters[key].requires_grad)
        else:
            module._buffers[key] = tensor
    return model


def load_teacher(paths: dict, device):
    import torch
    from pretrain.model.TemporalTransformer import TemporalTransformer
    embeddings = {}
    for modality, dim in (('audio', 512), ('visual', 768)):
        array = np.load(paths[modality + '_events'], allow_pickle=False)
        if array.shape != (25, dim) or not np.isfinite(array).all():
            raise ValueError(f'invalid {modality} text embeddings: {array.shape}')
        embeddings[modality] = torch.from_numpy(array).float().to(device)
    model = TemporalTransformer(input_v_2d_dim=768, input_v_3d_dim=1024, input_a_dim=512,
                                hidden_dim=1024, nhead=16, ff_dim=2048, num_layers=5,
                                dropout=0.3, pre_norm=False,
                                clap_event_feats=embeddings['audio'], clip_event_feats=embeddings['visual'], device='meta')
    materialize_state_dict(model, torch.load(paths['checkpoint'], map_location='cpu'))
    return model.to(device).eval()


def teacher_logits(model, record: dict, root: Path, device):
    import torch
    tensors = []
    for folder, dim in (('feats_CLAP', 512), ('feats_CLIP', 768)):
        path = root / 'pretrain/data/LLP' / folder / 'segment_feats' / (record['video'] + '.npy')
        value = np.load(path, allow_pickle=False)
        if value.shape != (10, dim) or not np.isfinite(value).all():
            raise ValueError(f'invalid teacher feature: {path}: {value.shape}')
        tensors.append(torch.from_numpy(value).float().unsqueeze(0).to(device))
    return model(*tensors, valid_mask=torch.ones((1, 10), device=device),
                 attn_mask=torch.ones((1, 10, 10), dtype=torch.bool, device=device))


def validate_threshold_source(paths: dict, root: Path, explicit_thresholds) -> None:
    frozen = root / 'pretrain/checkpoints/teacher_unav100_best.pt'
    if paths['checkpoint'] == frozen.resolve() and explicit_thresholds is None:
        record = json.loads((root / 'assets/frozen_teacher.json').read_text())
        for key, path in paths.items():
            if record.get('sha256', {}).get(key) != sha256(path):
                raise ValueError(f'frozen teacher input changed: {key}; recalibrate this teacher')
        return
    if explicit_thresholds is None:
        raise ValueError('a new teacher requires --threshold-dir from calibrate-teacher, not frozen thresholds')
    manifest = paths['audio_thresholds'].parent / 'manifest.json'
    if not manifest.is_file():
        raise ValueError('new teacher thresholds must include their calibration manifest.json')
    record = json.loads(manifest.read_text())
    if (record.get('state') != 'complete' or record.get('split') != 'val'
            or record.get('checkpoint_sha256') != sha256(paths['checkpoint'])):
        raise ValueError('threshold calibration is incomplete or belongs to a different teacher')
    if record.get('schema_version', 1) >= 2:
        for key in ('audio_events', 'visual_events'):
            if record.get('event_embeddings_sha256', {}).get(key) != sha256(paths[key]):
                raise ValueError('class embedding changed since calibration')
        if record.get('validation_split_sha256') != sha256(root / 'data/AVVP_val_pd.csv'):
            raise ValueError('validation split changed since calibration')
    for name in ('audio_thresholds', 'visual_thresholds'):
        if record.get('threshold_sha256', {}).get(name) != sha256(paths[name]):
            raise ValueError('threshold file has changed since calibration')


def generate(args: argparse.Namespace, root: Path = ROOT) -> dict:
    root = root.resolve()
    output = resolve_output(args.output, root)
    if args.threads < 1 or args.timeout_seconds <= 0:
        raise ValueError("threads and timeout-seconds must be positive")
    records = select_records(args.split, args.limit, root)
    paths = teacher_paths(root, getattr(args, 'checkpoint', None), getattr(args, 'threshold_dir', None))
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_threshold_source(paths, root, getattr(args, 'threshold_dir', None))
    features = {"audio": root / "pretrain/data/LLP/feats_CLAP/segment_feats",
                "visual": root / "pretrain/data/LLP/feats_CLIP/segment_feats"}
    for record in records:
        for folder in features.values():
            if not (folder / (record["video"] + ".npy")).is_file():
                raise FileNotFoundError(folder / (record["video"] + ".npy"))

    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items() if name != "checkpoint"}
    for key, shape in {"audio_events": (25, 512), "visual_events": (25, 768),
                       "audio_thresholds": (25,), "visual_thresholds": (25,)}.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {arrays[key].shape}")
    started = time.monotonic()
    model = load_teacher(paths, device)
    output.mkdir(parents=True, exist_ok=False)
    for modality in features:
        (output / modality).mkdir()
    report = dict(state="running", split=args.split, limit=args.limit, device=args.device,
                  samples_requested=len(records), samples=[], output=str(output),
                  fixed_inputs={key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
                  protocol="five-layer UnAV teacher; unchanged event_feats; fixed logit thresholds; weak masking; no feature normalization",
                  dense_ground_truth_used=False, threshold_recalibrated=False,
                  comparison_scope="only generated samples, not a full-dataset equivalence claim")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        with torch.no_grad():
            for record in records:
                if time.monotonic() - started >= args.timeout_seconds:
                    raise TimeoutError("generation time budget exhausted")
                logits = teacher_logits(model, record, root, device)
                weak = np.asarray([name in record["weak_labels"] for name in AVVP_CATEGORIES])
                item = dict(record, comparison={})
                for modality, value, frozen_dir in zip(("audio", "visual"), logits,
                        ("data/CLAP/a_segmentwise_pseudo_labels", "data/CLIP/v_segmentwise_pseudo_labels")):
                    labels = threshold_labels(value[0].cpu().numpy(), arrays[modality + "_thresholds"], weak)
                    np.save(output / modality / (record["video"] + ".npy"), labels)
                    item["comparison"][modality] = frozen_comparison(labels, root / frozen_dir / (record["video"] + ".npy"))
                report["samples"].append(item)
                if len(report["samples"]) % 100 == 0:
                    print(f"Generated {len(report['samples'])}/{len(records)}", flush=True)
        comparisons = [value for item in report["samples"] for value in item["comparison"].values()]
        report.update(state="complete", samples_generated=len(report["samples"]),
                      all_frozen_available=all(value["available"] for value in comparisons),
                      all_generated_equal_frozen=all(value.get("exact_equal", False) for value in comparisons))
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("state", "samples_generated", "all_frozen_available",
                                                "all_generated_equal_frozen", "elapsed_seconds", "output")}, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NEW directory inside this package; relative paths use package root")
    parser.add_argument('--checkpoint', type=Path, help='teacher state_dict; default is bundled frozen teacher')
    parser.add_argument('--threshold-dir', type=Path, help='matching calibrate-teacher output for a new teacher')
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--limit", type=int, help="first N videos total, in train/val/test order for all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args(argv)
    try:
        generate(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"generate_labels: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
