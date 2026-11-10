"""Preflight validation for the paper's LLP training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .paths import PROJECT_ROOT


@dataclass
class ArtifactCheck:
    name: str
    path: str
    expected_files: int
    present_files: int
    missing_files: int
    sample_shape: Tuple[int, ...] | None
    expected_shape: Tuple[int, ...]
    ok: bool
    invalid_files: int = 0
    errors: list = None


@dataclass
class PipelineReport:
    videos: int
    split_sizes: Dict[str, int]
    artifacts: List[ArtifactCheck]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.artifacts)


def _video_names(csv_path: Path) -> List[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [row["filename"][:11] for row in rows]


def _check_artifact(
    name: str,
    directory: Path,
    video_names: Sequence[str],
    expected_shape: Tuple[int, ...],
) -> ArtifactCheck:
    expected = {f"{video_name}.npy" for video_name in video_names}
    present = {item.name for item in directory.glob("*.npy")} if directory.is_dir() else set()
    missing = expected - present
    sample_shape = None
    invalid, errors = 0, []
    for filename in sorted(expected - missing):
        try:
            array = np.load(directory / filename, allow_pickle=False)
            if sample_shape is None:
                sample_shape = tuple(array.shape)
            if array.shape != expected_shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"invalid shape/dtype/values: {array.shape}/{array.dtype}")
            if "pseudo-label" in name and ((array < 0).any() or (array > 1).any()):
                raise ValueError("pseudo-label outside [0,1]")
        except (ValueError, OSError, EOFError) as exc:
            invalid += 1
            if len(errors) < 10:
                errors.append(f"{filename}: {exc}")
    return ArtifactCheck(name, str(directory), len(expected), len(present & expected),
                         len(missing), sample_shape, expected_shape,
                         not missing and invalid == 0, invalid, errors)


def validate_pipeline(root: Path = PROJECT_ROOT, label_root: Path | None = None) -> PipelineReport:
    split_paths = {
        "train": root / "data/AVVP_train.csv",
        "val": root / "data/AVVP_val_pd.csv",
        "test": root / "data/AVVP_test_pd.csv",
    }
    split_names = {name: _video_names(path) for name, path in split_paths.items()}
    names = [name for split in split_names.values() for name in split]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate 11-character feature key within/across LLP splits")
    specifications = [
        ("CLAP audio", root / "data/CLAP/features", (64, 768)),
        ("CLIP visual", root / "data/CLIP/features", (10, 768)),
        ("R2Plus1D", root / "pretrain/data/LLP/feats/r2plus1d_18", (10, 512)),
        ("audio pseudo-label", label_root / 'audio' if label_root else root / "data/CLAP/a_segmentwise_pseudo_labels", (10, 25)),
        ("visual pseudo-label", label_root / 'visual' if label_root else root / "data/CLIP/v_segmentwise_pseudo_labels", (10, 25)),
    ]
    return PipelineReport(
        videos=len(names),
        split_sizes={name: len(values) for name, values in split_names.items()},
        artifacts=[
            _check_artifact(name, path, names, shape)
            for name, path, shape in specifications
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument('--label-root', type=Path, help='custom generated labels, audio/ and visual/')
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()
    label_root = args.label_root
    if label_root is not None:
        label_root = (args.root / label_root).resolve() if not label_root.is_absolute() else label_root.resolve()
    report = validate_pipeline(args.root.resolve(), label_root)
    payload = {**asdict(report), "ok": report.ok}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"LLP videos: {report.videos} {report.split_sizes}")
        for item in report.artifacts:
            status = "OK" if item.ok else "FAIL"
            print(
                f"[{status}] {item.name}: {item.present_files}/{item.expected_files}, "
                f"missing={item.missing_files}, shape={item.sample_shape}"
            )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
