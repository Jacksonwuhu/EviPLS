"""Evaluate one frozen checkpoint on the canonical AVVP test split.

This evaluator is intentionally separate from evaluate_validation.py:
test evaluation is an end-point comparison only and must never be used for
candidate selection or hyper-parameter tuning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main_ipr as legacy  # noqa: E402
from dataloader import LLP_dataset, ToTensor  # noqa: E402
from nets.net_audiovisual import NREP  # noqa: E402


METRIC_NAMES = (
    "seg_audio",
    "seg_video",
    "seg_av",
    "seg_type",
    "seg_event",
    "eve_audio",
    "eve_video",
    "eve_av",
    "eve_type",
    "eve_event",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label_test",
        type=Path,
        default=Path("data/AVVP_test_pd.csv"),
    )
    parser.add_argument("--audio_dir", type=Path, default=Path("data/CLAP/features"))
    parser.add_argument("--video_dir", type=Path, default=Path("data/CLIP/features"))
    parser.add_argument(
        "--st_dir",
        type=Path,
        default=Path("pretrain/data/LLP/feats/r2plus1d_18"),
    )
    parser.add_argument(
        "--a_pseudo_data_dir",
        type=Path,
        default=Path("data/CLAP/a_segmentwise_pseudo_labels"),
    )
    parser.add_argument(
        "--v_pseudo_data_dir",
        type=Path,
        default=Path("data/CLIP/v_segmentwise_pseudo_labels"),
    )
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    test_label = args.label_test.expanduser().resolve()
    canonical_test = (REPO_ROOT / "data/AVVP_test_pd.csv").resolve()
    if test_label != canonical_test:
        raise ValueError(
            "Test evaluator refuses non-canonical split: "
            f"{test_label} != {canonical_test}"
        )

    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dataset = LLP_dataset(
        label=str(test_label),
        audio_dir=str(args.audio_dir),
        video_dir=str(args.video_dir),
        st_dir=str(args.st_dir),
        transform=transforms.Compose([ToTensor()]),
        v_pseudo_data_dir=str(args.v_pseudo_data_dir),
        a_pseudo_data_dir=str(args.a_pseudo_data_dir),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    audio_model = NREP(args.num_layers, args.temperature).to(device)
    video_model = copy.deepcopy(audio_model)
    audio_model.load_state_dict(checkpoint["model_audio"])
    video_model.load_state_dict(checkpoint["model_video"])
    eval_args = SimpleNamespace(
        loss_semantics="legacy",
        selected_thre=False,
        model_save_dir="",
        checkpoint="",
    )
    values = legacy.eval(
        eval_args,
        [audio_model, video_model],
        loader,
        str(test_label),
        dual_eval=True,
    )
    record = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "split": "test",
        "label": str(test_label),
        "pseudo_label_lineage": {
            "audio": str(args.a_pseudo_data_dir.expanduser().resolve()),
            "visual": str(args.v_pseudo_data_dir.expanduser().resolve()),
        },
        "metrics": dict(
            zip(METRIC_NAMES, (float(value) for value in values))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
