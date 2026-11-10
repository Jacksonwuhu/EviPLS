#!/usr/bin/env python3
"""LLP validation-only threshold calibration for a newly trained teacher.

Uses the original classwise max-F1 routine, including its historical tie
handling. No training/test dense annotations enter threshold fitting.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from generate_labels import (ROOT, AVVP_CATEGORIES, resolve_output, select_records,
                             teacher_paths, load_teacher, teacher_logits, sha256)


def dense_labels(path, records, anomalies=None):
    wanted = {r['filename'] for r in records}
    labels = {key: np.zeros((10, 25), dtype=np.float32) for key in wanted}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['filename'] not in wanted:
                continue
            start, end = int(row['onset']), int(row['offset'])
            if not 0 <= start <= end <= 10:
                # The canonical LLP audio CSV has this reversed interval.
                # Both original teacher and parser loaders use the empty [9:0]
                # NumPy slice; preserve that evaluation convention explicitly.
                known = (row['filename'], row['event_labels'], start, end) == ('8DF2PxdT5tE_390_400', 'Cheering', 9, 0)
                if not known:
                    raise ValueError(f'invalid event interval in {path}: {row}')
                import warnings
                warnings.warn('Canonical LLP Cheering annotation [9:0] is empty under the original loader; recorded, not repaired')
                if anomalies is not None:
                    anomalies.append(dict(row, source=path.name, handling='original empty NumPy slice; annotation unchanged'))
            labels[row['filename']][start:end, AVVP_CATEGORIES.index(row['event_labels'])] = 1
    return labels


def calibrate(args):
    import torch
    from pretrain.utils.eval_utils import get_best_f_threshold
    if args.threads < 1 or args.timeout_seconds <= 0:
        raise ValueError('positive threads and time budget required')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    output = resolve_output(args.output)
    paths = teacher_paths(ROOT, args.checkpoint)
    records = select_records('val', None)
    if len(records) != 649:
        raise ValueError('calibration requires the complete canonical 649-video LLP validation split')
    gt_paths = {'audio': ROOT / 'data/AVVP_eval_audio.csv', 'visual': ROOT / 'data/AVVP_eval_visual.csv'}
    annotation_anomalies = []
    ground_truth = {m: dense_labels(p, records, annotation_anomalies) for m, p in gt_paths.items()}
    device = torch.device(args.device)
    started = time.monotonic()
    model = load_teacher(paths, device)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=2, state='running', split='val', videos=len(records),
                  event_embeddings_sha256={k: sha256(paths[k]) for k in ('audio_events', 'visual_events')},
                  checkpoint_sha256=sha256(paths['checkpoint']),
                  checkpoint=str(paths['checkpoint']),
                  validation_split_sha256=sha256(ROOT / 'data/AVVP_val_pd.csv'),
                  annotation_sha256={k: sha256(v) for k, v in gt_paths.items()},
                  routine='original get_best_f_threshold; weak-label masking; legacy tie handling',
                  test_used_for_selection=False, annotation_anomalies=annotation_anomalies)
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    try:
        all_logits = {'audio': [], 'visual': []}
        with torch.no_grad():
            for i, record in enumerate(records):
                if time.monotonic() - started > args.timeout_seconds:
                    raise TimeoutError('calibration time budget exhausted')
                logits = teacher_logits(model, record, ROOT, device)
                weak = torch.tensor([c in record['weak_labels'] for c in AVVP_CATEGORIES], device=device)
                for modality, value in zip(('audio', 'visual'), logits):
                    all_logits[modality].append(torch.where(weak[None, :], value[0].float(), -float('inf')))
                if (i + 1) % 100 == 0:
                    print(f'Calibration: {i + 1}/{len(records)}', flush=True)
        report['threshold_sha256'] = {}
        for modality, prefix in (('audio', 'a'), ('visual', 'v')):
            scores = torch.cat(all_logits[modality], dim=0)
            labels = torch.from_numpy(np.concatenate([ground_truth[modality][r['filename']] for r in records])).to(device)
            thresholds = torch.zeros(25, device=device)
            for c in range(25):
                thresholds[c], _ = get_best_f_threshold(labels[:, c], scores[:, c])
            if torch.isnan(thresholds).any():
                raise ValueError('calibration produced NaN thresholds')
            path = output / f'{prefix}_classwise_threhsolds.npy'
            np.save(path, thresholds.cpu().numpy())
            report['threshold_sha256'][modality + '_thresholds'] = sha256(path)
        report['state'] = 'complete'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    args = parser.parse_args()
    try:
        calibrate(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'calibrate_teacher: {error}\n')


if __name__ == '__main__':
    main()
