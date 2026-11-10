#!/usr/bin/env bash
# Budgeted, immutable-output orchestration. Each stage keeps its own run.json/log.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ID=${RUN_ID:-full_$(date -u +%Y%m%dT%H%M%SZ)}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-6}
TEACHER_HOURS=${TEACHER_HOURS:-24}
STAGE_HOURS=${STAGE_HOURS:-4}
# Fail before starting if ANY planned stage would overwrite evidence.
for stage in teacher calibration labels main test; do
  for parent in artifacts models logger; do
    [[ ! -e "$parent/${RUN_ID}_${stage}" && ! -L "$parent/${RUN_ID}_${stage}" ]] || { echo "Existing run: $parent/${RUN_ID}_${stage}" >&2; exit 1; }
  done
done
"$PYTHON_BIN" reproduce.py train-teacher --run-id "${RUN_ID}_teacher" --data-root data/UnAV --device "$DEVICE" --timeout-hours "$TEACHER_HOURS"
TEACHER="artifacts/${RUN_ID}_teacher/teacher/checkpoints/checkpoint_best.pt"
[[ -f "$TEACHER" ]] || { echo 'No positive validation score selected a teacher checkpoint.' >&2; exit 1; }
"$PYTHON_BIN" reproduce.py calibrate-teacher --run-id "${RUN_ID}_calibration" --checkpoint "$TEACHER" --device "$DEVICE" --timeout-hours "$STAGE_HOURS"
"$PYTHON_BIN" reproduce.py generate-labels --run-id "${RUN_ID}_labels" --checkpoint "$TEACHER" --threshold-dir "artifacts/${RUN_ID}_calibration/thresholds" --device "$DEVICE" --timeout-hours "$STAGE_HOURS"
"$PYTHON_BIN" reproduce.py train-llp --run-id "${RUN_ID}_main" --label-root "artifacts/${RUN_ID}_labels/labels" --seed "$SEED" --device "$DEVICE" --timeout-hours "$STAGE_HOURS"
"$PYTHON_BIN" reproduce.py evaluate-llp --run-id "${RUN_ID}_test" --label-root "artifacts/${RUN_ID}_labels/labels" --checkpoint "models/${RUN_ID}_main/${RUN_ID}_main_r2.best" --timeout-hours "$STAGE_HOURS"
