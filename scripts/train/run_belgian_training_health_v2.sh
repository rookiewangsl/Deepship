#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${DEEPSHIP_PYTHON:-/home/slwang/.venvs/deepship/bin/python}"
DATA_ROOT="${BELGIAN_DATA_ROOT:-/home/slwang/deepship/datasets/belgian_ais/extracted/data_per_station_6_paper-window-6_10seconds-efficient_paper_split}"
MANIFEST="protocols/belgian_attention_v1/fold1/split_manifest.json"
CONFIG="configs/experiments/belgian_training_health_v2.json"
STATS_PATH="/home/slwang/deepship/analysis/belgian_training_sanity_v1/fold1_train_logmel_stats.json"
OVERFIT_REPORT="/home/slwang/deepship/analysis/belgian_training_sanity_v1/overfit_g0_fold1_seed42/overfit_report.json"
OUTPUT_ROOT="/home/slwang/deepship/runs/belgian_training_health_v2/formal_g0_fold1_seed42_balanced_batch"
LOG_PATH="/home/slwang/deepship/logs/formal_belgian_training_health_v2_g0_fold1_seed42_balanced_batch.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$DATA_ROOT" "$MANIFEST" "$CONFIG" "$STATS_PATH" "$OVERFIT_REPORT"; do
  if [[ ! -e "$required" ]]; then
    echo "Required Belgian health input is missing: $required" >&2
    exit 1
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Belgian health runner requires a clean fixed commit" >&2
  exit 1
fi
"$PYTHON_BIN" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="passed" and p["test_evaluated"] is False' \
  "$OVERFIT_REPORT"
if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite Belgian balanced-batch G0 output or log" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_ROOT")" "$(dirname "$LOG_PATH")"

echo "experiment=belgian_training_health_v2 variant=g0 fold=1 seed=42"
echo "sampling=strict_class_balanced_batch samples_per_class_per_epoch=1024"
echo "test_evaluation=disabled"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train/train_belgian_macnna_global.py \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split-manifest "$MANIFEST" \
  --experiment-config "$CONFIG" \
  --model-variant g0 \
  --seed 42 \
  --sampling-strategy strict_class_balanced_batch \
  --samples-per-class-per-epoch 1024 \
  --loss-strategy cross_entropy \
  --normalization-stats-path "$STATS_PATH" \
  --specaugment-frequency-mask-param 8 \
  --specaugment-time-mask-param 24 \
  --specaugment-frequency-masks 1 \
  --specaugment-time-masks 1 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --learning-rate 0.0003 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --min-learning-rate 0.000001 \
  --warmup-epochs 1 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.002 \
  --early-stopping-start-epoch 3 \
  --precision bf16 \
  --num-workers 8 \
  2>&1 | tee "$LOG_PATH"
