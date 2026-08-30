#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "stats" && "$MODE" != "overfit" && "$MODE" != "g0" ]]; then
  echo "Usage: bash scripts/train/run_belgian_training_sanity.sh {stats|overfit|g0}" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${DEEPSHIP_PYTHON:-/home/slwang/.venvs/deepship/bin/python}"
DATA_ROOT="${BELGIAN_DATA_ROOT:-/home/slwang/deepship/datasets/belgian_ais/extracted/data_per_station_6_paper-window-6_10seconds-efficient_paper_split}"
MANIFEST="protocols/belgian_attention_v1/fold1/split_manifest.json"
CONFIG="configs/experiments/belgian_training_sanity_v1.json"
ANALYSIS_ROOT="/home/slwang/deepship/analysis/belgian_training_sanity_v1"
RUN_ROOT="/home/slwang/deepship/runs/belgian_training_sanity_v1"
LOG_ROOT="/home/slwang/deepship/logs"
STATS_PATH="$ANALYSIS_ROOT/fold1_train_logmel_stats.json"
OVERFIT_ROOT="$ANALYSIS_ROOT/overfit_g0_fold1_seed42"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$DATA_ROOT" "$MANIFEST" "$CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "Required Belgian sanity input is missing: $required" >&2
    exit 1
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Belgian sanity runner requires a clean fixed commit" >&2
  exit 1
fi
mkdir -p "$ANALYSIS_ROOT" "$RUN_ROOT" "$LOG_ROOT"

if [[ "$MODE" == "stats" ]]; then
  LOG_PATH="$LOG_ROOT/belgian_training_sanity_v1_stats.log"
  if [[ -e "$STATS_PATH" || -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite Belgian normalization output or log" >&2
    exit 1
  fi
  nice -n 10 "$PYTHON_BIN" scripts/prepare/compute_belgian_train_normalization.py \
    --data-root "$DATA_ROOT" \
    --split-manifest "$MANIFEST" \
    --output "$STATS_PATH" \
    --batch-size 16 \
    --num-workers 16 \
    2>&1 | tee "$LOG_PATH"
elif [[ "$MODE" == "overfit" ]]; then
  LOG_PATH="$LOG_ROOT/belgian_training_sanity_v1_overfit_g0_fold1_seed42.log"
  if [[ ! -f "$STATS_PATH" ]]; then
    echo "Train-fold normalization must complete before the overfit check" >&2
    exit 1
  fi
  if [[ -e "$OVERFIT_ROOT" || -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite Belgian overfit output or log" >&2
    exit 1
  fi
  "$PYTHON_BIN" scripts/train/check_belgian_g0_overfit.py \
    --data-root "$DATA_ROOT" \
    --split-manifest "$MANIFEST" \
    --normalization-stats "$STATS_PATH" \
    --experiment-config "$CONFIG" \
    --output-root "$OVERFIT_ROOT" \
    --seed 42 \
    --batch-size 32 \
    --num-workers 8 \
    2>&1 | tee "$LOG_PATH"
else
  OUTPUT_ROOT="$RUN_ROOT/formal_g0_fold1_seed42_full"
  LOG_PATH="$LOG_ROOT/formal_belgian_training_sanity_v1_g0_fold1_seed42_full.log"
  if [[ ! -f "$STATS_PATH" || ! -f "$OVERFIT_ROOT/overfit_report.json" ]]; then
    echo "Normalization and overfit outputs must exist before full G0 training" >&2
    exit 1
  fi
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="passed" and p["test_evaluated"] is False' \
    "$OVERFIT_ROOT/overfit_report.json"
  if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite Belgian full G0 output or log" >&2
    exit 1
  fi
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train/train_belgian_macnna_global.py \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --split-manifest "$MANIFEST" \
    --experiment-config "$CONFIG" \
    --model-variant g0 \
    --seed 42 \
    --sampling-strategy full_epoch_shuffle \
    --loss-strategy effective_number \
    --effective-number-beta 0.999 \
    --normalization-stats-path "$STATS_PATH" \
    --batch-size 16 \
    --eval-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --epochs 30 \
    --learning-rate 0.0003 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --min-learning-rate 0.000001 \
    --warmup-epochs 1 \
    --early-stopping-patience 8 \
    --early-stopping-min-delta 0.002 \
    --early-stopping-start-epoch 5 \
    --precision bf16 \
    --num-workers 8 \
    2>&1 | tee "$LOG_PATH"
fi
