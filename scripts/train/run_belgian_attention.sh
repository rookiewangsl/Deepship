#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
FOLD="${2:-}"
SEED="${3:-}"
MODE="${4:-smoke}"

if [[ "$VARIANT" != "g0" && "$VARIANT" != "g1" ]]; then
  echo "Usage: bash scripts/train/run_belgian_attention.sh {g0|g1} {1|2|3} {42|43|44} [smoke|formal]" >&2
  exit 2
fi
if [[ "$FOLD" != "1" && "$FOLD" != "2" && "$FOLD" != "3" ]]; then
  echo "Unsupported fold: $FOLD" >&2
  exit 2
fi
if [[ "$SEED" != "42" && "$SEED" != "43" && "$SEED" != "44" ]]; then
  echo "Unsupported seed: $SEED" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${DEEPSHIP_PYTHON:-/home/slwang/.venvs/deepship/bin/python}"
DATA_ROOT="${BELGIAN_DATA_ROOT:-/home/slwang/deepship/datasets/belgian_ais/extracted/data_per_station_6_paper-window-6_10seconds-efficient_paper_split}"
MANIFEST="protocols/belgian_attention_v1/fold${FOLD}/split_manifest.json"
RUN_ROOT="/home/slwang/deepship/runs/belgian_attention_v1"
LOG_ROOT="/home/slwang/deepship/logs"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$DATA_ROOT" "$MANIFEST" configs/experiments/belgian_attention_v1.json; do
  if [[ ! -e "$required" ]]; then
    echo "Required Belgian input is missing: $required" >&2
    exit 1
  fi
done

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT="$RUN_ROOT/smoke_fold${FOLD}_${VARIANT}_seed${SEED}"
  LOG_PATH="$LOG_ROOT/smoke_belgian_attention_v1_fold${FOLD}_${VARIANT}_seed${SEED}.log"
  MODE_ARGS=(
    --epochs 1
    --batch-size 4
    --eval-batch-size 4
    --gradient-accumulation-steps 1
    --num-workers 0
    --max-train-batches 2
    --max-eval-batches 2
    --allow-experiment-overrides
  )
else
  OUTPUT_ROOT="$RUN_ROOT/formal_fold${FOLD}_${VARIANT}_seed${SEED}"
  LOG_PATH="$LOG_ROOT/formal_belgian_attention_v1_fold${FOLD}_${VARIANT}_seed${SEED}.log"
  MODE_ARGS=()
fi
if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" ]]; then
  echo "Refusing to reuse Belgian output or log: $OUTPUT_ROOT / $LOG_PATH" >&2
  exit 1
fi
mkdir -p "$LOG_ROOT"

echo "mode=$MODE variant=$VARIANT fold=$FOLD seed=$SEED"
echo "output_root=$OUTPUT_ROOT"
echo "test_evaluation=disabled"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train/train_belgian_macnna_global.py \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split-manifest "$MANIFEST" \
  --model-variant "$VARIANT" \
  --seed "$SEED" \
  "${MODE_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
