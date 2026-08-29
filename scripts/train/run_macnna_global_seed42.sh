#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
MODE="${2:-smoke}"
if [[ "$VARIANT" != "g0" && "$VARIANT" != "g0_c" && "$VARIANT" != "g1" ]]; then
  echo "Usage: bash scripts/train/run_macnna_global_seed42.sh {g0|g0_c|g1} [smoke|formal]" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/train/run_macnna_global_seed42.sh {g0|g0_c|g1} [smoke|formal]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DEEPSHIP_DATA_ROOT:-/home/slwang/deepship/datasets/DeepShip}"
PYTHON_BIN="${DEEPSHIP_PYTHON:-python}"
RUN_ROOT="/home/slwang/deepship/runs/macnna_global_v1"
LOG_ROOT="/home/slwang/deepship/logs"
mkdir -p "$LOG_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json
  --protocol-name vessel_name_disjoint
  --model-variant "$VARIANT"
  --clip-duration 3
  --n-mels 64
  --n-fft 1024
  --win-length 1024
  --hop-length 512
  --learning-rate 1e-2
  --momentum 0.9
  --min-learning-rate 1e-5
  --warmup-epochs 10
  --early-stopping-patience 10
  --early-stopping-min-delta 0.005
  --precision bf16
  --seed 42
  --prefetch-factor 2
  --log-interval 100
)

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT="$RUN_ROOT/smoke_${VARIANT}_3s_seed42"
  LOG_PATH="$LOG_ROOT/smoke_macnna_global_v1_${VARIANT}_3s_seed42.log"
  MODE_ARGS=(
    --epochs 1
    --batch-size 2
    --eval-batch-size 2
    --num-workers 0
    --max-train-batches 2
    --max-eval-batches 2
    --allow-experiment-overrides
  )
else
  OUTPUT_ROOT="$RUN_ROOT/formal_${VARIANT}_3s_seed42"
  LOG_PATH="$LOG_ROOT/formal_macnna_global_v1_${VARIANT}_3s_seed42.log"
  MODE_ARGS=(
    --epochs 100
    --batch-size 16
    --eval-batch-size 16
    --num-workers 8
  )
fi

echo "mode=$MODE"
echo "variant=$VARIANT"
echo "output_root=$OUTPUT_ROOT"
echo "log_path=$LOG_PATH"
echo "python_bin=$PYTHON_BIN"
echo "test_evaluation=disabled"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train/train_deepship_macnna_global.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee "$LOG_PATH"
