#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
MODE="${2:-smoke}"
if [[ "$VARIANT" != "g0" && "$VARIANT" != "g0_c" && "$VARIANT" != "g1" ]]; then
  echo "Usage: bash scripts/train/run_macnna_global_l20_seed42.sh {g0|g0_c|g1} [smoke|formal]" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/train/run_macnna_global_l20_seed42.sh {g0|g0_c|g1} [smoke|formal]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DEEPSHIP_DATA_ROOT:-/home/slwang/deepship/datasets/DeepShip}"
PYTHON_BIN="${DEEPSHIP_PYTHON:-python}"
RUN_ROOT="/home/slwang/deepship/runs/macnna_global_l20_v1"
LOG_ROOT="/home/slwang/deepship/logs"
mkdir -p "$LOG_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json
  --experiment-config configs/experiments/isolation_comparison_v1.json
  --g-series-config configs/experiments/macnna_global_l20_v1.json
  --protocol-name vessel_name_disjoint
  --model-variant "$VARIANT"
  --training-sampling vessel_balanced_dynamic
  --train-samples-per-epoch 14000
  --clip-duration 20
  --n-mels 64
  --n-fft 1024
  --win-length 1024
  --hop-length 512
  --optimizer adamw
  --learning-rate 3e-4
  --weight-decay 1e-2
  --gradient-accumulation-steps 4
  --max-grad-norm 1.0
  --min-learning-rate 1e-6
  --warmup-epochs 5
  --early-stopping-patience 8
  --early-stopping-min-delta 0.005
  --precision bf16
  --seed 42
  --prefetch-factor 2
  --log-interval 100
)

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT="$RUN_ROOT/smoke_${VARIANT}_l20_seed42"
  LOG_PATH="$LOG_ROOT/smoke_macnna_global_l20_v1_${VARIANT}_seed42.log"
  MODE_ARGS=(
    --epochs 1
    # Keep the formal physical batch size so the smoke is also a CUDA-memory gate.
    --batch-size 4
    --eval-batch-size 4
    --gradient-accumulation-steps 1
    --num-workers 0
    --max-train-batches 2
    --max-eval-batches 2
    --allow-experiment-overrides
  )
else
  OUTPUT_ROOT="$RUN_ROOT/formal_${VARIANT}_l20_seed42"
  LOG_PATH="$LOG_ROOT/formal_macnna_global_l20_v1_${VARIANT}_seed42.log"
  MODE_ARGS=(
    --epochs 50
    --batch-size 4
    --eval-batch-size 4
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
