#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
SPLIT_SEED="${2:-}"
MODEL_SEED="${3:-}"
MODE="${4:-smoke}"

if [[ "$VARIANT" != "g0" && "$VARIANT" != "g0_c" && "$VARIANT" != "g1" ]]; then
  echo "Usage: bash scripts/train/run_macnna_global_l20_repeat.sh {g0|g0_c|g1} {42|43|44} {42|43|44} [smoke|formal]" >&2
  exit 2
fi
if [[ "$SPLIT_SEED" != "42" && "$SPLIT_SEED" != "43" && "$SPLIT_SEED" != "44" ]]; then
  echo "Unsupported split seed: $SPLIT_SEED" >&2
  exit 2
fi
if [[ "$MODEL_SEED" != "42" && "$MODEL_SEED" != "43" && "$MODEL_SEED" != "44" ]]; then
  echo "Unsupported model seed: $MODEL_SEED" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DEEPSHIP_DATA_ROOT:-/home/slwang/deepship/datasets/DeepShip}"
PYTHON_BIN="${DEEPSHIP_PYTHON:-python}"
RUN_ROOT="/home/slwang/deepship/runs/macnna_global_l20_repeats_v1/split${SPLIT_SEED}"
LOG_ROOT="/home/slwang/deepship/logs"
G_CONFIG="configs/experiments/macnna_global_l20_repeats_v1.json"

if [[ "$SPLIT_SEED" == "42" ]]; then
  SPLIT_MANIFEST="protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json"
  EXPERIMENT_CONFIG="configs/experiments/isolation_comparison_v1.json"
else
  REPEAT_PROTOCOL_ROOT="protocols/macnna_global_l20_repeats_v1/split_seed${SPLIT_SEED}"
  SPLIT_MANIFEST="$REPEAT_PROTOCOL_ROOT/vessel_name_disjoint/split_manifest.json"
  EXPERIMENT_CONFIG="$REPEAT_PROTOCOL_ROOT/experiment_config.json"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$SPLIT_MANIFEST" "$EXPERIMENT_CONFIG" "$G_CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "Required frozen input is missing: $required" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --split-manifest "$SPLIT_MANIFEST"
  --experiment-config "$EXPERIMENT_CONFIG"
  --g-series-config "$G_CONFIG"
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
  --seed "$MODEL_SEED"
  --prefetch-factor 2
  --log-interval 100
)

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT="$RUN_ROOT/smoke_${VARIANT}_l20_split${SPLIT_SEED}_seed${MODEL_SEED}"
  LOG_PATH="$LOG_ROOT/smoke_macnna_global_l20_repeats_v1_${VARIANT}_split${SPLIT_SEED}_seed${MODEL_SEED}.log"
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
  OUTPUT_ROOT="$RUN_ROOT/formal_${VARIANT}_l20_split${SPLIT_SEED}_seed${MODEL_SEED}"
  LOG_PATH="$LOG_ROOT/formal_macnna_global_l20_repeats_v1_${VARIANT}_split${SPLIT_SEED}_seed${MODEL_SEED}.log"
  MODE_ARGS=(
    --epochs 50
    --batch-size 4
    --eval-batch-size 4
    --num-workers 8
  )
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse output path: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite log: $LOG_PATH" >&2
  exit 1
fi
mkdir -p "$LOG_ROOT"

echo "mode=$MODE"
echo "variant=$VARIANT"
echo "split_seed=$SPLIT_SEED"
echo "model_seed=$MODEL_SEED"
echo "output_root=$OUTPUT_ROOT"
echo "log_path=$LOG_PATH"
echo "python_bin=$PYTHON_BIN"
echo "test_evaluation=disabled"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/train/train_deepship_macnna_global.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee "$LOG_PATH"
