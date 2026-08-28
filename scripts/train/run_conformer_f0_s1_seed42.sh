#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/train/run_conformer_f0_s1_seed42.sh [smoke|formal]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DEEPSHIP_DATA_ROOT:-/home/slwang/deepship/datasets/DeepShip}"
RUN_ROOT="/home/slwang/deepship/runs/conformer_sampling_v1"
LOG_ROOT="/home/slwang/deepship/logs"
mkdir -p "$LOG_ROOT"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --split-manifest protocols/isolation_comparison_v1/vessel_name_disjoint/split_manifest.json
  --protocol-name vessel_name_disjoint
  --pretrained-revision 1afaab48b41d924fbbcae05d8c5d88836c4a5719
  --clip-duration 20
  --finetuning-mode frozen
  --training-sampling recording_balanced_dynamic
  --train-samples-per-epoch 14000
  --batch-size 1
  --eval-batch-size 2
  --gradient-accumulation-steps 8
  --precision bf16
  --disable-gradient-checkpointing
  --encoder-learning-rate 5e-6
  --head-learning-rate 1e-4
  --min-learning-rate 1e-6
  --weight-decay 1e-2
  --warmup-ratio 0.05
  --warmup-start-factor 0.1
  --early-stopping-patience 3
  --early-stopping-min-delta 0.005
  --seed 42
  --prefetch-factor 2
  --log-interval 100
)

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT="$RUN_ROOT/smoke_f0_s1_recording_dynamic_20s_seed42"
  LOG_PATH="$LOG_ROOT/smoke_sampling_v1_f0_s1_recording_dynamic_20s_seed42.log"
  MODE_ARGS=(
    --epochs 1
    --max-train-batches 2
    --max-eval-batches 2
    --num-workers 0
  )
else
  OUTPUT_ROOT="$RUN_ROOT/formal_f0_s1_recording_dynamic_20s_seed42"
  LOG_PATH="$LOG_ROOT/formal_sampling_v1_f0_s1_recording_dynamic_20s_seed42.log"
  MODE_ARGS=(
    --epochs 8
    --num-workers 4
  )
fi

echo "mode=$MODE"
echo "output_root=$OUTPUT_ROOT"
echo "log_path=$LOG_PATH"
echo "test_evaluation=disabled"

PYTHONUNBUFFERED=1 python scripts/train/train_deepship_conformer.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee "$LOG_PATH"
