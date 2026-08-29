#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

for split_seed in 42 43 44; do
  for model_seed in 42 43 44; do
    if [[ "$split_seed" == "42" && "$model_seed" == "42" ]]; then
      continue
    fi
    for variant in g0 g0_c g1; do
      DEEPSHIP_PYTHON="${DEEPSHIP_PYTHON:-/home/slwang/.venvs/deepship/bin/python}" \
        bash scripts/train/run_macnna_global_l20_repeat.sh \
        "$variant" "$split_seed" "$model_seed" formal
    done
  done
done
