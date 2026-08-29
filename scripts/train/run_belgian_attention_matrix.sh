#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

for fold in 1 2 3; do
  for variant in g0 g1; do
    for seed in 42 43 44; do
      DEEPSHIP_PYTHON="${DEEPSHIP_PYTHON:-/home/slwang/.venvs/deepship/bin/python}" \
        bash scripts/train/run_belgian_attention.sh "$variant" "$fold" "$seed" formal
    done
  done
done

