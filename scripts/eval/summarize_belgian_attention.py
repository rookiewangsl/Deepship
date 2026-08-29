from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.belgian_repeats import summarize_belgian_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all 18 Belgian G0/G1 cells.")
    parser.add_argument(
        "--run-root",
        default="/home/slwang/deepship/runs/belgian_attention_v1",
    )
    parser.add_argument(
        "--output-root",
        default="/home/slwang/deepship/analysis/belgian_attention_v1",
    )
    parser.add_argument("--resamples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_path = output_root / "development_summary.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Belgian summary: {output_path}")
    summary = summarize_belgian_matrix(
        args.run_root,
        resamples=args.resamples,
        seed=args.seed,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

