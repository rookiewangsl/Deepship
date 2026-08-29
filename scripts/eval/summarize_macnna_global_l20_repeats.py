from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.macnna_global_repeats import (  # noqa: E402
    summarize_global_repeats,
    write_global_repeat_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize all 27 DeepShip L20 repeat runs.")
    parser.add_argument(
        "--existing-root",
        default="/home/slwang/deepship/runs/macnna_global_l20_v1",
    )
    parser.add_argument(
        "--repeat-root",
        default="/home/slwang/deepship/runs/macnna_global_l20_repeats_v1",
    )
    parser.add_argument(
        "--experiment-config",
        default=str(ROOT / "configs" / "experiments" / "macnna_global_l20_repeats_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        default="/home/slwang/deepship/analysis/macnna_global_l20_repeats_v1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_global_repeats(
        args.existing_root,
        args.repeat_root,
        args.experiment_config,
    )
    write_global_repeat_summary(summary, args.output_dir)
    print(json.dumps(summary["decision_gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
