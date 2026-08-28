from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.group_prediction_analysis import (  # noqa: E402
    analyze_group_predictions,
    load_group_predictions,
    write_group_prediction_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze group predictions with class-stratified bootstrap intervals.",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key-field", default="vessel_key")
    parser.add_argument("--comparison-predictions")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, class_names, source_sha256 = load_group_predictions(
        args.predictions, group_key_field=args.group_key_field
    )
    comparison_rows = None
    comparison_sha256 = None
    if args.comparison_predictions:
        comparison_rows, comparison_names, comparison_sha256 = load_group_predictions(
            args.comparison_predictions, group_key_field=args.group_key_field
        )
        if comparison_names != class_names:
            raise ValueError("Comparison predictions use different class columns")
    analysis = analyze_group_predictions(
        rows,
        class_names,
        bootstrap_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
        comparison_rows=comparison_rows,
    )
    write_group_prediction_analysis(
        analysis,
        rows,
        class_names,
        args.output_dir,
        source_path=args.predictions,
        source_sha256=source_sha256,
        comparison_path=args.comparison_predictions,
        comparison_sha256=comparison_sha256,
    )
    point = analysis["point_estimate"]
    bootstrap = analysis["bootstrap"]
    assert isinstance(point, dict) and isinstance(bootstrap, dict)
    macro_f1 = bootstrap["macro_f1"]
    assert isinstance(macro_f1, dict)
    print(
        f"groups={analysis['group_count']} macro_f1={float(point['macro_f1']):.4f} "
        f"ci=({float(macro_f1['lower']):.4f}, {float(macro_f1['upper']):.4f})"
    )
    print(f"Analysis written to: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
