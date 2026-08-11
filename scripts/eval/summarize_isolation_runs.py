from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.isolation_summary import (  # noqa: E402
    summarize_isolation_runs,
    write_isolation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize all nine formal DeepShip isolation runs.",
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument(
        "--output-dir",
        help="Summary destination (default: <runs-root>/summary)",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--protocol-root",
        default=str(ROOT / "protocols" / "isolation_comparison_v1"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else runs_root / "summary"
    )
    summary = summarize_isolation_runs(
        runs_root,
        experiment_config=args.config,
        protocol_root=args.protocol_root,
    )
    write_isolation_summary(summary, output_dir)
    print(f"Validated {summary['run_count']} formal runs from commit {summary['git_commit']}")
    print(f"Summary written to: {output_dir}")


if __name__ == "__main__":
    main()
