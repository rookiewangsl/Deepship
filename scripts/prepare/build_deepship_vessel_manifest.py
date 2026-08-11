from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship_metadata import clean_deepship_metadata  # noqa: E402
from src.utils.pathing import default_deepship_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a cleaned DeepShip recording-to-vessel manifest.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-dir", default="outputs/deepship_metadata")
    parser.add_argument(
        "--aliases",
        default=str(ROOT / "configs" / "deepship_vessel_aliases.csv"),
        help="CSV with raw_name and canonical_name columns.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = clean_deepship_metadata(
        args.data_root,
        args.output_dir,
        alias_path=args.aliases,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
