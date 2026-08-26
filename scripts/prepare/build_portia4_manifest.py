from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.portia import build_portia4_manifest, load_portia4_records, write_portia4_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic MMSI-disjoint PORTIA-4 external-evaluation manifest."
    )
    parser.add_argument(
        "--annotation-root",
        required=True,
        help="Directory containing train.csv, val.csv and test.csv from Single_Vessel_Classification.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_portia4_records(args.annotation_root)
    manifest, summary = build_portia4_manifest(
        records, seed=args.seed, development_fraction=args.development_fraction
    )
    write_portia4_outputs(args.output_dir, manifest, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit("PORTIA-4 manifest validation failed")


if __name__ == "__main__":
    main()
