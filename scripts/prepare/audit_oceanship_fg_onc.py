from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.oceanship_onc import (  # noqa: E402
    audit_fg_metadata,
    build_probe_candidates,
    load_fg_metadata,
    write_audit_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Oceanship-FG metadata and build a small, deterministic list of "
            "events for an ONC archive lookup probe. No audio is downloaded."
        )
    )
    parser.add_argument("--fg-train-csv", required=True)
    parser.add_argument("--fg-test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-margin-seconds", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_fg_metadata([args.fg_train_csv, args.fg_test_csv])
    summary = audit_fg_metadata(records)
    candidates = build_probe_candidates(
        records,
        per_class=args.candidates_per_class,
        seed=args.seed,
        query_margin_seconds=args.query_margin_seconds,
    )
    write_audit_outputs(args.output_dir, summary, candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] == "failed":
        raise SystemExit("Oceanship-FG metadata audit failed")


if __name__ == "__main__":
    main()
