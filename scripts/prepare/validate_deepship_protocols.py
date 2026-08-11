from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship_audit import load_experiment_config  # noqa: E402
from src.data.deepship_protocol_validation import (  # noqa: E402
    load_split_manifest,
    validate_protocol_manifest,
)
from src.utils.pathing import default_deepship_root  # noqa: E402


PROTOCOLS = ("segment_level", "recording_disjoint", "vessel_name_disjoint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate frozen DeepShip protocol manifests against local audio files.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--protocol-root",
        default=str(ROOT / "protocols" / "isolation_comparison_v1"),
    )
    parser.add_argument(
        "--protocol",
        choices=("all", *PROTOCOLS),
        default="all",
    )
    parser.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Validate without rewriting committed validation_report.json files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    config = load_experiment_config(args.config)
    audit_report = json.loads(
        (protocol_root / "audit" / "identity_audit.json").read_text(encoding="utf-8")
    )
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    reports = []
    for protocol in protocols:
        protocol_dir = protocol_root / protocol
        manifest = load_split_manifest(protocol_dir / "split_manifest.json")
        report = validate_protocol_manifest(
            manifest,
            config,
            audit_report,
            data_root=args.data_root,
        )
        if not args.no_write_reports:
            (protocol_dir / "validation_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        reports.append(report)
        if report["status"] != "passed":
            print(json.dumps(reports, indent=2, ensure_ascii=False))
            raise SystemExit(f"Protocol validation failed: {protocol}")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
