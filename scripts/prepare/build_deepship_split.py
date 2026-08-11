from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship_audit import load_experiment_config, stable_json_hash  # noqa: E402
from src.data.deepship_protocols import (  # noqa: E402
    compile_protocol,
    load_identity_manifest,
    load_inventory,
    write_protocol_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen DeepShip manifests for the isolation comparison.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--audit-dir",
        default=str(ROOT / "protocols" / "isolation_comparison_v1" / "audit"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "protocols" / "isolation_comparison_v1"),
    )
    parser.add_argument(
        "--protocol",
        choices=["all", "segment_level", "recording_disjoint", "vessel_name_disjoint"],
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    config = load_experiment_config(args.config)
    inventory_rows = load_inventory(audit_dir / "dataset_inventory.csv")
    identity_rows = load_identity_manifest(audit_dir / "recording_identity_manifest.csv")
    with (audit_dir / "identity_exclusions.csv").open(newline="", encoding="utf-8") as handle:
        import csv

        exclusion_rows = list(csv.DictReader(handle))
    audit_report = json.loads((audit_dir / "identity_audit.json").read_text(encoding="utf-8"))
    if audit_report["status"] != "passed":
        raise SystemExit("Identity audit has not passed")
    inventory_hash = stable_json_hash(inventory_rows)
    identity_hash = stable_json_hash(identity_rows)
    if inventory_hash != audit_report["dataset_inventory_sha256"]:
        raise SystemExit("Dataset inventory hash does not match identity audit")
    if identity_hash != audit_report["recording_identity_manifest_sha256"]:
        raise SystemExit("Identity manifest hash does not match identity audit")

    enabled_protocols = config["split"]["protocols"]  # type: ignore[index]
    protocols = enabled_protocols if args.protocol == "all" else [args.protocol]
    summaries = []
    for protocol in protocols:
        manifest, recording_assignments, group_assignments, report = compile_protocol(
            str(protocol),
            config,
            inventory_rows,
            identity_rows,
            exclusion_rows,
            source_inventory_sha256=inventory_hash,
            source_identity_sha256=identity_hash,
        )
        output_dir = Path(args.output_root) / str(protocol)
        write_protocol_outputs(
            output_dir,
            manifest,
            recording_assignments,
            group_assignments,
            report,
        )
        summaries.append(report)
        if report["status"] != "passed":
            raise SystemExit(f"Protocol validation failed: {protocol}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
