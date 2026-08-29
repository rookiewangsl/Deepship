from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship_audit import load_experiment_config, stable_json_hash  # noqa: E402
from src.data.deepship_protocols import (  # noqa: E402
    load_identity_manifest,
    load_inventory,
    write_protocol_outputs,
)
from src.data.deepship_repeats import compile_repeat_vessel_split  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic repeated DeepShip vessel-disjoint manifests."
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--audit-dir",
        default=str(ROOT / "protocols" / "isolation_comparison_v1" / "audit"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "protocols" / "macnna_global_l20_repeats_v1"),
    )
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[43, 44])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    base_config = load_experiment_config(args.base_config)
    inventory_rows = load_inventory(audit_dir / "dataset_inventory.csv")
    identity_rows = load_identity_manifest(audit_dir / "recording_identity_manifest.csv")
    with (audit_dir / "identity_exclusions.csv").open(newline="", encoding="utf-8") as handle:
        exclusion_rows = list(csv.DictReader(handle))
    audit_report = json.loads((audit_dir / "identity_audit.json").read_text(encoding="utf-8"))
    if audit_report.get("status") != "passed":
        raise SystemExit("Identity audit has not passed")
    inventory_hash = stable_json_hash(inventory_rows)
    identity_hash = stable_json_hash(identity_rows)
    if inventory_hash != audit_report.get("dataset_inventory_sha256"):
        raise SystemExit("Dataset inventory hash does not match identity audit")
    if identity_hash != audit_report.get("recording_identity_manifest_sha256"):
        raise SystemExit("Identity manifest hash does not match identity audit")

    index_rows = []
    for split_seed in sorted(set(args.split_seeds)):
        if split_seed == int(base_config["split"]["split_seed"]):  # type: ignore[index]
            raise SystemExit("The base split already covers this seed; only build new split seeds")
        split_root = output_root / f"split_seed{split_seed}"
        if split_root.exists() and any(split_root.iterdir()):
            raise FileExistsError(f"Repeat split output is not empty: {split_root}")
        config, manifest, recordings, groups, report = compile_repeat_vessel_split(
            base_config,
            inventory_rows,
            identity_rows,
            exclusion_rows,
            split_seed=split_seed,
            source_inventory_sha256=inventory_hash,
            source_identity_sha256=identity_hash,
        )
        if report.get("status") != "passed":
            raise SystemExit(f"Repeat split validation failed for seed {split_seed}: {report}")
        audit_output = split_root / "audit"
        audit_output.mkdir(parents=True, exist_ok=False)
        for name in (
            "dataset_inventory.csv",
            "recording_identity_manifest.csv",
            "identity_exclusions.csv",
            "identity_audit.json",
        ):
            shutil.copyfile(audit_dir / name, audit_output / name)
        (split_root / "experiment_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_protocol_outputs(
            split_root / "vessel_name_disjoint",
            manifest,
            recordings,
            groups,
            report,
        )
        index_rows.append(
            {
                "split_seed": split_seed,
                "experiment_id": config["experiment_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "selected_recordings": report["selected_recordings"],
                "selected_vessel_name_groups": report["selected_vessel_name_groups"],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "repeat_split_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_split_seed": base_config["split"]["split_seed"],  # type: ignore[index]
                "repeat_splits": index_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
