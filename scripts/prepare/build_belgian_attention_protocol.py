from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.belgian_ais import (  # noqa: E402
    build_fold_manifests,
    load_belgian_records,
    validate_fold_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen Belgian AIS four-class UTC-date-disjoint development folds.",
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    parser.add_argument("--fold-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty protocol root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records, audit = load_belgian_records(
        args.metadata_csv,
        args.split_dir,
        max_distance_km=args.max_distance_km,
    )
    manifests, index = build_fold_manifests(records, audit, folds=3, fold_seed=args.fold_seed)
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "metadata_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validations = []
    for fold, manifest in enumerate(manifests, start=1):
        fold_dir = output_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_fold_manifest(manifest)
        if validation["status"] != "passed":
            raise RuntimeError(f"Generated Belgian fold {fold} is invalid: {validation}")
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (fold_dir / "validation_report.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validations.append(validation)
    sealed_dir = output_root / "sealed_test"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    sealed = index.pop("sealed_test_manifest")
    (sealed_dir / "split_manifest.json").write_text(
        json.dumps(sealed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index["validations"] = validations
    index["eligible_record_count"] = len(records)
    (output_root / "protocol_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
