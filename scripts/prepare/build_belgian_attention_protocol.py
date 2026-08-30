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
    filter_strict_development_audio,
    load_belgian_records,
    validate_fold_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen Belgian AIS four-class UTC-date-disjoint development folds.",
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Extracted audio root; only development audio is inspected.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument(
        "--source-protocol-root",
        default=None,
        help="Optional existing frozen protocol whose validation-date assignments are preserved.",
    )
    return parser


def load_frozen_date_assignments(protocol_root: str | None) -> dict[str, int] | None:
    if protocol_root is None:
        return None
    root = Path(protocol_root).expanduser().resolve()
    assignments: dict[str, int] = {}
    for fold in range(1, 4):
        manifest = json.loads(
            (root / f"fold{fold}" / "split_manifest.json").read_text(encoding="utf-8")
        )
        validation = validate_fold_manifest(manifest)
        if validation["status"] != "passed":
            raise ValueError(f"Source Belgian fold {fold} is invalid: {validation}")
        for row in manifest["records"]:
            if row["split"] != "val":
                continue
            date = str(row["calendar_date"])
            previous = assignments.setdefault(date, fold - 1)
            if previous != fold - 1:
                raise ValueError(f"UTC date appears as validation in multiple folds: {date}")
    return assignments


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
    records, audio_audit = filter_strict_development_audio(
        records,
        data_root=args.data_root,
    )
    frozen_date_assignments = load_frozen_date_assignments(args.source_protocol_root)
    manifests, index = build_fold_manifests(
        records,
        audit,
        folds=3,
        fold_seed=args.fold_seed,
        audio_audit=audio_audit,
        frozen_date_assignments=frozen_date_assignments,
    )
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "metadata_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    persisted_audio_audit = dict(audio_audit)
    persisted_audio_audit.pop("inventory", None)
    (audit_dir / "development_audio_admission.json").write_text(
        json.dumps(persisted_audio_audit, indent=2, ensure_ascii=False) + "\n",
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
    index["strict_development_record_count"] = audio_audit["development_admitted"]
    (output_root / "protocol_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
