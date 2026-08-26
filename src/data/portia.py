"""Build a leakage-safe four-class external-evaluation manifest for PORTIA."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import csv
import json
import math
from pathlib import Path
import random

from src.data.deepship_audit import CLASS_NAMES, stable_json_hash


PORTIA_TO_DEEPSHIP_CLASS = {
    "cargo": "Cargo",
    "passenger": "Passenger",
    "tanker": "Tank",
    "tug": "Tug",
}
OFFICIAL_SPLITS = ("train", "val", "test")
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class PortiaWindowRecord:
    """One valid PORTIA single-vessel window with its AIS-derived group key."""

    window_id: str
    class_name: str
    label_index: int
    primary_mmsi: str
    primary_distance_km: float
    source_split: str


def _normalise_mmsi(raw_value: str) -> str:
    try:
        numeric_value = Decimal(raw_value.strip())
    except (InvalidOperation, AttributeError) as error:
        raise ValueError(f"Invalid PORTIA MMSI: {raw_value!r}") from error
    if not numeric_value.is_finite() or numeric_value <= 0 or numeric_value != numeric_value.to_integral_value():
        raise ValueError(f"Invalid PORTIA MMSI: {raw_value!r}")
    return str(int(numeric_value))


def _normalise_window_id(raw_value: str) -> str:
    window_id = raw_value.strip()
    if not window_id or Path(window_id).name != window_id or not window_id.endswith(".wav"):
        raise ValueError(f"Invalid PORTIA window ID: {raw_value!r}")
    return window_id


def load_portia4_records(annotation_root: str | Path) -> list[PortiaWindowRecord]:
    """Load the four directly comparable classes from official PORTIA CSVs.

    Only official ``Single_Vessel_Classification`` rows with exactly one labelled
    vessel are eligible.  The source split is retained for provenance but is not
    used for evaluation allocation: a single MMSI can occur in several official
    files, so an independent external split must be MMSI-disjoint.
    """

    root = Path(annotation_root).expanduser().resolve()
    records: list[PortiaWindowRecord] = []
    seen_windows: set[str] = set()
    for source_split in OFFICIAL_SPLITS:
        csv_path = root / f"{source_split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"PORTIA annotation CSV is missing: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_class = row.get("primary_class", "").strip().lower()
                if raw_class not in PORTIA_TO_DEEPSHIP_CLASS:
                    continue
                if row.get("label_vessel", "").strip() != "1" or row.get("n_vessels", "").strip() != "1":
                    continue
                try:
                    primary_mmsi = _normalise_mmsi(row.get("primary_mmsi", ""))
                    window_id = _normalise_window_id(row.get("window_id", ""))
                    distance_km = float(row.get("primary_dist", ""))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(distance_km) or distance_km < 0:
                    continue
                if window_id in seen_windows:
                    raise ValueError(f"PORTIA window appears more than once: {window_id}")
                seen_windows.add(window_id)
                class_name = PORTIA_TO_DEEPSHIP_CLASS[raw_class]
                records.append(
                    PortiaWindowRecord(
                        window_id=window_id,
                        class_name=class_name,
                        label_index=CLASS_TO_INDEX[class_name],
                        primary_mmsi=primary_mmsi,
                        primary_distance_km=distance_km,
                        source_split=source_split,
                    )
                )
    if not records:
        raise ValueError("No valid four-class single-vessel PORTIA rows were found")
    return sorted(records, key=lambda record: (record.class_name, record.primary_mmsi, record.window_id))


def build_portia4_manifest(
    records: list[PortiaWindowRecord],
    *,
    seed: int = 42,
    development_fraction: float = 0.2,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create a deterministic 20/80 MMSI-disjoint development/test allocation."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be strictly between zero and one")
    by_class_and_mmsi: dict[str, dict[str, list[PortiaWindowRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_class_and_mmsi[record.class_name][record.primary_mmsi].append(record)

    assigned_rows: list[dict[str, object]] = []
    allocation: dict[str, object] = {}
    for class_name in CLASS_TO_INDEX:
        groups = by_class_and_mmsi[class_name]
        if len(groups) < 2:
            raise ValueError(f"PORTIA class {class_name} has fewer than two MMSI groups")
        mmsi_keys = sorted(groups)
        shuffled = mmsi_keys[:]
        random.Random(f"{seed}:portia4:{class_name}").shuffle(shuffled)
        development_count = min(len(shuffled) - 1, max(1, math.ceil(len(shuffled) * development_fraction)))
        development_mmsi = set(shuffled[:development_count])
        allocation[class_name] = {
            "development_mmsi": sorted(development_mmsi),
            "test_mmsi": sorted(set(mmsi_keys) - development_mmsi),
        }
        for mmsi, group_records in groups.items():
            split = "development" if mmsi in development_mmsi else "test"
            for record in group_records:
                assigned_rows.append({"split": split, **asdict(record)})

    assigned_rows.sort(key=lambda row: (str(row["split"]), str(row["class_name"]), str(row["primary_mmsi"]), str(row["window_id"])))
    checks = validate_portia4_rows(assigned_rows)
    manifest_without_hash: dict[str, object] = {
        "schema_version": 1,
        "dataset": "PORTIA v2",
        "task": "four-class zero-shot external evaluation",
        "label_mapping": PORTIA_TO_DEEPSHIP_CLASS,
        "eligibility": {
            "source": "Single_Vessel_Classification",
            "label_vessel": 1,
            "n_vessels": 1,
            "primary_classes": sorted(PORTIA_TO_DEEPSHIP_CLASS),
            "requires_primary_mmsi": True,
        },
        "allocation": {
            "unit": "primary_mmsi",
            "development_fraction": development_fraction,
            "seed": seed,
            "per_class": allocation,
        },
        "audio_path_rule": "window_id is the flat WAV filename below the configured PORTIA audio root",
        "rows": assigned_rows,
    }
    manifest = {**manifest_without_hash, "manifest_sha256": stable_json_hash(manifest_without_hash)}
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "windows_by_split_and_class": {
            split: {
                class_name: sum(row["split"] == split and row["class_name"] == class_name for row in assigned_rows)
                for class_name in CLASS_TO_INDEX
            }
            for split in ("development", "test")
        },
        "mmsi_by_split_and_class": {
            split: {
                class_name: len({str(row["primary_mmsi"]) for row in assigned_rows if row["split"] == split and row["class_name"] == class_name})
                for class_name in CLASS_TO_INDEX
            }
            for split in ("development", "test")
        },
        "source_split_counts": dict(sorted(Counter(str(row["source_split"]) for row in assigned_rows).items())),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    return manifest, summary


def validate_portia4_rows(rows: list[dict[str, object]]) -> dict[str, bool]:
    """Validate class coverage, unique windows, and zero MMSI leakage."""

    splits_by_mmsi: dict[str, set[str]] = defaultdict(set)
    classes_by_mmsi: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_mmsi[str(row["primary_mmsi"])].add(str(row["split"]))
        classes_by_mmsi[str(row["primary_mmsi"])].add(str(row["class_name"]))
    present_classes = {str(row["class_name"]) for row in rows}
    return {
        "has_all_deepship_classes": present_classes == set(CLASS_TO_INDEX),
        "unique_windows": len({str(row["window_id"]) for row in rows}) == len(rows),
        "mmsi_disjoint_between_development_and_test": all(len(splits) == 1 for splits in splits_by_mmsi.values()),
        "mmsi_has_one_class": all(len(classes) == 1 for classes in classes_by_mmsi.values()),
        "both_splits_nonempty": {str(row["split"]) for row in rows} == {"development", "test"},
    }


def write_portia4_outputs(
    output_dir: str | Path,
    manifest: dict[str, object],
    summary: dict[str, object],
) -> None:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "portia4_mmsi_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / "portia4_mmsi_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
