from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterable

import soundfile as sf

from src.data.deepship_metadata import METADATA_FILES
from src.utils.pathing import resolve_manifest_path, validate_manifest_relative_path


CLASS_NAMES = tuple(METADATA_FILES)
INVENTORY_FIELDS = [
    "relative_path",
    "class_name",
    "sample_rate",
    "num_frames",
    "channels",
    "duration_seconds",
    "file_size_bytes",
    "content_sha256",
    "full_segments",
]
IDENTITY_FIELDS = [
    "relative_path",
    "class_name",
    "match_status",
    "match_method",
    "match_confidence",
    "metadata_record_id",
    "ais_type_code",
    "metadata_date",
    "metadata_time",
    "raw_vessel_name",
    "canonical_vessel_name",
    "mmsi",
    "vessel_key",
    "ambiguous_vessel_name",
]
EXCLUSION_FIELDS = [
    "relative_path",
    "class_name",
    "reason",
    "match_status",
    "match_confidence",
]


def load_experiment_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, object]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Experiment config schema_version must be 1")
    if tuple(config.get("classes", [])) != CLASS_NAMES:
        raise ValueError(f"Experiment classes must be {CLASS_NAMES}")

    model = config.get("model")
    if not isinstance(model, dict) or int(model.get("expected_num_parameters", 0)) <= 0:
        raise ValueError("model.expected_num_parameters must be a positive integer")

    features = config.get("features")
    if not isinstance(features, dict) or float(features.get("clip_duration_seconds", 0)) <= 0:
        raise ValueError("features.clip_duration_seconds must be positive")

    split = config.get("split")
    if not isinstance(split, dict):
        raise ValueError("split configuration is required")
    targets = split.get("target_segments_per_class")
    if not isinstance(targets, dict) or set(targets) != {"train", "val", "test"}:
        raise ValueError("split.target_segments_per_class must define train, val, and test")
    if any(int(value) <= 0 for value in targets.values()):
        raise ValueError("All target segment counts must be positive")
    expected_protocols = {"segment_level", "recording_disjoint", "vessel_name_disjoint"}
    if set(split.get("protocols", [])) != expected_protocols:
        raise ValueError(f"split.protocols must be {sorted(expected_protocols)}")

    training = config.get("training")
    seeds = training.get("model_seeds") if isinstance(training, dict) else None
    if not isinstance(seeds, list) or len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("training.model_seeds must contain unique values")


def validate_relative_path(value: str) -> PurePosixPath:
    return validate_manifest_relative_path(value)


def stable_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory_rows(
    data_root: str | Path,
    manifest_rows: Iterable[dict[str, object]],
    *,
    clip_duration_seconds: float,
    hash_audio: bool = True,
) -> list[dict[str, object]]:
    root = Path(data_root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for manifest_row in sorted(manifest_rows, key=lambda row: str(row["relative_path"])):
        relative_path = str(manifest_row["relative_path"])
        relative = validate_relative_path(relative_path)
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate recording in manifest: {relative_path}")
        seen_paths.add(relative_path)
        audio_path = resolve_manifest_path(root, relative.as_posix())
        if not audio_path.is_file():
            raise FileNotFoundError(f"Manifest audio is unavailable: {relative_path}")
        info = sf.info(str(audio_path))
        clip_frames = int(round(info.samplerate * clip_duration_seconds))
        rows.append(
            {
                "relative_path": relative_path,
                "class_name": str(manifest_row["class_name"]),
                "sample_rate": info.samplerate,
                "num_frames": info.frames,
                "channels": info.channels,
                "duration_seconds": round(info.frames / info.samplerate, 6),
                "file_size_bytes": audio_path.stat().st_size,
                "content_sha256": file_sha256(audio_path) if hash_audio else "",
                "full_segments": info.frames // clip_frames,
            }
        )
    return rows


def portable_identity_rows(
    manifest_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sorted(manifest_rows, key=lambda row: str(row["relative_path"])):
        validate_relative_path(str(source["relative_path"]))
        rows.append({field: source.get(field, "") for field in IDENTITY_FIELDS})
    return rows


def build_exclusion_rows(
    identity_rows: Iterable[dict[str, object]],
    *,
    allowed_confidence: set[str],
) -> list[dict[str, object]]:
    exclusions: list[dict[str, object]] = []
    for row in identity_rows:
        status = str(row["match_status"])
        confidence = str(row["match_confidence"])
        if status != "matched" or not row["vessel_key"]:
            reason = "unresolved_vessel_identity"
        elif confidence not in allowed_confidence:
            reason = "match_confidence_not_allowed"
        else:
            continue
        exclusions.append(
            {
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "reason": reason,
                "match_status": status,
                "match_confidence": confidence,
            }
        )
    return exclusions


def _class_counts(rows: Iterable[dict[str, object]], field: str | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        amount = int(row[field]) if field is not None else 1
        counts[str(row["class_name"])] += amount
    return {class_name: counts.get(class_name, 0) for class_name in CLASS_NAMES}


def build_audit_report(
    config: dict[str, object],
    inventory_rows: list[dict[str, object]],
    identity_rows: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    *,
    metadata_record_count: int,
    unmatched_metadata_count: int,
    parse_issues: list[dict[str, object]],
) -> dict[str, object]:
    validate_experiment_config(config)
    inventory_by_path = {str(row["relative_path"]): row for row in inventory_rows}
    identity_by_path = {str(row["relative_path"]): row for row in identity_rows}
    if set(inventory_by_path) != set(identity_by_path):
        raise ValueError("Inventory and identity manifests must contain identical recording paths")

    excluded_paths = {str(row["relative_path"]) for row in exclusions}
    included_identity = [
        row for row in identity_rows if str(row["relative_path"]) not in excluded_paths
    ]
    included_paths = {str(row["relative_path"]) for row in included_identity}
    included_inventory = [
        row for row in inventory_rows if str(row["relative_path"]) in included_paths
    ]

    confidence_counts = Counter(str(row["match_confidence"]) for row in included_identity)
    method_counts = Counter(str(row["match_method"]) for row in included_identity)
    ambiguous_rows = [row for row in included_identity if row["ambiguous_vessel_name"] is True]
    vessel_groups_by_class: dict[str, set[str]] = defaultdict(set)
    vessel_classes: dict[str, set[str]] = defaultdict(set)
    for row in included_identity:
        class_name = str(row["class_name"])
        vessel_key = str(row["vessel_key"])
        vessel_groups_by_class[class_name].add(vessel_key)
        vessel_classes[vessel_key].add(class_name)

    content_groups: dict[str, list[str]] = defaultdict(list)
    content_classes: dict[str, set[str]] = defaultdict(set)
    content_vessels: dict[str, set[str]] = defaultdict(set)
    for inventory_row in inventory_rows:
        content_hash = str(inventory_row.get("content_sha256", ""))
        if not content_hash:
            continue
        relative_path = str(inventory_row["relative_path"])
        content_groups[content_hash].append(relative_path)
        content_classes[content_hash].add(str(inventory_row["class_name"]))
        vessel_key = str(identity_by_path[relative_path].get("vessel_key", ""))
        if vessel_key:
            content_vessels[content_hash].add(vessel_key)
    duplicate_content_groups = {
        content_hash: sorted(paths)
        for content_hash, paths in sorted(content_groups.items())
        if len(paths) > 1
    }
    duplicate_hashes_crossing_classes = sorted(
        content_hash
        for content_hash in duplicate_content_groups
        if len(content_classes[content_hash]) > 1
    )
    duplicate_hashes_crossing_vessels = sorted(
        content_hash
        for content_hash in duplicate_content_groups
        if len(content_vessels[content_hash]) > 1
    )

    split = config["split"]
    assert isinstance(split, dict)
    targets = split["target_segments_per_class"]
    assert isinstance(targets, dict)
    target_total = sum(int(value) for value in targets.values())
    all_segments = _class_counts(inventory_rows, "full_segments")
    vessel_segments = _class_counts(included_inventory, "full_segments")
    recordings_by_class = _class_counts(inventory_rows)
    vessel_recordings_by_class = _class_counts(included_inventory)
    groups_by_class = {
        class_name: len(vessel_groups_by_class[class_name]) for class_name in CLASS_NAMES
    }

    feasibility = {
        "target_segments_per_class_total": target_total,
        "segment_level": {
            class_name: all_segments[class_name] >= target_total for class_name in CLASS_NAMES
        },
        "recording_disjoint_preliminary": {
            class_name: (
                all_segments[class_name] >= target_total and recordings_by_class[class_name] >= 3
            )
            for class_name in CLASS_NAMES
        },
        "vessel_name_disjoint_preliminary": {
            class_name: (
                vessel_segments[class_name] >= target_total and groups_by_class[class_name] >= 3
            )
            for class_name in CLASS_NAMES
        },
        "note": "Preliminary checks use total capacity and group counts; exact group allocation is validated by the protocol compiler.",
    }

    validation = {
        "inventory_paths_unique": len(inventory_by_path) == len(inventory_rows),
        "identity_paths_unique": len(identity_by_path) == len(identity_rows),
        "inventory_identity_paths_equal": set(inventory_by_path) == set(identity_by_path),
        "all_classes_present": all(recordings_by_class[class_name] > 0 for class_name in CLASS_NAMES),
        "included_identities_have_vessel_key": all(
            bool(row["vessel_key"]) for row in included_identity
        ),
        "vessel_keys_do_not_cross_classes": all(
            len(classes) == 1 for classes in vessel_classes.values()
        ),
        "all_recordings_content_hashed": all(
            bool(row.get("content_sha256")) for row in inventory_rows
        ),
        "duplicate_content_does_not_cross_classes": not duplicate_hashes_crossing_classes,
        "duplicate_content_does_not_cross_vessel_keys": not duplicate_hashes_crossing_vessels,
    }
    all_budgets_preliminarily_feasible = all(
        all(bool(value) for value in feasibility[protocol].values())
        for protocol in (
            "segment_level",
            "recording_disjoint_preliminary",
            "vessel_name_disjoint_preliminary",
        )
    )

    sanitized_issues = []
    for issue in parse_issues:
        sanitized = dict(issue)
        if "source_path" in sanitized:
            sanitized["source_file"] = Path(str(sanitized.pop("source_path"))).name
        sanitized_issues.append(sanitized)

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": (
            "passed"
            if all(validation.values()) and all_budgets_preliminarily_feasible
            else "failed"
        ),
        "dataset_inventory_sha256": stable_json_hash(inventory_rows),
        "recording_identity_manifest_sha256": stable_json_hash(identity_rows),
        "recordings": len(inventory_rows),
        "recordings_by_class": recordings_by_class,
        "full_segments_by_class": all_segments,
        "metadata_records": metadata_record_count,
        "unmatched_metadata_records": unmatched_metadata_count,
        "identity_included_recordings": len(included_identity),
        "identity_included_recordings_by_class": vessel_recordings_by_class,
        "identity_excluded_recordings": len(exclusions),
        "identity_match_confidence": dict(sorted(confidence_counts.items())),
        "identity_match_methods": dict(sorted(method_counts.items())),
        "vessel_name_groups": len(vessel_classes),
        "vessel_name_groups_by_class": groups_by_class,
        "vessel_keys_crossing_classes": sorted(
            vessel_key for vessel_key, classes in vessel_classes.items() if len(classes) > 1
        ),
        "ambiguous_name_recordings_included_conservatively": len(ambiguous_rows),
        "ambiguous_canonical_names": sorted(
            {str(row["canonical_vessel_name"]) for row in ambiguous_rows}
        ),
        "metadata_parse_issues": sanitized_issues,
        "duplicate_content_check": {
            "status": "completed" if validation["all_recordings_content_hashed"] else "not_run",
            "duplicate_groups": len(duplicate_content_groups),
            "duplicate_files": sum(len(paths) for paths in duplicate_content_groups.values()),
            "groups": duplicate_content_groups,
            "hashes_crossing_classes": duplicate_hashes_crossing_classes,
            "hashes_crossing_vessel_keys": duplicate_hashes_crossing_vessels,
        },
        "preliminary_budget_feasibility": feasibility,
        "validation": validation,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_audit_outputs(
    output_dir: str | Path,
    inventory_rows: list[dict[str, object]],
    identity_rows: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_inventory.csv", inventory_rows, INVENTORY_FIELDS)
    write_csv(output / "recording_identity_manifest.csv", identity_rows, IDENTITY_FIELDS)
    write_csv(output / "identity_exclusions.csv", exclusions, EXCLUSION_FIELDS)
    (output / "identity_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
