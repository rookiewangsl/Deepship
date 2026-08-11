from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Iterable

from src.data.deepship_audit import CLASS_NAMES, stable_json_hash
from src.utils.pathing import validate_manifest_relative_path


SPLITS = ("train", "val", "test")
ASSIGNMENT_ORDER = ("test", "val", "train")
SEGMENT_FIELDS = [
    "split",
    "class_name",
    "label_index",
    "relative_path",
    "group_key",
    "vessel_key",
    "segment_index",
    "start_frame",
    "num_frames",
    "sample_rate",
    "total_segments",
]
RECORDING_ASSIGNMENT_FIELDS = [
    "relative_path",
    "class_name",
    "vessel_key",
    "partitions",
    "selected_segments",
]
GROUP_ASSIGNMENT_FIELDS = [
    "group_key",
    "class_name",
    "split",
    "available_segments",
    "recordings",
]


def load_inventory(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            relative_path = validate_manifest_relative_path(source["relative_path"]).as_posix()
            rows.append(
                {
                    "relative_path": relative_path,
                    "class_name": source["class_name"],
                    "sample_rate": int(source["sample_rate"]),
                    "num_frames": int(source["num_frames"]),
                    "channels": int(source["channels"]),
                    "duration_seconds": float(source["duration_seconds"]),
                    "file_size_bytes": int(source["file_size_bytes"]),
                    "content_sha256": source.get("content_sha256", ""),
                    "full_segments": int(source["full_segments"]),
                }
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def load_identity_manifest(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            relative_path = validate_manifest_relative_path(source["relative_path"]).as_posix()
            ambiguous_value = source["ambiguous_vessel_name"]
            ambiguous: bool | str
            if ambiguous_value == "True":
                ambiguous = True
            elif ambiguous_value == "False":
                ambiguous = False
            else:
                ambiguous = ""
            metadata_record_id: int | str = (
                int(source["metadata_record_id"]) if source["metadata_record_id"] else ""
            )
            rows.append(
                {
                    **source,
                    "relative_path": relative_path,
                    "metadata_record_id": metadata_record_id,
                    "ambiguous_vessel_name": ambiguous,
                }
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _segment_rows_for_recording(
    recording: dict[str, object],
    *,
    group_key: str,
    split: str,
    clip_duration_seconds: float,
) -> list[dict[str, object]]:
    sample_rate = int(recording["sample_rate"])
    clip_frames = int(round(sample_rate * clip_duration_seconds))
    total_segments = int(recording["num_frames"]) // clip_frames
    class_name = str(recording["class_name"])
    label_index = CLASS_NAMES.index(class_name)
    return [
        {
            "split": split,
            "class_name": class_name,
            "label_index": label_index,
            "relative_path": recording["relative_path"],
            "group_key": group_key,
            "segment_index": segment_index,
            "start_frame": segment_index * clip_frames,
            "num_frames": clip_frames,
            "sample_rate": sample_rate,
            "total_segments": total_segments,
        }
        for segment_index in range(total_segments)
    ]


def _choose_group_subset(
    groups: list[dict[str, object]],
    *,
    target: int,
    reserve_capacity: int,
    reserve_groups: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    total_capacity = sum(int(group["available_segments"]) for group in groups)
    maximum_selected_capacity = total_capacity - reserve_capacity
    if maximum_selected_capacity < target:
        raise ValueError(
            f"Insufficient group capacity: target={target}, total={total_capacity}, "
            f"reserve={reserve_capacity}"
        )

    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for group_index, group in enumerate(groups):
        capacity = int(group["available_segments"])
        if capacity <= 0:
            continue
        for current_total, subset in list(reachable.items())[::-1]:
            new_total = current_total + capacity
            if new_total > maximum_selected_capacity or new_total in reachable:
                continue
            reachable[new_total] = subset + (group_index,)

    candidates = [
        (total, subset)
        for total, subset in reachable.items()
        if total >= target and len(groups) - len(subset) >= reserve_groups
    ]
    if not candidates:
        raise ValueError(
            f"No group-disjoint allocation can meet target={target} while reserving "
            f"capacity={reserve_capacity} and groups={reserve_groups}"
        )
    _, selected_indexes = min(candidates, key=lambda item: (item[0], -len(item[1]), item[1]))
    selected_index_set = set(selected_indexes)
    selected = [group for index, group in enumerate(groups) if index in selected_index_set]
    remaining = [group for index, group in enumerate(groups) if index not in selected_index_set]
    return selected, remaining


def _build_groups(
    protocol: str,
    inventory_rows: list[dict[str, object]],
    identity_by_path: dict[str, dict[str, object]],
    excluded_paths: set[str],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    content_counts = Counter(
        str(recording.get("content_sha256", ""))
        for recording in inventory_rows
        if recording.get("content_sha256")
    )
    for recording in inventory_rows:
        relative_path = str(recording["relative_path"])
        class_name = str(recording["class_name"])
        if protocol == "vessel_name_disjoint":
            if relative_path in excluded_paths:
                continue
            vessel_key = str(identity_by_path[relative_path]["vessel_key"])
            if not vessel_key:
                raise ValueError(f"Vessel protocol recording has no vessel_key: {relative_path}")
            group_key = vessel_key
        elif protocol == "recording_disjoint":
            content_hash = str(recording.get("content_sha256", ""))
            group_key = (
                f"CONTENT:{content_hash}"
                if content_hash and content_counts[content_hash] > 1
                else relative_path
            )
        else:
            raise ValueError(f"Unsupported group protocol: {protocol}")
        grouped[(class_name, group_key)].append(recording)

    groups_by_class: dict[str, list[dict[str, object]]] = {name: [] for name in CLASS_NAMES}
    for (class_name, group_key), recordings in sorted(grouped.items()):
        groups_by_class[class_name].append(
            {
                "group_key": group_key,
                "class_name": class_name,
                "available_segments": sum(int(row["full_segments"]) for row in recordings),
                "recordings": sorted(recordings, key=lambda row: str(row["relative_path"])),
            }
        )
    return groups_by_class


def _compile_segment_level(
    config: dict[str, object],
    inventory_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split_config = config["split"]
    features = config["features"]
    assert isinstance(split_config, dict) and isinstance(features, dict)
    seed = int(split_config["split_seed"])
    targets = split_config["target_segments_per_class"]
    assert isinstance(targets, dict)
    clip_duration = float(features["clip_duration_seconds"])
    selected_segments: list[dict[str, object]] = []

    for class_name in CLASS_NAMES:
        candidates: list[dict[str, object]] = []
        for recording in inventory_rows:
            if recording["class_name"] != class_name:
                continue
            candidates.extend(
                _segment_rows_for_recording(
                    recording,
                    group_key="",
                    split="",
                    clip_duration_seconds=clip_duration,
                )
            )
        rng = random.Random(f"{seed}:segment_level:{class_name}")
        rng.shuffle(candidates)
        offset = 0
        for split in SPLITS:
            count = int(targets[split])
            chosen = candidates[offset : offset + count]
            if len(chosen) != count:
                raise ValueError(f"Class {class_name} cannot provide {count} {split} segments")
            for segment in chosen:
                segment["split"] = split
                segment["group_key"] = (
                    f"{segment['relative_path']}#{int(segment['segment_index']):06d}"
                )
            selected_segments.extend(chosen)
            offset += count
    return selected_segments, []


def _compile_group_protocol(
    protocol: str,
    config: dict[str, object],
    inventory_rows: list[dict[str, object]],
    identity_by_path: dict[str, dict[str, object]],
    excluded_paths: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split_config = config["split"]
    features = config["features"]
    assert isinstance(split_config, dict) and isinstance(features, dict)
    seed = int(split_config["split_seed"])
    targets = split_config["target_segments_per_class"]
    assert isinstance(targets, dict)
    clip_duration = float(features["clip_duration_seconds"])
    groups_by_class = _build_groups(protocol, inventory_rows, identity_by_path, excluded_paths)
    selected_segments: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    for class_name in CLASS_NAMES:
        rng = random.Random(f"{seed}:{protocol}:{class_name}")
        remaining_groups = groups_by_class[class_name][:]
        rng.shuffle(remaining_groups)
        total_capacity = sum(int(group["available_segments"]) for group in remaining_groups)
        target_total = sum(int(targets[split]) for split in SPLITS)
        desired_test_capacity = round(total_capacity * int(targets["test"]) / target_total)
        desired_val_capacity = round(total_capacity * int(targets["val"]) / target_total)
        test_groups, remaining_groups = _choose_group_subset(
            remaining_groups,
            target=max(int(targets["test"]), desired_test_capacity),
            reserve_capacity=int(targets["train"]) + desired_val_capacity,
            reserve_groups=2,
        )
        val_groups, train_groups = _choose_group_subset(
            remaining_groups,
            target=max(int(targets["val"]), desired_val_capacity),
            reserve_capacity=int(targets["train"]),
            reserve_groups=1,
        )
        assignments: dict[str, list[dict[str, object]]] = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }

        for split, groups in assignments.items():
            for group in groups:
                assignment_rows.append(
                    {
                        "group_key": group["group_key"],
                        "class_name": class_name,
                        "split": split,
                        "available_segments": group["available_segments"],
                        "recordings": len(group["recordings"]),
                    }
                )
                candidates: list[dict[str, object]] = []
                for recording in group["recordings"]:
                    candidates.extend(
                        _segment_rows_for_recording(
                            recording,
                            group_key=str(group["group_key"]),
                            split=split,
                            clip_duration_seconds=clip_duration,
                        )
                    )
                group["candidate_segments"] = candidates

        for split in SPLITS:
            candidates = []
            for group in assignments[split]:
                candidates.extend(group["candidate_segments"])
            split_rng = random.Random(f"{seed}:{protocol}:{class_name}:{split}:segments")
            split_rng.shuffle(candidates)
            count = int(targets[split])
            if len(candidates) < count:
                raise ValueError(
                    f"{protocol} class {class_name} {split} has {len(candidates)} segments, "
                    f"fewer than target {count}"
                )
            selected_segments.extend(candidates[:count])

    return selected_segments, assignment_rows


def _segment_id(segment: dict[str, object]) -> str:
    return f"{segment['relative_path']}#{int(segment['segment_index']):06d}"


def _partition_map(
    segments: Iterable[dict[str, object]],
    key_field: str,
) -> dict[str, set[str]]:
    partitions: dict[str, set[str]] = defaultdict(set)
    for segment in segments:
        partitions[str(segment[key_field])].add(str(segment["split"]))
    return partitions


def _build_recording_assignments(
    segments: list[dict[str, object]],
    identity_by_path: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    partitions = _partition_map(segments, "relative_path")
    selected_counts = Counter(str(segment["relative_path"]) for segment in segments)
    class_by_path = {
        str(segment["relative_path"]): str(segment["class_name"]) for segment in segments
    }
    return [
        {
            "relative_path": relative_path,
            "class_name": class_by_path[relative_path],
            "vessel_key": identity_by_path.get(relative_path, {}).get("vessel_key", ""),
            "partitions": ";".join(sorted(partitions[relative_path])),
            "selected_segments": selected_counts[relative_path],
        }
        for relative_path in sorted(partitions)
    ]


def compile_protocol(
    protocol: str,
    config: dict[str, object],
    inventory_rows: list[dict[str, object]],
    identity_rows: list[dict[str, object]],
    exclusion_rows: list[dict[str, object]],
    *,
    source_inventory_sha256: str,
    source_identity_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    protocols = config["split"]["protocols"]  # type: ignore[index]
    if protocol not in protocols:
        raise ValueError(f"Protocol is not enabled by experiment config: {protocol}")
    identity_by_path = {str(row["relative_path"]): row for row in identity_rows}
    excluded_paths = {str(row["relative_path"]) for row in exclusion_rows}
    if {str(row["relative_path"]) for row in inventory_rows} != set(identity_by_path):
        raise ValueError("Inventory and identity rows do not cover identical paths")

    if protocol == "segment_level":
        segments, group_assignments = _compile_segment_level(config, inventory_rows)
    else:
        segments, group_assignments = _compile_group_protocol(
            protocol,
            config,
            inventory_rows,
            identity_by_path,
            excluded_paths,
        )
    segments.sort(
        key=lambda row: (
            SPLITS.index(str(row["split"])),
            str(row["class_name"]),
            str(row["relative_path"]),
            int(row["segment_index"]),
        )
    )
    group_assignments.sort(
        key=lambda row: (
            str(row["class_name"]),
            str(row["split"]),
            str(row["group_key"]),
        )
    )
    recording_assignments = _build_recording_assignments(segments, identity_by_path)
    for segment in segments:
        segment["vessel_key"] = identity_by_path.get(
            str(segment["relative_path"]), {}
        ).get("vessel_key", "")

    segment_ids = [_segment_id(segment) for segment in segments]
    recording_partitions = _partition_map(segments, "relative_path")
    vessel_partitions: dict[str, set[str]] = defaultdict(set)
    for segment in segments:
        identity = identity_by_path.get(str(segment["relative_path"]), {})
        vessel_key = str(identity.get("vessel_key", ""))
        if vessel_key:
            vessel_partitions[vessel_key].add(str(segment["split"]))
    selected_counts: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in CLASS_NAMES} for split in SPLITS
    }
    for segment in segments:
        selected_counts[str(segment["split"])][str(segment["class_name"])] += 1

    targets = config["split"]["target_segments_per_class"]  # type: ignore[index]
    exact_budget = all(
        selected_counts[split][class_name] == int(targets[split])
        for split in SPLITS
        for class_name in CLASS_NAMES
    )
    recording_overlap = sorted(
        path for path, partitions in recording_partitions.items() if len(partitions) > 1
    )
    vessel_overlap = sorted(
        key for key, partitions in vessel_partitions.items() if len(partitions) > 1
    )
    validation = {
        "segment_ids_unique": len(segment_ids) == len(set(segment_ids)),
        "exact_target_segments_per_class": exact_budget,
        "recording_disjoint": not recording_overlap,
        "vessel_name_disjoint": not vessel_overlap,
    }
    required_checks = ["segment_ids_unique", "exact_target_segments_per_class"]
    if protocol in {"recording_disjoint", "vessel_name_disjoint"}:
        required_checks.append("recording_disjoint")
    if protocol == "vessel_name_disjoint":
        required_checks.append("vessel_name_disjoint")
    status = "passed" if all(validation[name] for name in required_checks) else "failed"

    manifest_core = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol": protocol,
        "split_seed": config["split"]["split_seed"],  # type: ignore[index]
        "source_inventory_sha256": source_inventory_sha256,
        "source_identity_sha256": source_identity_sha256,
        "target_segments_per_class": targets,
        "segments": segments,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": stable_json_hash(manifest_core),
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol": protocol,
        "status": status,
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_segments": len(segments),
        "selected_segments_by_split_and_class": selected_counts,
        "selected_recordings": len(recording_partitions),
        "selected_vessel_name_groups": len(vessel_partitions),
        "recordings_crossing_partitions": recording_overlap,
        "vessel_name_groups_crossing_partitions": vessel_overlap,
        "group_assignments": len(group_assignments),
        "unused_groups": sum(row["split"] == "unused" for row in group_assignments),
        "validation": validation,
        "required_validation_checks": required_checks,
    }
    return manifest, recording_assignments, group_assignments, report


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_outputs(
    output_dir: str | Path,
    manifest: dict[str, object],
    recording_assignments: list[dict[str, object]],
    group_assignments: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "recording_assignments.csv",
        recording_assignments,
        RECORDING_ASSIGNMENT_FIELDS,
    )
    _write_csv(
        output / "group_assignments.csv",
        group_assignments,
        GROUP_ASSIGNMENT_FIELDS,
    )
