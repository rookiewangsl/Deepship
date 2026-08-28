from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], str]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {source}")
    return rows, _sha256(source)


def load_manifest(path: str | Path) -> tuple[dict[str, object], str]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError(f"Invalid split manifest: {source}")
    return payload, _sha256(source)


def _effective_count(probabilities: Iterable[float]) -> float:
    values = [float(value) for value in probabilities if value > 0]
    return 1.0 / sum(value * value for value in values) if values else 0.0


def _top_share(probabilities: Iterable[float], count: int) -> float:
    return sum(sorted((float(value) for value in probabilities), reverse=True)[:count])


def _summary(values: Iterable[float]) -> dict[str, float]:
    items = sorted(float(value) for value in values)
    if not items:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": items[0],
        "median": float(median(items)),
        "mean": sum(items) / len(items),
        "max": items[-1],
    }


def build_sampling_audit(
    manifest: dict[str, object],
    inventory_rows: list[dict[str, str]],
    recording_assignment_rows: list[dict[str, str]],
    *,
    clip_duration_seconds: float = 20.0,
    epoch_samples: int | None = None,
) -> dict[str, object]:
    if clip_duration_seconds <= 0:
        raise ValueError("clip_duration_seconds must be positive")
    segments = manifest["segments"]
    assert isinstance(segments, list)
    train_segments = [row for row in segments if row.get("split") == "train"]
    if not train_segments:
        raise ValueError("Manifest contains no training segments")
    epoch_samples = len(train_segments) if epoch_samples is None else epoch_samples
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be positive")

    class_labels: dict[str, int] = {}
    for segment in train_segments:
        class_name = str(segment["class_name"])
        label = int(segment["label_index"])
        incumbent = class_labels.setdefault(class_name, label)
        if incumbent != label:
            raise ValueError(f"Conflicting labels for class {class_name}")
    class_names = [name for name, _ in sorted(class_labels.items(), key=lambda item: item[1])]

    inventory_by_path = {row["relative_path"]: row for row in inventory_rows}
    assignments = [
        row
        for row in recording_assignment_rows
        if "train" in {value for value in row["partitions"].split(";") if value}
    ]
    if not assignments:
        raise ValueError("Recording assignments contain no training recordings")
    duplicate_paths = [
        path
        for path, count in Counter(
            row["relative_path"] for row in assignments
        ).items()
        if count > 1
    ]
    if duplicate_paths:
        raise ValueError(f"Duplicate recording assignments: {duplicate_paths[:5]}")
    missing_inventory = sorted(
        row["relative_path"] for row in assignments if row["relative_path"] not in inventory_by_path
    )
    if missing_inventory:
        raise ValueError(f"Training recordings missing from inventory: {missing_inventory[:5]}")

    assignment_by_path = {row["relative_path"]: row for row in assignments}
    manifest_train_paths = {str(row["relative_path"]) for row in train_segments}
    missing_assignments = sorted(manifest_train_paths.difference(assignment_by_path))
    if missing_assignments:
        raise ValueError(
            f"Training segments missing recording assignments: {missing_assignments[:5]}"
        )

    recordings_by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    recordings_by_vessel: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for assignment in assignments:
        relative_path = assignment["relative_path"]
        class_name = assignment["class_name"]
        vessel_key = assignment["vessel_key"]
        if class_name not in class_labels:
            raise ValueError(f"Unexpected training class in assignments: {class_name}")
        if not vessel_key:
            raise ValueError(f"Training recording has no vessel key: {relative_path}")
        inventory = inventory_by_path[relative_path]
        duration_seconds = float(inventory["duration_seconds"])
        item: dict[str, object] = {
            "relative_path": relative_path,
            "class_name": class_name,
            "vessel_key": vessel_key,
            "selected_segments": int(assignment["selected_segments"]),
            "sample_rate": int(inventory["sample_rate"]),
            "num_frames": int(inventory["num_frames"]),
            "duration_seconds": duration_seconds,
            "random_crop_span_seconds": max(0.0, duration_seconds - clip_duration_seconds),
            "nonoverlap_windows": max(1, int(duration_seconds // clip_duration_seconds)),
            "shorter_than_clip": duration_seconds < clip_duration_seconds,
        }
        recordings_by_class[class_name].append(item)
        recordings_by_vessel[(class_name, vessel_key)].append(item)

    total_s0 = sum(
        int(item["selected_segments"])
        for values in recordings_by_class.values()
        for item in values
    )
    if total_s0 != len(train_segments):
        raise ValueError(
            f"Recording selected-segment total {total_s0} does not match manifest "
            f"{len(train_segments)}"
        )
    num_classes = len(class_names)
    for class_name in class_names:
        recordings = recordings_by_class[class_name]
        vessels = {str(item["vessel_key"]) for item in recordings}
        for item in recordings:
            vessel_recordings = recordings_by_vessel[(class_name, str(item["vessel_key"]))]
            item["s0_probability"] = int(item["selected_segments"]) / total_s0
            item["s1_probability"] = 1.0 / num_classes / len(recordings)
            item["s2_probability"] = 1.0 / num_classes / len(vessels) / len(vessel_recordings)
            for policy in ("s0", "s1", "s2"):
                item[f"{policy}_expected_draws"] = epoch_samples * float(
                    item[f"{policy}_probability"]
                )

    runtime_starts: dict[str, list[int]] = defaultdict(list)
    for segment in train_segments:
        relative_path = str(segment["relative_path"])
        item = assignment_by_path[relative_path]
        inventory = inventory_by_path[relative_path]
        source_rate = int(inventory["sample_rate"])
        total_frames = int(inventory["num_frames"])
        requested_frames = int(round(source_rate * clip_duration_seconds))
        anchor_center = int(segment["start_frame"]) + int(segment["num_frames"]) // 2
        max_start = max(0, total_frames - requested_frames)
        start = min(max(0, anchor_center - requested_frames // 2), max_start)
        runtime_starts[relative_path].append(start)
        if item["class_name"] != segment["class_name"]:
            raise ValueError(f"Class mismatch for {relative_path}")

    adjacent_overlap_fractions: list[float] = []
    exact_duplicate_contexts = 0
    for recordings in recordings_by_class.values():
        for item in recordings:
            starts = runtime_starts[str(item["relative_path"])]
            unique_starts = sorted(set(starts))
            requested_frames = int(round(int(item["sample_rate"]) * clip_duration_seconds))
            overlaps = [
                max(0.0, (requested_frames - (right - left)) / requested_frames)
                for left, right in zip(unique_starts, unique_starts[1:])
            ]
            adjacent_overlap_fractions.extend(overlaps)
            item["unique_runtime_windows"] = len(unique_starts)
            item["exact_duplicate_windows"] = len(starts) - len(unique_starts)
            item["median_adjacent_overlap_fraction"] = (
                float(median(overlaps)) if overlaps else 0.0
            )
            exact_duplicate_contexts += len(starts) - len(unique_starts)

    recording_rows = sorted(
        [item for values in recordings_by_class.values() for item in values],
        key=lambda item: (
            str(item["class_name"]),
            str(item["vessel_key"]),
            str(item["relative_path"]),
        ),
    )
    vessel_rows = []
    for (class_name, vessel_key), recordings in sorted(recordings_by_vessel.items()):
        row: dict[str, object] = {
            "class_name": class_name,
            "vessel_key": vessel_key,
            "recordings": len(recordings),
            "duration_seconds": sum(float(item["duration_seconds"]) for item in recordings),
            "selected_segments": sum(int(item["selected_segments"]) for item in recordings),
        }
        for policy in ("s0", "s1", "s2"):
            row[f"{policy}_probability"] = sum(
                float(item[f"{policy}_probability"]) for item in recordings
            )
            row[f"{policy}_expected_draws"] = epoch_samples * float(
                row[f"{policy}_probability"]
            )
        vessel_rows.append(row)

    class_summary = {}
    for class_name in class_names:
        recordings = recordings_by_class[class_name]
        vessels = [row for row in vessel_rows if row["class_name"] == class_name]
        class_s0_total = sum(float(item["s0_probability"]) for item in recordings)
        per_policy = {}
        for policy in ("s0", "s1", "s2"):
            recording_probabilities = [
                float(item[f"{policy}_probability"]) for item in recordings
            ]
            vessel_probabilities = [
                float(item[f"{policy}_probability"]) for item in vessels
            ]
            recording_total = sum(recording_probabilities)
            vessel_total = sum(vessel_probabilities)
            per_policy[policy] = {
                "effective_recordings_within_class": _effective_count(
                    value / recording_total for value in recording_probabilities
                ),
                "effective_vessels_within_class": _effective_count(
                    value / vessel_total for value in vessel_probabilities
                ),
            }
        class_summary[class_name] = {
            "recordings": len(recordings),
            "vessels": len(vessels),
            "selected_segments": sum(int(item["selected_segments"]) for item in recordings),
            "duration_hours": sum(float(item["duration_seconds"]) for item in recordings) / 3600.0,
            "recording_duration_seconds": _summary(
                float(item["duration_seconds"]) for item in recordings
            ),
            "recordings_shorter_than_clip": sum(
                bool(item["shorter_than_clip"]) for item in recordings
            ),
            "s0_top_10_recording_share_within_class": _top_share(
                (float(item["s0_probability"]) for item in recordings), 10
            )
            / class_s0_total,
            "s0_top_5_vessel_share_within_class": _top_share(
                (float(item["s0_probability"]) for item in vessels), 5
            )
            / class_s0_total,
            "policy_effective_counts": per_policy,
        }

    policy_summary = {}
    for policy in ("s0", "s1", "s2"):
        recording_probabilities = [float(item[f"{policy}_probability"]) for item in recording_rows]
        vessel_probabilities = [float(item[f"{policy}_probability"]) for item in vessel_rows]
        policy_summary[policy] = {
            "recording_probability_sum": sum(recording_probabilities),
            "vessel_probability_sum": sum(vessel_probabilities),
            "effective_recordings": _effective_count(recording_probabilities),
            "effective_vessels": _effective_count(vessel_probabilities),
            "recording_expected_draws": _summary(
                float(item[f"{policy}_expected_draws"]) for item in recording_rows
            ),
            "vessel_expected_draws": _summary(
                float(item[f"{policy}_expected_draws"]) for item in vessel_rows
            ),
        }

    return {
        "schema_version": 1,
        "protocol": manifest.get("protocol"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "split": "train",
        "class_names": class_names,
        "clip_duration_seconds": clip_duration_seconds,
        "epoch_samples": epoch_samples,
        "recordings": len(recording_rows),
        "vessels": len(vessel_rows),
        "selected_segments": len(train_segments),
        "class_summary": class_summary,
        "fixed_context_redundancy": {
            "unique_runtime_windows": sum(
                int(item["unique_runtime_windows"]) for item in recording_rows
            ),
            "exact_duplicate_windows": exact_duplicate_contexts,
            "adjacent_window_overlap_fraction": _summary(adjacent_overlap_fractions),
        },
        "policies": {
            "s0": "current fixed manifest-anchor exposure",
            "s1": "uniform class, then uniform recording, then dynamic crop",
            "s2": "uniform class, then uniform vessel, then uniform recording, then dynamic crop",
        },
        "policy_summary": policy_summary,
        "recording_rows": recording_rows,
        "vessel_rows": vessel_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sampling_audit(
    audit: dict[str, object],
    output_dir: str | Path,
    *,
    source_hashes: dict[str, str],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    recording_rows = list(audit["recording_rows"])
    vessel_rows = list(audit["vessel_rows"])
    payload = {
        **{
            key: value
            for key, value in audit.items()
            if key not in {"recording_rows", "vessel_rows"}
        },
        "source_sha256": source_hashes,
    }
    (output / "sampling_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output / "recording_sampling_exposure.csv", recording_rows)
    _write_csv(output / "vessel_sampling_exposure.csv", vessel_rows)
    redundancy = audit["fixed_context_redundancy"]
    policy_summary = audit["policy_summary"]
    assert isinstance(redundancy, dict) and isinstance(policy_summary, dict)
    overlap = redundancy["adjacent_window_overlap_fraction"]
    assert isinstance(overlap, dict)
    lines = [
        "# DeepShip training sampling audit",
        "",
        f"Protocol: `{audit['protocol']}`",
        f"Training recordings/vessels: {audit['recordings']}/{audit['vessels']}",
        f"Epoch sample budget: {audit['epoch_samples']}",
        f"Median overlap between adjacent unique {float(audit['clip_duration_seconds']):g}s "
        f"contexts: {float(overlap['median']):.1%}",
        f"Exact duplicate runtime contexts caused by boundary clamping: "
        f"{redundancy['exact_duplicate_windows']}",
        "",
        "| Policy | Effective recordings | Effective vessels |",
        "|---|---:|---:|",
    ]
    for policy in ("s0", "s1", "s2"):
        summary = policy_summary[policy]
        lines.append(
            f"| {policy.upper()} | {float(summary['effective_recordings']):.1f} | "
            f"{float(summary['effective_vessels']):.1f} |"
        )
    lines.extend(
        [
            "",
            "S1 and S2 values are theoretical exposure probabilities for a fixed sample budget. "
            "They do not modify the frozen train/validation/test group assignment.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
