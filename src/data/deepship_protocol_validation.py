from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import soundfile as sf

from src.data.deepship_audit import CLASS_NAMES, stable_json_hash
from src.data.deepship_protocols import SPLITS
from src.utils.pathing import resolve_manifest_path, validate_manifest_relative_path


def load_split_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_protocol_manifest(
    manifest: dict[str, object],
    config: dict[str, object],
    audit_report: dict[str, object],
    *,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    calculated_hash = stable_json_hash(manifest_core)
    segments = manifest.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Protocol manifest segments must be a list")

    invalid_paths: list[str] = []
    segment_ids: list[str] = []
    recording_partitions: dict[str, set[str]] = defaultdict(set)
    group_partitions: dict[str, set[str]] = defaultdict(set)
    selected_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    invalid_segment_bounds: list[str] = []
    missing_files: list[str] = []
    audio_metadata_mismatches: list[str] = []
    audio_info_cache: dict[str, object] = {}

    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Every manifest segment must be an object")
        relative_path = str(segment.get("relative_path", ""))
        try:
            validate_manifest_relative_path(relative_path)
        except ValueError:
            invalid_paths.append(relative_path)
            continue
        split = str(segment.get("split", ""))
        class_name = str(segment.get("class_name", ""))
        if split in selected_counts:
            selected_counts[split][class_name] += 1
        segment_id = f"{relative_path}#{int(segment['segment_index']):06d}"
        segment_ids.append(segment_id)
        recording_partitions[relative_path].add(split)
        group_partitions[str(segment.get("group_key", ""))].add(split)

        start_frame = int(segment["start_frame"])
        num_frames = int(segment["num_frames"])
        if start_frame < 0 or num_frames <= 0:
            invalid_segment_bounds.append(segment_id)
        if data_root is None:
            continue
        if relative_path not in audio_info_cache:
            audio_path = resolve_manifest_path(data_root, relative_path)
            if not audio_path.is_file():
                missing_files.append(relative_path)
                audio_info_cache[relative_path] = None
            else:
                audio_info_cache[relative_path] = sf.info(str(audio_path))
        info = audio_info_cache[relative_path]
        if info is None:
            continue
        if int(segment["sample_rate"]) != info.samplerate:
            audio_metadata_mismatches.append(segment_id)
        if start_frame + num_frames > info.frames:
            invalid_segment_bounds.append(segment_id)

    targets = config["split"]["target_segments_per_class"]  # type: ignore[index]
    exact_budget = all(
        selected_counts[split][class_name] == int(targets[split])
        for split in SPLITS
        for class_name in CLASS_NAMES
    )
    recording_overlap = sorted(
        key for key, partitions in recording_partitions.items() if len(partitions) > 1
    )
    group_overlap = sorted(
        key for key, partitions in group_partitions.items() if len(partitions) > 1
    )
    protocol = str(manifest.get("protocol", ""))
    checks = {
        "schema_version_matches": manifest.get("schema_version") == 1,
        "experiment_id_matches": manifest.get("experiment_id") == config.get("experiment_id"),
        "split_seed_matches": manifest.get("split_seed") == config["split"]["split_seed"],  # type: ignore[index]
        "target_budget_matches": manifest.get("target_segments_per_class") == targets,
        "manifest_hash_matches": manifest.get("manifest_sha256") == calculated_hash,
        "source_inventory_hash_matches": (
            manifest.get("source_inventory_sha256") == audit_report.get("dataset_inventory_sha256")
        ),
        "source_identity_hash_matches": (
            manifest.get("source_identity_sha256")
            == audit_report.get("recording_identity_manifest_sha256")
        ),
        "paths_are_portable": not invalid_paths,
        "segment_ids_unique": len(segment_ids) == len(set(segment_ids)),
        "exact_target_segments_per_class": exact_budget,
        "group_keys_are_disjoint": not group_overlap,
        "recordings_are_disjoint": not recording_overlap,
        "source_files_exist": not missing_files,
        "audio_metadata_matches": not audio_metadata_mismatches,
        "segment_bounds_valid": not invalid_segment_bounds,
    }
    required = [
        "schema_version_matches",
        "experiment_id_matches",
        "split_seed_matches",
        "target_budget_matches",
        "manifest_hash_matches",
        "source_inventory_hash_matches",
        "source_identity_hash_matches",
        "paths_are_portable",
        "segment_ids_unique",
        "exact_target_segments_per_class",
        "group_keys_are_disjoint",
        "source_files_exist",
        "audio_metadata_matches",
        "segment_bounds_valid",
    ]
    if protocol in {"recording_disjoint", "vessel_name_disjoint"}:
        required.append("recordings_are_disjoint")
    status = "passed" if all(checks[name] for name in required) else "failed"

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol": protocol,
        "status": status,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "calculated_manifest_sha256": calculated_hash,
        "segments": len(segments),
        "recordings": len(recording_partitions),
        "groups": len(group_partitions),
        "selected_segments_by_split_and_class": {
            split: {class_name: selected_counts[split][class_name] for class_name in CLASS_NAMES}
            for split in SPLITS
        },
        "checks": checks,
        "required_checks": required,
        "diagnostics": {
            "invalid_paths_count": len(invalid_paths),
            "invalid_paths_examples": invalid_paths[:20],
            "recording_overlap_count": len(recording_overlap),
            "recording_overlap_examples": recording_overlap[:20],
            "group_overlap_count": len(group_overlap),
            "group_overlap_examples": group_overlap[:20],
            "missing_files_count": len(missing_files),
            "missing_files_examples": missing_files[:20],
            "audio_metadata_mismatch_count": len(audio_metadata_mismatches),
            "audio_metadata_mismatch_examples": audio_metadata_mismatches[:20],
            "invalid_segment_bounds_count": len(invalid_segment_bounds),
            "invalid_segment_bounds_examples": invalid_segment_bounds[:20],
        },
    }
