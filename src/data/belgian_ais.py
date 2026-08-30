"""Audited Belgian AIS metadata, date-disjoint folds, and Mel datasets."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import soundfile as sf
import torch
import torch.nn.functional as nnf
import torchaudio
from torch.utils.data import Dataset, Sampler

from src.data.deepship import CLASS_NAMES, CLASS_TO_INDEX
from src.utils.pathing import (
    resolve_manifest_path,
    resolve_path,
    validate_manifest_relative_path,
)


BELGIAN_CLASS_MAP = {
    "Cargo": "Cargo",
    "Tanker": "Tank",
    "Passenger": "Passenger",
    "Tug": "Tug",
    "Towing": "Tug",
    "Large-Towing": "Tug",
}
OFFICIAL_SPLITS = ("train", "val", "test")
TARGET_SAMPLE_RATE = 16_000
PROTOCOL_SCHEMA_VERSION = 1
STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION = 2
STRICT_AUDIO_POLICY = {
    "source_sample_rate_hz": 48_000,
    "duration_seconds": 10.0,
    "duration_rule": "frames == source_sample_rate_hz * duration_seconds",
    "accepted_source_channels": [1, 2],
    "channel_policy": "fixed_channel_0",
    "selected_channel_index": 0,
}


@dataclass(frozen=True)
class BelgianRecord:
    relative_path: str
    class_name: str
    label_index: int
    vessel_type: str
    official_split: str
    event_time: str
    calendar_date: str
    station: str
    distance_km: float
    activity: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def map_belgian_vessel_type(vessel_type: str) -> str | None:
    return BELGIAN_CLASS_MAP.get(vessel_type.strip())


def parse_utc_date(raw_event_time: str) -> str:
    text = raw_event_time.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"event_time must include a timezone: {raw_event_time!r}")
    return parsed.astimezone(timezone.utc).date().isoformat()


def load_official_split_paths(
    split_dir: str | Path,
) -> tuple[dict[str, str], dict[str, object]]:
    root = resolve_path(split_dir)
    path_splits: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = {}
    for split in OFFICIAL_SPLITS:
        source = root / f"{split}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing official split file: {source}")
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
        relative_paths = [line.split(maxsplit=1)[0] for line in lines if line]
        counts[split] = len(relative_paths)
        for raw_path in relative_paths:
            relative_path = validate_manifest_relative_path(raw_path).as_posix()
            path_splits[relative_path].append(split)
    path_to_split: dict[str, str] = {}
    same_split_duplicate_lines = 0
    cross_split_paths: list[str] = []
    for relative_path, appearances in sorted(path_splits.items()):
        unique_splits = set(appearances)
        same_split_duplicate_lines += len(appearances) - len(unique_splits)
        if len(unique_splits) != 1:
            cross_split_paths.append(relative_path)
            continue
        path_to_split[relative_path] = next(iter(unique_splits))
    return path_to_split, {
        "line_counts": counts,
        "listed_unique_paths": len(path_splits),
        "unique_paths": len(path_to_split),
        "same_split_duplicate_lines": same_split_duplicate_lines,
        "cross_split_paths_count": len(cross_split_paths),
        "cross_split_paths_examples": cross_split_paths[:20],
        "cross_split_policy": "exclude from all protocol splits",
    }


def _supervision_signature(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        row.get(field, "").strip()
        for field in ("vessel_type", "event_time", "station", "distance", "activity")
    )


def load_belgian_records(
    metadata_csv: str | Path,
    split_dir: str | Path,
    *,
    max_distance_km: float = 5.0,
) -> tuple[list[BelgianRecord], dict[str, object]]:
    """Load the official intersection and exclude conflicting duplicate supervision."""

    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")
    metadata_path = resolve_path(metadata_csv)
    path_to_split, split_audit = load_official_split_paths(split_dir)
    required = {
        "file_location",
        "vessel_type",
        "activity",
        "distance",
        "event_time",
        "station",
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata_rows = 0
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Belgian metadata is missing columns: {sorted(missing)}")
        for row in reader:
            metadata_rows += 1
            raw_path = row.get("file_location", "")
            relative_path = validate_manifest_relative_path(raw_path).as_posix()
            grouped[relative_path].append(dict(row))

    records: list[BelgianRecord] = []
    duplicate_paths = 0
    conflicting_duplicate_paths: list[str] = []
    unassigned_metadata_paths = 0
    excluded_non_target = Counter()
    excluded_distance = Counter()
    excluded_missing = Counter()
    for relative_path, rows in sorted(grouped.items()):
        if len(rows) > 1:
            duplicate_paths += 1
        signatures = {_supervision_signature(row) for row in rows}
        if len(signatures) > 1:
            conflicting_duplicate_paths.append(relative_path)
            continue
        official_split = path_to_split.get(relative_path)
        if official_split is None:
            unassigned_metadata_paths += 1
            continue
        row = rows[0]
        vessel_type = row["vessel_type"].strip()
        class_name = map_belgian_vessel_type(vessel_type)
        if class_name is None:
            excluded_non_target[vessel_type or "<missing>"] += 1
            continue
        try:
            distance_km = float(row["distance"])
        except (TypeError, ValueError):
            excluded_missing["distance"] += 1
            continue
        if not (distance_km >= 0.0 and distance_km <= max_distance_km):
            excluded_distance[class_name] += 1
            continue
        try:
            calendar_date = parse_utc_date(row["event_time"])
        except (TypeError, ValueError):
            excluded_missing["event_time"] += 1
            continue
        station = row["station"].strip()
        if not station:
            excluded_missing["station"] += 1
            continue
        records.append(
            BelgianRecord(
                relative_path=relative_path,
                class_name=class_name,
                label_index=CLASS_TO_INDEX[class_name],
                vessel_type=vessel_type,
                official_split=official_split,
                event_time=row["event_time"].strip(),
                calendar_date=calendar_date,
                station=station,
                distance_km=distance_km,
                activity=row["activity"].strip(),
            )
        )

    metadata_paths = set(grouped)
    missing_metadata_paths = sorted(set(path_to_split).difference(metadata_paths))
    audit = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_rows": metadata_rows,
        "metadata_unique_paths": len(grouped),
        "metadata_duplicate_paths": duplicate_paths,
        "conflicting_duplicate_paths_count": len(conflicting_duplicate_paths),
        "conflicting_duplicate_paths_examples": conflicting_duplicate_paths[:20],
        "unassigned_metadata_paths": unassigned_metadata_paths,
        "official_paths_missing_metadata_count": len(missing_metadata_paths),
        "official_paths_missing_metadata_examples": missing_metadata_paths[:20],
        "official_split": split_audit,
        "filters": {
            "class_map": BELGIAN_CLASS_MAP,
            "max_distance_km": max_distance_km,
            "duplicate_policy": "exclude paths with conflicting supervision; keep one identical row",
        },
        "excluded_non_target": dict(sorted(excluded_non_target.items())),
        "excluded_distance_by_class": {
            name: excluded_distance.get(name, 0) for name in CLASS_NAMES
        },
        "excluded_missing": dict(sorted(excluded_missing.items())),
        "eligible_records": len(records),
        "eligible_by_official_split_and_class": summarize_by_split_class(records),
        "eligible_unique_dates_by_split": {
            split: len({record.calendar_date for record in records if record.official_split == split})
            for split in OFFICIAL_SPLITS
        },
        "eligible_stations": dict(Counter(record.station for record in records)),
    }
    return records, audit


def summarize_by_split_class(records: Iterable[BelgianRecord]) -> dict[str, dict[str, int]]:
    counts = Counter((record.official_split, record.class_name) for record in records)
    return {
        split: {name: counts[(split, name)] for name in CLASS_NAMES}
        for split in OFFICIAL_SPLITS
    }


def record_from_dict(data: Mapping[str, object]) -> BelgianRecord:
    relative_path = validate_manifest_relative_path(str(data["relative_path"])).as_posix()
    class_name = str(data["class_name"])
    if class_name not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown Belgian class: {class_name}")
    label_index = int(data["label_index"])
    if label_index != CLASS_TO_INDEX[class_name]:
        raise ValueError("Belgian class_name and label_index disagree")
    official_split = str(data["official_split"])
    if official_split not in OFFICIAL_SPLITS:
        raise ValueError(f"Unknown official split: {official_split}")
    calendar_date = str(data["calendar_date"])
    if parse_utc_date(str(data["event_time"])) != calendar_date:
        raise ValueError("Belgian event_time and calendar_date disagree")
    return BelgianRecord(
        relative_path=relative_path,
        class_name=class_name,
        label_index=label_index,
        vessel_type=str(data["vessel_type"]),
        official_split=official_split,
        event_time=str(data["event_time"]),
        calendar_date=calendar_date,
        station=str(data["station"]),
        distance_km=float(data["distance_km"]),
        activity=str(data.get("activity", "")),
    )


def _fold_objective(
    fold_counts: list[Counter[str]],
    fold_sizes: list[int],
    totals: Counter[str],
) -> float:
    target_size = sum(fold_sizes) / len(fold_sizes)
    score = sum(((size - target_size) / max(1.0, target_size)) ** 2 for size in fold_sizes)
    for class_name in CLASS_NAMES:
        target = totals[class_name] / len(fold_counts)
        score += 4.0 * sum(
            ((counts[class_name] - target) / max(1.0, target)) ** 2
            for counts in fold_counts
        )
    return score


def assign_date_folds(
    records: Sequence[BelgianRecord],
    *,
    folds: int = 3,
    seed: int = 42,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    by_date: dict[str, list[BelgianRecord]] = defaultdict(list)
    for record in records:
        by_date[record.calendar_date].append(record)
    if len(by_date) < folds:
        raise ValueError("Not enough calendar dates for the requested folds")
    totals = Counter(record.class_name for record in records)
    date_counts = {
        date: Counter(record.class_name for record in rows) for date, rows in by_date.items()
    }
    rng = random.Random(seed)
    jitter = {date: rng.random() for date in by_date}
    ordered_dates = sorted(
        by_date,
        key=lambda date: (
            -sum(date_counts[date][name] / max(1, totals[name]) for name in CLASS_NAMES),
            -len(by_date[date]),
            jitter[date],
            date,
        ),
    )
    assignments: dict[str, int] = {}
    fold_counts = [Counter() for _ in range(folds)]
    fold_sizes = [0 for _ in range(folds)]
    for date in ordered_dates:
        candidates: list[tuple[float, int, int]] = []
        for fold in range(folds):
            candidate_counts = [Counter(counts) for counts in fold_counts]
            candidate_sizes = list(fold_sizes)
            candidate_counts[fold].update(date_counts[date])
            candidate_sizes[fold] += len(by_date[date])
            candidates.append(
                (_fold_objective(candidate_counts, candidate_sizes, totals), fold_sizes[fold], fold)
            )
        _, _, selected = min(candidates)
        assignments[date] = selected
        fold_counts[selected].update(date_counts[date])
        fold_sizes[selected] += len(by_date[date])

    missing = {
        fold: [name for name in CLASS_NAMES if fold_counts[fold][name] == 0]
        for fold in range(folds)
    }
    missing = {fold: names for fold, names in missing.items() if names}
    if missing:
        raise ValueError(f"Date-disjoint folds are missing target classes: {missing}")
    return assignments


def build_fold_manifests(
    records: Sequence[BelgianRecord],
    audit: Mapping[str, object],
    *,
    folds: int = 3,
    fold_seed: int = 42,
    audio_audit: Mapping[str, object] | None = None,
    frozen_date_assignments: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sealed_test = [record for record in records if record.official_split == "test"]
    sealed_test_dates = {record.calendar_date for record in sealed_test}
    development_candidates = [
        record for record in records if record.official_split in {"train", "val"}
    ]
    excluded_shared_test_date = [
        record for record in development_candidates if record.calendar_date in sealed_test_dates
    ]
    development = [
        record for record in development_candidates if record.calendar_date not in sealed_test_dates
    ]
    if frozen_date_assignments is None:
        assignments = assign_date_folds(development, folds=folds, seed=fold_seed)
        assignment_policy = "deterministic_balancing_search"
    else:
        assignments = {str(date): int(fold) for date, fold in frozen_date_assignments.items()}
        development_dates = {record.calendar_date for record in development}
        missing_dates = sorted(development_dates.difference(assignments))
        invalid_folds = sorted(
            {fold for date, fold in assignments.items() if date in development_dates and fold not in range(folds)}
        )
        if missing_dates or invalid_folds:
            raise ValueError(
                "Frozen date assignments do not cover the strict development pool: "
                f"missing_dates={missing_dates}, invalid_folds={invalid_folds}"
            )
        assignment_policy = "preserved_from_pre_audio_audit_frozen_manifests"
    manifests: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for fold in range(folds):
        rows = []
        for record in development:
            split = "val" if assignments[record.calendar_date] == fold else "train"
            rows.append({**asdict(record), "split": split})
        body: dict[str, object] = {
            "schema_version": (
                STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION
                if audio_audit is not None
                else PROTOCOL_SCHEMA_VERSION
            ),
            "experiment_id": "belgian_attention_v1",
            "protocol": "utc_date_disjoint",
            "fold": fold + 1,
            "fold_seed": fold_seed,
            "source_metadata_sha256": audit["metadata_sha256"],
            "filters": audit["filters"],
            "test_policy": "sealed; test records are not present in development manifests",
            "records": sorted(rows, key=lambda row: (str(row["split"]), str(row["relative_path"]))),
        }
        if audio_audit is not None:
            body["audio_policy"] = dict(STRICT_AUDIO_POLICY)
            body["development_audio_inventory_sha256"] = audio_audit[
                "admitted_inventory_sha256"
            ]
        body["manifest_sha256"] = canonical_sha256(body)
        manifests.append(body)
        counts = Counter((str(row["split"]), str(row["class_name"])) for row in rows)
        train_dates = {str(row["calendar_date"]) for row in rows if row["split"] == "train"}
        val_dates = {str(row["calendar_date"]) for row in rows if row["split"] == "val"}
        fold_summaries.append(
            {
                "fold": fold + 1,
                "manifest_sha256": body["manifest_sha256"],
                "counts": {
                    split: {name: counts[(split, name)] for name in CLASS_NAMES}
                    for split in ("train", "val")
                },
                "train_dates": len(train_dates),
                "val_dates": len(val_dates),
                "date_overlap": sorted(train_dates.intersection(val_dates)),
            }
        )
    sealed_payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": "belgian_attention_v1",
        "policy": "sealed_official_test",
        "source_metadata_sha256": audit["metadata_sha256"],
        "records": [asdict(record) for record in sorted(sealed_test, key=lambda item: item.relative_path)],
    }
    sealed_payload["manifest_sha256"] = canonical_sha256(sealed_payload)
    index = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": "belgian_attention_v1",
        "fold_seed": fold_seed,
        "date_assignment_policy": assignment_policy,
        "folds": fold_summaries,
        "sealed_test_dates": sorted(sealed_test_dates),
        "development_records_excluded_for_test_date_isolation": len(
            excluded_shared_test_date
        ),
        "development_dates_excluded_for_test_date_isolation": sorted(
            {record.calendar_date for record in excluded_shared_test_date}
        ),
        "sealed_test_manifest": sealed_payload,
    }
    if audio_audit is not None:
        index["development_audio_policy"] = dict(STRICT_AUDIO_POLICY)
        index["development_audio_inventory_sha256"] = audio_audit[
            "admitted_inventory_sha256"
        ]
        index["sealed_test_audio_status"] = "not_inspected"
    return manifests, index


def filter_strict_development_audio(
    records: Sequence[BelgianRecord],
    *,
    data_root: str | Path,
) -> tuple[list[BelgianRecord], dict[str, object]]:
    """Admit exact 10 s mono/stereo development files without reading sealed test audio."""

    root = resolve_path(data_root)
    sealed_test = [record for record in records if record.official_split == "test"]
    sealed_test_dates = {record.calendar_date for record in sealed_test}
    shared_test_date_records = [
        record
        for record in records
        if record.official_split in {"train", "val"}
        and record.calendar_date in sealed_test_dates
    ]
    development = [
        record
        for record in records
        if record.official_split in {"train", "val"}
        and record.calendar_date not in sealed_test_dates
    ]
    admitted_paths: set[str] = set()
    inventory: list[dict[str, object]] = []
    rejected = Counter()
    rejected_examples: list[dict[str, object]] = []
    for record in development:
        path = resolve_manifest_path(root, record.relative_path)
        reason = ""
        details: dict[str, object] = {"relative_path": record.relative_path}
        if not path.is_file():
            reason = "missing"
        else:
            info = sf.info(str(path))
            details.update(
                {
                    "source_sample_rate_hz": int(info.samplerate),
                    "source_channels": int(info.channels),
                    "source_frames": int(info.frames),
                }
            )
            if info.samplerate != STRICT_AUDIO_POLICY["source_sample_rate_hz"]:
                reason = "sample_rate"
            elif info.channels not in STRICT_AUDIO_POLICY["accepted_source_channels"]:
                reason = "channels"
            elif info.frames != int(info.samplerate * STRICT_AUDIO_POLICY["duration_seconds"]):
                reason = "not_exact_10_seconds"
        if reason:
            rejected[reason] += 1
            if len(rejected_examples) < 50:
                rejected_examples.append({**details, "reason": reason})
            continue
        admitted_paths.add(record.relative_path)
        inventory.append(
            {
                **details,
                "selected_channel_index": STRICT_AUDIO_POLICY["selected_channel_index"],
            }
        )
    inventory.sort(key=lambda row: str(row["relative_path"]))
    inventory_payload = {
        "audio_policy": dict(STRICT_AUDIO_POLICY),
        "records": inventory,
    }
    filtered = [
        record
        for record in records
        if record.official_split == "test"
        or record.calendar_date in sealed_test_dates
        or record.relative_path in admitted_paths
    ]
    admitted_records = [
        record
        for record in filtered
        if record.official_split != "test" and record.calendar_date not in sealed_test_dates
    ]
    report = {
        "schema_version": STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION,
        "policy": dict(STRICT_AUDIO_POLICY),
        "development_candidates": len(development),
        "development_admitted": len(admitted_records),
        "development_rejected": len(development) - len(admitted_records),
        "retention_fraction": len(admitted_records) / max(1, len(development)),
        "admitted_by_class": dict(Counter(record.class_name for record in admitted_records)),
        "admitted_by_channels": dict(
            Counter(str(row["source_channels"]) for row in inventory)
        ),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "rejected_examples": rejected_examples,
        "sealed_test_records_preserved": len(sealed_test),
        "development_records_skipped_for_test_date_isolation": len(shared_test_date_records),
        "sealed_test_audio_status": "not_inspected",
        "admitted_inventory_sha256": canonical_sha256(inventory_payload),
        "inventory": inventory,
    }
    return filtered, report


def validate_fold_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    body = dict(manifest)
    recorded_hash = str(body.pop("manifest_sha256", ""))
    calculated_hash = canonical_sha256(body)
    records = [dict(row) for row in body.get("records", [])]  # type: ignore[arg-type]
    train_dates = {str(row["calendar_date"]) for row in records if row["split"] == "train"}
    val_dates = {str(row["calendar_date"]) for row in records if row["split"] == "val"}
    paths = [str(row["relative_path"]) for row in records]
    counts = Counter((str(row["split"]), str(row["class_name"])) for row in records)
    schema_version = int(body.get("schema_version", 0))
    checks = {
        "schema_version_supported": schema_version in {
            PROTOCOL_SCHEMA_VERSION,
            STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION,
        },
        "manifest_hash_matches": bool(recorded_hash) and recorded_hash == calculated_hash,
        "paths_unique": len(paths) == len(set(paths)),
        "test_absent": all(str(row.get("official_split")) != "test" for row in records),
        "dates_disjoint": not train_dates.intersection(val_dates),
        "all_classes_present": all(
            counts[(split, name)] > 0 for split in ("train", "val") for name in CLASS_NAMES
        ),
    }
    if schema_version == STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION:
        inventory_hash = str(body.get("development_audio_inventory_sha256", ""))
        checks["strict_audio_policy_matches"] = body.get("audio_policy") == STRICT_AUDIO_POLICY
        checks["audio_inventory_hash_present"] = (
            len(inventory_hash) == 64
            and all(character in "0123456789abcdef" for character in inventory_hash)
        )
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "calculated_manifest_sha256": calculated_hash,
        "recorded_manifest_sha256": recorded_hash,
        "date_overlap": sorted(train_dates.intersection(val_dates)),
    }


class ClassDateBalancedEpochSampler(Sampler[int]):
    """Equal class quotas with round-robin exposure across UTC dates."""

    def __init__(
        self,
        records: Sequence[BelgianRecord],
        *,
        seed: int = 42,
        samples_per_class: int | None = None,
    ) -> None:
        self.records = records
        self.seed = int(seed)
        self.epoch = 0
        by_class = Counter(record.class_name for record in records)
        missing = [name for name in CLASS_NAMES if by_class[name] == 0]
        if missing:
            raise ValueError(f"Sampler is missing target classes: {missing}")
        maximum_balanced_quota = min(by_class[name] for name in CLASS_NAMES)
        self.samples_per_class = (
            maximum_balanced_quota if samples_per_class is None else int(samples_per_class)
        )
        if self.samples_per_class <= 0 or self.samples_per_class > maximum_balanced_quota:
            raise ValueError(
                "samples_per_class must be positive and no greater than the rarest class count"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(CLASS_NAMES) * self.samples_per_class

    def _sample_class(self, class_name: str, rng: random.Random) -> list[int]:
        by_date: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            if record.class_name == class_name:
                by_date[record.calendar_date].append(index)
        dates = sorted(by_date)
        rng.shuffle(dates)
        queues: dict[str, deque[int]] = {}
        for date in dates:
            indexes = list(by_date[date])
            rng.shuffle(indexes)
            queues[date] = deque(indexes)
        selected: list[int] = []
        while len(selected) < self.samples_per_class:
            progressed = False
            for date in dates:
                if queues[date]:
                    selected.append(queues[date].popleft())
                    progressed = True
                    if len(selected) == self.samples_per_class:
                        break
            if not progressed:
                raise RuntimeError("Sampler exhausted a class before reaching its quota")
        return selected

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        selected: list[int] = []
        for class_name in CLASS_NAMES:
            selected.extend(self._sample_class(class_name, rng))
        rng.shuffle(selected)
        return iter(selected)

    def audit(self) -> dict[str, object]:
        indexes = list(iter(self))
        selected = [self.records[index] for index in indexes]
        unique_files = len({record.relative_path for record in selected})

        def distance_bin(distance: float) -> str:
            if distance <= 1.0:
                return "0-1"
            if distance <= 2.0:
                return "1-2"
            if distance <= 3.0:
                return "2-3"
            return "3-5"

        return {
            "epoch": self.epoch,
            "samples": len(selected),
            "unique_files": unique_files,
            "duplicate_draws": len(selected) - unique_files,
            "by_class": dict(Counter(record.class_name for record in selected)),
            "unique_dates_by_class": {
                name: len({record.calendar_date for record in selected if record.class_name == name})
                for name in CLASS_NAMES
            },
            "by_station": dict(Counter(record.station for record in selected)),
            "by_distance_km": dict(
                Counter(distance_bin(record.distance_km) for record in selected)
            ),
        }


class BelgianMelDataset(Dataset):
    def __init__(
        self,
        records: Sequence[BelgianRecord],
        *,
        data_root: str | Path,
        sample_rate: int = TARGET_SAMPLE_RATE,
        clip_duration: float = 10.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 64,
        source_sample_rate: int = 48_000,
        channel_policy: str = "fixed_channel_0",
        require_exact_source_duration: bool = True,
        return_index: bool = False,
    ) -> None:
        self.records = list(records)
        self.data_root = resolve_path(data_root)
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(round(sample_rate * clip_duration))
        self.source_sample_rate = int(source_sample_rate)
        self.source_clip_samples = int(round(source_sample_rate * clip_duration))
        if channel_policy != "fixed_channel_0":
            raise ValueError("Belgian channel_policy must be fixed_channel_0")
        self.channel_policy = channel_policy
        self.require_exact_source_duration = bool(require_exact_source_duration)
        self.return_index = return_index
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = resolve_manifest_path(self.data_root, record.relative_path)
        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if source_rate != self.source_sample_rate:
            raise ValueError(
                f"Unexpected source sample rate for {record.relative_path}: {source_rate}"
            )
        if audio.shape[1] not in {1, 2}:
            raise ValueError(
                f"Unexpected source channels for {record.relative_path}: {audio.shape[1]}"
            )
        if self.require_exact_source_duration and audio.shape[0] != self.source_clip_samples:
            raise ValueError(
                f"Source is not exactly the frozen duration for {record.relative_path}: "
                f"{audio.shape[0]} frames"
            )
        waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0)
        if source_rate != self.sample_rate:
            resampler = self.resamplers.get(source_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(source_rate, self.sample_rate)
                self.resamplers[source_rate] = resampler
            waveform = resampler(waveform)
        if waveform.size(-1) > self.clip_samples:
            waveform = waveform[..., : self.clip_samples]
        elif waveform.size(-1) < self.clip_samples:
            waveform = nnf.pad(waveform, (0, self.clip_samples - waveform.size(-1)))
        mel = self.to_db(self.mel_transform(waveform))
        if self.return_index:
            return mel, record.label_index, index
        return mel, record.label_index


def audit_audio_files(
    records: Sequence[BelgianRecord],
    *,
    data_root: str | Path,
    expected_sample_rate: int = 48_000,
    expected_duration_seconds: float = 10.0,
) -> dict[str, object]:
    missing: list[str] = []
    mismatches: list[dict[str, object]] = []
    for record in records:
        path = resolve_manifest_path(data_root, record.relative_path)
        if not path.is_file():
            missing.append(record.relative_path)
            continue
        info = sf.info(str(path))
        if (
            info.samplerate != expected_sample_rate
            or info.channels not in {1, 2}
            or info.frames != int(info.samplerate * expected_duration_seconds)
        ):
            mismatches.append(
                {
                    "relative_path": record.relative_path,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "duration_seconds": info.frames / info.samplerate,
                }
            )
    return {
        "status": "passed" if not missing and not mismatches else "failed",
        "files": len(records),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "metadata_mismatch_count": len(mismatches),
        "metadata_mismatch_examples": mismatches[:20],
        "audio_policy": dict(STRICT_AUDIO_POLICY),
    }
