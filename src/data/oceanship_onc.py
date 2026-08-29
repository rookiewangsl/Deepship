from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


TARGET_LABELS = ("Cargo", "Passenger", "Tanker", "Tug")
DEFAULT_DEVICE_CODES = ("ICLISTENAF2523", "ICLISTENAF2556")

_FG_WAV_PATTERN = re.compile(
    r"^(?P<event_timestamp>\d{8}T\d{6}\.\d{3}Z)_"
    r"(?P<event_index>\d+)_id_(?P<scenario_id>\d+)_"
    r"typecargo_(?P<type_code>\d+)(?:_(?P<chunk_index>\d+))?\.pt$"
)
_ONC_TOKEN_URL_PATTERN = re.compile(r"(?i)(token=)[^&\s]+")
_ONC_TOKEN_FIELD_PATTERN = re.compile(
    r"(?i)(['\"]?token['\"]?\s*[:=]\s*['\"]?)[^,\s'\"]+"
)


def _parse_utc(value: str, *, compact: bool) -> datetime:
    pattern = "%Y%m%dT%H%M%S.%fZ" if compact else "%Y-%m-%dT%H:%M:%S.%fZ"
    return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)


def _format_onc_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalise_mmsi(value: str) -> str:
    value = value.strip()
    if value.endswith(".0"):
        value = value[:-2]
    if not value.isdigit():
        raise ValueError(f"Invalid MMSI: {value!r}")
    return value


def redact_onc_error(value: object) -> str:
    """Remove API tokens from ONC exception text before it reaches logs."""
    text = _ONC_TOKEN_URL_PATTERN.sub(r"\1<redacted>", str(value))
    return _ONC_TOKEN_FIELD_PATTERN.sub(r"\1<redacted>", text)


@dataclass(frozen=True)
class OceanshipFGRecord:
    source_split: str
    wav_path: str
    label: str
    mmsi: str
    event_timestamp_utc: str
    ais_timestamp_utc: str
    event_index: int
    scenario_id: int
    type_code: int
    chunk_index: int | None

    @property
    def event_key(self) -> str:
        return (
            f"{self.event_timestamp_utc}:{self.event_index}:"
            f"{self.scenario_id}:{self.type_code}:{self.mmsi}"
        )


def parse_fg_row(row: dict[str, str], source_split: str) -> OceanshipFGRecord:
    missing = {"wav_path", "label", "mmsi", "ais_timestamp"} - set(row)
    if missing:
        raise ValueError(f"FG metadata is missing columns: {sorted(missing)}")

    wav_name = Path(row["wav_path"]).name
    match = _FG_WAV_PATTERN.fullmatch(wav_name)
    if match is None:
        raise ValueError(f"Unrecognised Oceanship-FG wav_path: {row['wav_path']!r}")

    event_timestamp = match.group("event_timestamp")
    _parse_utc(event_timestamp, compact=True)
    ais_timestamp = row["ais_timestamp"].strip()
    _parse_utc(ais_timestamp, compact=True)

    chunk = match.group("chunk_index")
    return OceanshipFGRecord(
        source_split=source_split,
        wav_path=row["wav_path"].strip(),
        label=row["label"].strip(),
        mmsi=_normalise_mmsi(str(row["mmsi"])),
        event_timestamp_utc=event_timestamp,
        ais_timestamp_utc=ais_timestamp,
        event_index=int(match.group("event_index")),
        scenario_id=int(match.group("scenario_id")),
        type_code=int(match.group("type_code")),
        chunk_index=None if chunk is None else int(chunk),
    )


def load_fg_metadata(paths: Sequence[str | Path]) -> list[OceanshipFGRecord]:
    records: list[OceanshipFGRecord] = []
    for path_value in paths:
        path = Path(path_value)
        source_split = "train" if "train" in path.stem.lower() else "test"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(parse_fg_row(row, source_split))
    return records


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit_fg_metadata(records: Sequence[OceanshipFGRecord]) -> dict[str, object]:
    target = [record for record in records if record.label in TARGET_LABELS]
    per_class: dict[str, dict[str, int]] = {}
    for label in TARGET_LABELS:
        class_records = [record for record in target if record.label == label]
        per_class[label] = {
            "rows": len(class_records),
            "events": len({record.event_key for record in class_records}),
            "mmsi": len({record.mmsi for record in class_records}),
            "days": len({record.event_timestamp_utc[:8] for record in class_records}),
        }

    labels_by_mmsi: dict[str, set[str]] = {}
    splits_by_mmsi: dict[str, set[str]] = {}
    deltas: list[float] = []
    for record in target:
        labels_by_mmsi.setdefault(record.mmsi, set()).add(record.label)
        splits_by_mmsi.setdefault(record.mmsi, set()).add(record.source_split)
        event_time = _parse_utc(record.event_timestamp_utc, compact=True)
        ais_time = _parse_utc(record.ais_timestamp_utc, compact=True)
        deltas.append((ais_time - event_time).total_seconds())

    conflicts = sorted(mmsi for mmsi, labels in labels_by_mmsi.items() if len(labels) > 1)
    split_overlap = sorted(mmsi for mmsi, splits in splits_by_mmsi.items() if len(splits) > 1)
    checks = {
        "all_target_classes_present": all(per_class[label]["rows"] > 0 for label in TARGET_LABELS),
        "target_mmsi_are_label_consistent": not conflicts,
        "event_timestamps_parsed": len(target) > 0,
        "official_split_is_mmsi_disjoint": not split_overlap,
    }
    return {
        "status": "metadata_audit_complete" if all(list(checks.values())[:3]) else "failed",
        "rows_all_labels": len(records),
        "rows_target_four_classes": len(target),
        "events_target_four_classes": len({record.event_key for record in target}),
        "mmsi_target_four_classes": len(labels_by_mmsi),
        "per_class": per_class,
        "mmsi_label_conflicts": conflicts,
        "official_train_test_mmsi_overlap_count": len(split_overlap),
        "official_train_test_mmsi_overlap_examples": split_overlap[:20],
        "event_to_ais_delta_seconds": {
            "minimum": min(deltas) if deltas else None,
            "p25": _quantile(deltas, 0.25),
            "median": _quantile(deltas, 0.5),
            "p75": _quantile(deltas, 0.75),
            "maximum": max(deltas) if deltas else None,
        },
        "timestamp_policy": {
            "onc_lookup_anchor": "wav_path filename event timestamp",
            "do_not_use_as_lookup_anchor": "ais_timestamp column",
            "reason": "The AIS column is not temporally aligned with the released clip event.",
        },
        "checks": checks,
    }


def build_probe_candidates(
    records: Sequence[OceanshipFGRecord],
    *,
    per_class: int = 3,
    seed: int = 42,
    query_margin_seconds: int = 300,
) -> list[dict[str, object]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if query_margin_seconds < 20:
        raise ValueError("query_margin_seconds must be at least 20")

    random_generator = random.Random(seed)
    output: list[dict[str, object]] = []
    for label in TARGET_LABELS:
        event_records: dict[str, OceanshipFGRecord] = {}
        for record in records:
            if record.label == label:
                event_records.setdefault(record.event_key, record)
        candidates = sorted(
            event_records.values(),
            key=lambda record: (record.event_timestamp_utc, record.mmsi, record.event_key),
        )
        random_generator.shuffle(candidates)

        selected: list[OceanshipFGRecord] = []
        used_mmsi: set[str] = set()
        used_days: set[str] = set()
        for require_new_day in (True, False):
            for record in candidates:
                day = record.event_timestamp_utc[:8]
                if record.mmsi in used_mmsi:
                    continue
                if require_new_day and day in used_days:
                    continue
                selected.append(record)
                used_mmsi.add(record.mmsi)
                used_days.add(day)
                if len(selected) == per_class:
                    break
            if len(selected) == per_class:
                break
        if len(selected) < per_class:
            raise ValueError(f"Not enough independent probe candidates for {label}")

        for record in selected:
            event_time = _parse_utc(record.event_timestamp_utc, compact=True)
            output.append(
                {
                    "candidate_id": hashlib.sha256(record.event_key.encode()).hexdigest()[:16],
                    "label": record.label,
                    "mmsi": record.mmsi,
                    "source_split": record.source_split,
                    "source_wav_path": record.wav_path,
                    "event_timestamp_utc": record.event_timestamp_utc,
                    "query_from_utc": _format_onc_utc(
                        event_time - timedelta(seconds=query_margin_seconds)
                    ),
                    "query_to_utc": _format_onc_utc(
                        event_time + timedelta(seconds=query_margin_seconds)
                    ),
                    "candidate_window_start_utc": _format_onc_utc(event_time),
                    "candidate_window_end_utc": _format_onc_utc(
                        event_time + timedelta(seconds=20)
                    ),
                }
            )
    return output


def write_audit_outputs(
    output_dir: str | Path,
    summary: dict[str, object],
    candidates: Iterable[dict[str, object]],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    rows = list(candidates)
    if not rows:
        raise ValueError("Candidate manifest must not be empty")
    with (root / "onc_probe_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_archive_candidates(
    candidates: Sequence[dict[str, str]],
    onc_client: object,
    *,
    device_codes: Sequence[str] = DEFAULT_DEVICE_CODES,
    extension: str = "wav",
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for candidate in candidates:
        for device_code in device_codes:
            params = {
                "deviceCode": device_code,
                "extension": extension,
                "dateFrom": candidate["query_from_utc"],
                "dateTo": candidate["query_to_utc"],
            }
            response = onc_client.getArchivefile(params, allPages=True)
            files = response.get("files", []) if isinstance(response, dict) else []
            for filename in files:
                results.append(
                    {
                        **candidate,
                        "device_code": device_code,
                        "archive_filename": filename,
                    }
                )
    return results


def read_probe_candidates(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_archive_index(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
