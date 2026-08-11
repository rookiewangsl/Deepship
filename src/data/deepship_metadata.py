from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import re
from typing import Iterable

import soundfile as sf


METADATA_FILES = {
    "Cargo": "cargo-metafile",
    "Passenger": "passengership-metafile",
    "Tank": "tanker-metafile",
    "Tug": "tug-metafile",
}

AMBIGUOUS_CANONICAL_NAMES = {
    "CABO DE",
    "QUEEN OF",
    "SPIRIT OF",
}

MANIFEST_FIELDS = [
    "recording_path",
    "relative_path",
    "class_name",
    "folder_name",
    "folder_record_id",
    "folder_date",
    "filename_time",
    "audio_duration_seconds",
    "match_status",
    "match_method",
    "match_confidence",
    "metadata_source",
    "metadata_source_line",
    "metadata_record_id",
    "ais_type_code",
    "metadata_date",
    "metadata_time",
    "metadata_duration_seconds",
    "raw_vessel_name",
    "canonical_vessel_name",
    "mmsi",
    "vessel_key",
    "ambiguous_vessel_name",
]


@dataclass(frozen=True)
class MetadataRecord:
    class_name: str
    source_path: str
    source_line: int
    raw_record_id: int
    ais_type_code: str
    raw_vessel_name: str
    canonical_vessel_name: str
    mmsi: str
    vessel_key: str
    ambiguous_vessel_name: bool
    recording_date: str
    recording_time: str
    duration_seconds: float
    trailing_fields: tuple[str, ...]


@dataclass(frozen=True)
class MetadataAudioRecord:
    path: str
    relative_path: str
    class_name: str
    folder_name: str
    folder_record_id: int
    folder_date: str
    filename_time: str
    duration_seconds: float


@dataclass(frozen=True)
class RecordMatch:
    audio_index: int
    metadata_index: int
    method: str
    confidence: str


def normalize_vessel_name(
    raw_name: str,
    aliases: dict[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Return canonical name, optional MMSI, and ambiguity flag."""
    cleaned = " ".join(raw_name.replace("\ufeff", "").strip(" ,;\t").upper().split())
    mmsi = ""
    match = re.fullmatch(r"(.+?)\s+(\d{9})", cleaned)
    if match:
        cleaned = match.group(1).strip()
        mmsi = match.group(2)

    alias_map = aliases or {}
    canonical = alias_map.get(cleaned, cleaned)
    ambiguous = canonical in AMBIGUOUS_CANONICAL_NAMES
    return canonical, mmsi, ambiguous


def make_vessel_key(canonical_name: str, mmsi: str) -> str:
    if mmsi:
        return f"MMSI:{mmsi}"
    if not canonical_name:
        return ""
    digest = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()[:12]
    return f"NAME:{digest}"


def load_aliases(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    alias_path = Path(path).expanduser().resolve()
    if not alias_path.exists():
        raise FileNotFoundError(f"Vessel alias file not found: {alias_path}")

    aliases: dict[str, str] = {}
    with alias_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_name = " ".join((row.get("raw_name") or "").strip().upper().split())
            canonical_name = " ".join((row.get("canonical_name") or "").strip().upper().split())
            if raw_name and canonical_name:
                aliases[raw_name] = canonical_name
    return aliases


def parse_metadata_records(
    root_dir: str | Path,
    *,
    aliases: dict[str, str] | None = None,
) -> tuple[list[MetadataRecord], list[dict[str, object]]]:
    root = Path(root_dir).expanduser().resolve()
    records: list[MetadataRecord] = []
    issues: list[dict[str, object]] = []

    for class_name, filename in METADATA_FILES.items():
        source = root / class_name / filename
        if not source.exists():
            issues.append(
                {
                    "issue_type": "missing_metadata_file",
                    "class_name": class_name,
                    "source_path": str(source),
                }
            )
            continue

        with source.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for source_line, row in enumerate(csv.reader(handle), start=1):
                if not row or not row[0].strip():
                    continue
                if len(row) < 6 or not row[0].strip().isdigit():
                    issues.append(
                        {
                            "issue_type": "malformed_metadata_row",
                            "class_name": class_name,
                            "source_path": str(source),
                            "source_line": source_line,
                            "raw_row": row,
                        }
                    )
                    continue
                try:
                    duration_seconds = float(row[5].strip())
                except ValueError:
                    issues.append(
                        {
                            "issue_type": "invalid_metadata_duration",
                            "class_name": class_name,
                            "source_path": str(source),
                            "source_line": source_line,
                            "raw_row": row,
                        }
                    )
                    continue

                raw_name = row[2].strip()
                canonical_name, mmsi, ambiguous = normalize_vessel_name(raw_name, aliases)
                records.append(
                    MetadataRecord(
                        class_name=class_name,
                        source_path=str(source),
                        source_line=source_line,
                        raw_record_id=int(row[0].strip()),
                        ais_type_code=row[1].strip(),
                        raw_vessel_name=raw_name,
                        canonical_vessel_name=canonical_name,
                        mmsi=mmsi,
                        vessel_key=make_vessel_key(canonical_name, mmsi),
                        ambiguous_vessel_name=ambiguous,
                        recording_date=row[3].strip(),
                        recording_time=row[4].strip().zfill(6),
                        duration_seconds=duration_seconds,
                        trailing_fields=tuple(value.strip() for value in row[6:] if value.strip()),
                    )
                )
    return records, issues


def _valid_time(value: str) -> bool:
    if not re.fullmatch(r"\d{6}", value):
        return False
    hour, minute, second = int(value[:2]), int(value[2:4]), int(value[4:])
    return hour < 24 and minute < 60 and second < 60


def scan_metadata_audio(root_dir: str | Path) -> list[MetadataAudioRecord]:
    root = Path(root_dir).expanduser().resolve()
    records: list[MetadataAudioRecord] = []
    for class_name in METADATA_FILES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for wav_path in sorted(class_dir.rglob("*.wav")):
            if wav_path.name.startswith("._"):
                continue
            folder_match = re.search(r"-(\d+)$", wav_path.parent.name)
            date_match = re.match(r"(\d{8})", wav_path.parent.name)
            if folder_match is None or date_match is None:
                raise ValueError(f"Unexpected DeepShip recording folder: {wav_path.parent}")
            info = sf.info(str(wav_path))
            filename_time = wav_path.stem if _valid_time(wav_path.stem) else ""
            records.append(
                MetadataAudioRecord(
                    path=str(wav_path.resolve()),
                    relative_path=wav_path.relative_to(root).as_posix(),
                    class_name=class_name,
                    folder_name=wav_path.parent.name,
                    folder_record_id=int(folder_match.group(1)),
                    folder_date=date_match.group(1),
                    filename_time=filename_time,
                    duration_seconds=info.frames / info.samplerate,
                )
            )
    return records


def _duration_matches(audio: MetadataAudioRecord, metadata: MetadataRecord) -> bool:
    return abs(audio.duration_seconds - metadata.duration_seconds) <= 1.01


def match_recordings(
    audio_records: list[MetadataAudioRecord],
    metadata_records: list[MetadataRecord],
) -> tuple[list[RecordMatch], set[int], set[int]]:
    matches: list[RecordMatch] = []
    used_audio: set[int] = set()
    used_metadata: set[int] = set()

    def add_match(audio_index: int, metadata_index: int, method: str, confidence: str) -> None:
        if audio_index in used_audio or metadata_index in used_metadata:
            raise ValueError("Attempted to reuse an audio or metadata record")
        matches.append(RecordMatch(audio_index, metadata_index, method, confidence))
        used_audio.add(audio_index)
        used_metadata.add(metadata_index)

    def candidates(audio_index: int, predicate) -> list[int]:
        audio = audio_records[audio_index]
        return [
            metadata_index
            for metadata_index, metadata in enumerate(metadata_records)
            if metadata_index not in used_metadata
            and metadata.class_name == audio.class_name
            and predicate(audio, metadata)
        ]

    # File timestamps are the strongest key and are unique for almost all recordings.
    for audio_index, audio in enumerate(audio_records):
        if not audio.filename_time:
            continue
        found = candidates(
            audio_index,
            lambda a, m: a.folder_date == m.recording_date and a.filename_time == m.recording_time,
        )
        if len(found) == 1:
            add_match(audio_index, found[0], "exact_date_time", "high")
        elif len(found) > 1:
            signatures = {
                (
                    metadata_records[index].vessel_key,
                    metadata_records[index].duration_seconds,
                )
                for index in found
            }
            if len(signatures) == 1:
                add_match(audio_index, found[0], "duplicate_equivalent_date_time", "high")

    # Cargo, Passenger, and Tug retain stable source IDs; this also resolves the
    # initial uncorrupted Tank rows without relying on the reset IDs.
    for audio_index, audio in enumerate(audio_records):
        if audio_index in used_audio:
            continue
        found = candidates(
            audio_index,
            lambda a, m: (
                a.folder_record_id == m.raw_record_id and a.folder_date == m.recording_date
            ),
        )
        if len(found) == 1:
            add_match(audio_index, found[0], "record_id_and_date", "high")

    # A date and measured duration pair recovers Tank rows whose IDs shifted.
    for audio_index, audio in enumerate(audio_records):
        if audio_index in used_audio:
            continue
        found = candidates(
            audio_index,
            lambda a, m: a.folder_date == m.recording_date and _duration_matches(a, m),
        )
        if len(found) == 1:
            add_match(audio_index, found[0], "date_and_duration", "medium")

    # When multiple short recordings share a date, pair them by duration with a
    # deterministic one-to-one choice only if every best choice is unique.
    for class_name in METADATA_FILES:
        dates = sorted(
            {
                audio.folder_date
                for index, audio in enumerate(audio_records)
                if index not in used_audio and audio.class_name == class_name
            }
        )
        for recording_date in dates:
            audio_indexes = [
                index
                for index, audio in enumerate(audio_records)
                if index not in used_audio
                and audio.class_name == class_name
                and audio.folder_date == recording_date
            ]
            metadata_indexes = [
                index
                for index, metadata in enumerate(metadata_records)
                if index not in used_metadata
                and metadata.class_name == class_name
                and metadata.recording_date == recording_date
            ]
            if not audio_indexes or len(audio_indexes) != len(metadata_indexes):
                continue
            proposals: dict[int, int] = {}
            for audio_index in audio_indexes:
                ranked = sorted(
                    (
                        abs(
                            audio_records[audio_index].duration_seconds
                            - metadata_records[metadata_index].duration_seconds
                        ),
                        metadata_records[metadata_index].recording_time,
                        metadata_index,
                    )
                    for metadata_index in metadata_indexes
                )
                if ranked[0][0] <= 1.01 and (len(ranked) == 1 or ranked[0][0] < ranked[1][0]):
                    proposals[audio_index] = ranked[0][2]
            if len(proposals) == len(audio_indexes) and len(set(proposals.values())) == len(proposals):
                for audio_index in sorted(proposals):
                    add_match(audio_index, proposals[audio_index], "same_date_duration_assignment", "medium")

    # Two Tank paths contain a one-year folder typo but preserve both timestamp
    # and duration. Requiring both to be unique prevents broad fuzzy matching.
    for audio_index, audio in enumerate(audio_records):
        if audio_index in used_audio or not audio.filename_time:
            continue
        found = candidates(
            audio_index,
            lambda a, m: a.filename_time == m.recording_time and _duration_matches(a, m),
        )
        if len(found) == 1:
            add_match(audio_index, found[0], "time_and_duration_date_mismatch", "medium")

    return matches, used_audio, used_metadata


def build_manifest_rows(
    audio_records: list[MetadataAudioRecord],
    metadata_records: list[MetadataRecord],
    matches: Iterable[RecordMatch],
) -> list[dict[str, object]]:
    by_audio = {match.audio_index: match for match in matches}
    rows: list[dict[str, object]] = []
    for audio_index, audio in enumerate(audio_records):
        match = by_audio.get(audio_index)
        metadata = metadata_records[match.metadata_index] if match else None
        rows.append(
            {
                "recording_path": audio.path,
                "relative_path": audio.relative_path,
                "class_name": audio.class_name,
                "folder_name": audio.folder_name,
                "folder_record_id": audio.folder_record_id,
                "folder_date": audio.folder_date,
                "filename_time": audio.filename_time,
                "audio_duration_seconds": round(audio.duration_seconds, 6),
                "match_status": "matched" if metadata else "unresolved",
                "match_method": match.method if match else "",
                "match_confidence": match.confidence if match else "none",
                "metadata_source": metadata.source_path if metadata else "",
                "metadata_source_line": metadata.source_line if metadata else "",
                "metadata_record_id": metadata.raw_record_id if metadata else "",
                "ais_type_code": metadata.ais_type_code if metadata else "",
                "metadata_date": metadata.recording_date if metadata else "",
                "metadata_time": metadata.recording_time if metadata else "",
                "metadata_duration_seconds": metadata.duration_seconds if metadata else "",
                "raw_vessel_name": metadata.raw_vessel_name if metadata else "",
                "canonical_vessel_name": metadata.canonical_vessel_name if metadata else "",
                "mmsi": metadata.mmsi if metadata else "",
                "vessel_key": metadata.vessel_key if metadata else "",
                "ambiguous_vessel_name": metadata.ambiguous_vessel_name if metadata else "",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_cleaning_outputs(
    output_dir: str | Path,
    audio_records: list[MetadataAudioRecord],
    metadata_records: list[MetadataRecord],
    matches: list[RecordMatch],
    used_audio: set[int],
    used_metadata: set[int],
    parse_issues: list[dict[str, object]],
) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = build_manifest_rows(audio_records, metadata_records, matches)
    _write_csv(output / "recording_vessel_manifest.csv", manifest_rows, MANIFEST_FIELDS)

    unresolved_rows: list[dict[str, object]] = []
    for audio_index, audio in enumerate(audio_records):
        if audio_index not in used_audio:
            unresolved_rows.append(
                {
                    "issue_type": "audio_without_metadata_match",
                    "class_name": audio.class_name,
                    "audio_relative_path": audio.relative_path,
                    "metadata_source_line": "",
                    "raw_record_id": audio.folder_record_id,
                    "recording_date": audio.folder_date,
                    "recording_time": audio.filename_time,
                    "raw_vessel_name": "",
                    "details": "No unique metadata row satisfied the conservative matching rules.",
                }
            )
    for metadata_index, metadata in enumerate(metadata_records):
        if metadata_index not in used_metadata:
            unresolved_rows.append(
                {
                    "issue_type": "metadata_without_audio_match",
                    "class_name": metadata.class_name,
                    "audio_relative_path": "",
                    "metadata_source_line": metadata.source_line,
                    "raw_record_id": metadata.raw_record_id,
                    "recording_date": metadata.recording_date,
                    "recording_time": metadata.recording_time,
                    "raw_vessel_name": metadata.raw_vessel_name,
                    "details": "Metadata row was not linked to a local WAV file.",
                }
            )
    unresolved_fields = [
        "issue_type",
        "class_name",
        "audio_relative_path",
        "metadata_source_line",
        "raw_record_id",
        "recording_date",
        "recording_time",
        "raw_vessel_name",
        "details",
    ]
    _write_csv(output / "unresolved_records.csv", unresolved_rows, unresolved_fields)

    vessel_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        if row["vessel_key"]:
            vessel_groups[str(row["vessel_key"])].append(row)
    vessel_rows: list[dict[str, object]] = []
    for vessel_key, rows in sorted(vessel_groups.items()):
        vessel_rows.append(
            {
                "vessel_key": vessel_key,
                "canonical_vessel_name": rows[0]["canonical_vessel_name"],
                "mmsi": rows[0]["mmsi"],
                "ambiguous_vessel_name": rows[0]["ambiguous_vessel_name"],
                "classes": ";".join(sorted({str(row["class_name"]) for row in rows})),
                "num_recordings": len(rows),
            }
        )
    _write_csv(
        output / "vessels.csv",
        vessel_rows,
        [
            "vessel_key",
            "canonical_vessel_name",
            "mmsi",
            "ambiguous_vessel_name",
            "classes",
            "num_recordings",
        ],
    )

    class_audio_counts = Counter(record.class_name for record in audio_records)
    class_match_counts = Counter(audio_records[match.audio_index].class_name for match in matches)
    class_vessel_counts = Counter()
    for row in vessel_rows:
        for class_name in str(row["classes"]).split(";"):
            class_vessel_counts[class_name] += 1
    method_counts = Counter(match.method for match in matches)
    confidence_counts = Counter(match.confidence for match in matches)
    ambiguous_rows = [row for row in manifest_rows if row["ambiguous_vessel_name"] is True]

    report = {
        "num_audio_recordings": len(audio_records),
        "num_metadata_records": len(metadata_records),
        "num_matched_recordings": len(matches),
        "num_unresolved_recordings": len(audio_records) - len(used_audio),
        "num_unmatched_metadata_records": len(metadata_records) - len(used_metadata),
        "num_canonical_vessel_groups": len(vessel_rows),
        "num_ambiguous_name_recordings": len(ambiguous_rows),
        "ambiguous_canonical_names": sorted(
            {str(row["canonical_vessel_name"]) for row in ambiguous_rows}
        ),
        "audio_recordings_by_class": {
            class_name: class_audio_counts.get(class_name, 0) for class_name in METADATA_FILES
        },
        "matched_recordings_by_class": {
            class_name: class_match_counts.get(class_name, 0) for class_name in METADATA_FILES
        },
        "canonical_vessel_groups_by_class": {
            class_name: class_vessel_counts.get(class_name, 0) for class_name in METADATA_FILES
        },
        "match_methods": dict(sorted(method_counts.items())),
        "match_confidence": dict(sorted(confidence_counts.items())),
        "metadata_parse_issues": parse_issues,
        "validation": {
            "manifest_rows_equal_audio_files": len(manifest_rows) == len(audio_records),
            "unique_audio_paths": len({row["recording_path"] for row in manifest_rows})
            == len(manifest_rows),
            "matched_rows_have_vessel_key": all(
                row["vessel_key"] for row in manifest_rows if row["match_status"] == "matched"
            ),
            "vessel_keys_crossing_classes": sorted(
                row["vessel_key"] for row in vessel_rows if ";" in str(row["classes"])
            ),
        },
    }
    (output / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def clean_deepship_metadata(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    alias_path: str | Path | None = None,
) -> dict[str, object]:
    aliases = load_aliases(alias_path)
    metadata_records, parse_issues = parse_metadata_records(data_root, aliases=aliases)
    audio_records = scan_metadata_audio(data_root)
    matches, used_audio, used_metadata = match_recordings(audio_records, metadata_records)
    return save_cleaning_outputs(
        output_dir,
        audio_records,
        metadata_records,
        matches,
        used_audio,
        used_metadata,
        parse_issues,
    )


def metadata_record_to_dict(record: MetadataRecord) -> dict[str, object]:
    return asdict(record)
