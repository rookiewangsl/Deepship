from __future__ import annotations

import unittest

from src.data.deepship_metadata import (
    MetadataAudioRecord,
    MetadataRecord,
    make_vessel_key,
    match_recordings,
    normalize_vessel_name,
)


def make_metadata(
    *,
    record_id: int = 1,
    date: str = "20171104",
    time: str = "203623",
    duration: float = 10.0,
    name: str = "SEASPAN SWIFT",
) -> MetadataRecord:
    canonical, mmsi, ambiguous = normalize_vessel_name(name)
    return MetadataRecord(
        class_name="Cargo",
        source_path="cargo-metafile",
        source_line=1,
        raw_record_id=record_id,
        ais_type_code="70",
        raw_vessel_name=name,
        canonical_vessel_name=canonical,
        mmsi=mmsi,
        vessel_key=make_vessel_key(canonical, mmsi),
        ambiguous_vessel_name=ambiguous,
        recording_date=date,
        recording_time=time,
        duration_seconds=duration,
        trailing_fields=(),
    )


def make_audio(
    *,
    record_id: int = 1,
    date: str = "20171104",
    time: str = "203623",
    duration: float = 10.0,
) -> MetadataAudioRecord:
    return MetadataAudioRecord(
        path="/data/Cargo/20171104-1/203623.wav",
        relative_path="Cargo/20171104-1/203623.wav",
        class_name="Cargo",
        folder_name=f"{date}-{record_id}",
        folder_record_id=record_id,
        folder_date=date,
        filename_time=time,
        duration_seconds=duration,
    )


class VesselNameTests(unittest.TestCase):
    def test_normalizes_spacing_and_extracts_mmsi(self) -> None:
        canonical, mmsi, ambiguous = normalize_vessel_name("  Paganino   218821000, ")

        self.assertEqual(canonical, "PAGANINO")
        self.assertEqual(mmsi, "218821000")
        self.assertFalse(ambiguous)

    def test_marks_truncated_name_ambiguous(self) -> None:
        canonical, _, ambiguous = normalize_vessel_name("QUEEN OF")

        self.assertEqual(canonical, "QUEEN OF")
        self.assertTrue(ambiguous)


class MatchingTests(unittest.TestCase):
    def test_prefers_exact_date_and_time(self) -> None:
        audio = [make_audio()]
        metadata = [make_metadata()]

        matches, used_audio, used_metadata = match_recordings(audio, metadata)

        self.assertEqual(matches[0].method, "exact_date_time")
        self.assertEqual(used_audio, {0})
        self.assertEqual(used_metadata, {0})

    def test_accepts_unique_time_duration_with_date_typo(self) -> None:
        audio = [make_audio(date="20181104")]
        metadata = [make_metadata(date="20171104")]

        matches, _, _ = match_recordings(audio, metadata)

        self.assertEqual(matches[0].method, "time_and_duration_date_mismatch")
        self.assertEqual(matches[0].confidence, "medium")

    def test_leaves_non_unique_candidate_unresolved(self) -> None:
        audio = [make_audio(date="20181104")]
        metadata = [
            make_metadata(date="20171104", name="SHIP A"),
            make_metadata(date="20161104", name="SHIP B"),
        ]

        matches, used_audio, _ = match_recordings(audio, metadata)

        self.assertEqual(matches, [])
        self.assertEqual(used_audio, set())

    def test_does_not_force_same_date_when_duration_disagrees(self) -> None:
        audio = [make_audio(record_id=99, time="", duration=100.0)]
        metadata = [make_metadata(time="120000", duration=10.0)]

        matches, used_audio, _ = match_recordings(audio, metadata)

        self.assertEqual(matches, [])
        self.assertEqual(used_audio, set())


if __name__ == "__main__":
    unittest.main()
