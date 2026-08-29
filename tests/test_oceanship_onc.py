from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.data.oceanship_onc import (
    audit_fg_metadata,
    build_probe_candidates,
    load_fg_metadata,
    parse_fg_row,
    query_archive_candidates,
    redact_onc_error,
    write_audit_outputs,
)


FIELDS = ["wav_path", "label", "mmsi", "ais_timestamp"]


def row(label: str, mmsi: int, day: int, index: int, chunk: int | None = None) -> dict[str, str]:
    suffix = "" if chunk is None else f"_{chunk}"
    return {
        "wav_path": f"./wav/202007{day:02d}T120000.000Z_{index}_id_5_typecargo_70{suffix}.pt",
        "label": label,
        "mmsi": str(mmsi),
        "ais_timestamp": f"202007{day:02d}T130000.000Z",
    }


class OceanshipONCTests(unittest.TestCase):
    def test_redacts_token_from_onc_error_text(self) -> None:
        message = (
            "Bad request: https://example.test/api?deviceCode=X&token=secret-value "
            "and {'token': 'another-secret'}"
        )
        redacted = redact_onc_error(message)
        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("another-secret", redacted)
        self.assertEqual(redacted.count("<redacted>"), 2)

    def test_parses_event_timestamp_and_optional_chunk(self) -> None:
        record = parse_fg_row(row("Cargo", 123456789, 15, 7, chunk=2), "train")
        self.assertEqual(record.event_timestamp_utc, "20200715T120000.000Z")
        self.assertEqual(record.chunk_index, 2)
        self.assertEqual(record.mmsi, "123456789")

    def test_audit_exposes_official_split_overlap_and_timestamp_delta(self) -> None:
        rows = []
        for class_index, label in enumerate(("Cargo", "Passenger", "Tanker", "Tug")):
            rows.append(parse_fg_row(row(label, 1000 + class_index, 15 + class_index, class_index), "train"))
        rows.append(parse_fg_row(row("Cargo", 1000, 25, 99), "test"))
        summary = audit_fg_metadata(rows)
        self.assertEqual(summary["status"], "metadata_audit_complete")
        self.assertEqual(summary["official_train_test_mmsi_overlap_count"], 1)
        self.assertEqual(summary["event_to_ais_delta_seconds"]["median"], 3600.0)
        self.assertFalse(summary["checks"]["official_split_is_mmsi_disjoint"])

    def test_builds_balanced_probe_candidates_and_query_filters(self) -> None:
        records = []
        for class_index, label in enumerate(("Cargo", "Passenger", "Tanker", "Tug")):
            for offset in range(3):
                records.append(
                    parse_fg_row(
                        row(label, 10000 + class_index * 10 + offset, 15 + offset, class_index * 10 + offset),
                        "train",
                    )
                )
        candidates = build_probe_candidates(records, per_class=2, seed=42)
        self.assertEqual(len(candidates), 8)
        self.assertEqual({candidate["label"] for candidate in candidates}, {"Cargo", "Passenger", "Tanker", "Tug"})

        class FakeONC:
            def __init__(self) -> None:
                self.calls = []

            def getArchivefile(self, params, allPages=False):
                self.calls.append((params, allPages))
                return {"files": [f"{params['deviceCode']}_sample.wav"]}

        client = FakeONC()
        archive_rows = query_archive_candidates(candidates[:1], client, device_codes=["A", "B"])
        self.assertEqual(len(archive_rows), 2)
        self.assertTrue(all(call[1] for call in client.calls))
        self.assertEqual(client.calls[0][0]["extension"], "wav")

    def test_cli_input_and_outputs_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for split in ("train", "test"):
                path = root / f"oceanship_fg_{split}.csv"
                paths.append(path)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    for class_index, label in enumerate(("Cargo", "Passenger", "Tanker", "Tug")):
                        writer.writerow(row(label, 2000 + class_index + (100 if split == "test" else 0), 15 + class_index, class_index))
            records = load_fg_metadata(paths)
            summary = audit_fg_metadata(records)
            candidates = build_probe_candidates(records, per_class=1)
            output = root / "out"
            write_audit_outputs(output, summary, candidates)
            self.assertTrue((output / "metadata_audit.json").is_file())
            with (output / "onc_probe_candidates.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
