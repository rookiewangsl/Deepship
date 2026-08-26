from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.data.portia import build_portia4_manifest, load_portia4_records


FIELDS = ["window_id", "label_vessel", "n_vessels", "primary_class", "primary_dist", "primary_mmsi"]


class PortiaManifestTests(unittest.TestCase):
    def _write_annotations(self, root: Path) -> None:
        classes = ("cargo", "passenger", "tanker", "tug")
        rows = []
        for class_index, raw_class in enumerate(classes):
            for group_index in range(2):
                rows.append(
                    {
                        "window_id": f"{raw_class}_{group_index}.wav",
                        "label_vessel": "1",
                        "n_vessels": "1",
                        "primary_class": raw_class,
                        "primary_dist": "0.5",
                        "primary_mmsi": f"{1000 + class_index * 10 + group_index}.0",
                    }
                )
        rows.extend(
            [
                {"window_id": "ignored_other.wav", "label_vessel": "1", "n_vessels": "1", "primary_class": "other", "primary_dist": "0.5", "primary_mmsi": "999"},
                {"window_id": "ignored_multi.wav", "label_vessel": "1", "n_vessels": "2", "primary_class": "cargo", "primary_dist": "0.5", "primary_mmsi": "998"},
            ]
        )
        for split, subset in {"train": rows[:5], "val": rows[5:8], "test": rows[8:]}.items():
            with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(subset)

    def test_builds_deterministic_mmsi_disjoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_annotations(root)
            records = load_portia4_records(root)
            first_manifest, first_summary = build_portia4_manifest(records, seed=42)
            second_manifest, _ = build_portia4_manifest(records, seed=42)

        self.assertEqual(len(records), 8)
        self.assertEqual(first_manifest["manifest_sha256"], second_manifest["manifest_sha256"])
        self.assertEqual(first_summary["status"], "passed")
        self.assertTrue(all(first_summary["checks"].values()))
        self.assertEqual(first_summary["mmsi_by_split_and_class"]["development"], {"Cargo": 1, "Passenger": 1, "Tank": 1, "Tug": 1})

    def test_rejects_duplicate_window_across_source_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_annotations(root)
            with (root / "test.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["cargo_0.wav", "1", "1", "cargo", "0.5", "1000"])
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                load_portia4_records(root)


if __name__ == "__main__":
    unittest.main()
