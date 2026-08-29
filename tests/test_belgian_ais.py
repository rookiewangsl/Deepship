from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.data.belgian_ais import (
    BelgianRecord,
    ClassDateBalancedEpochSampler,
    assign_date_folds,
    build_fold_manifests,
    load_belgian_records,
    validate_fold_manifest,
)
from src.evaluation.belgian_attention import (
    date_balanced_metrics,
    paired_date_cluster_bootstrap,
)


def record(index: int, class_name: str, date: str, *, split: str = "train") -> BelgianRecord:
    labels = {"Cargo": 0, "Passenger": 1, "Tank": 2, "Tug": 3}
    vessel_types = {"Cargo": "Cargo", "Passenger": "Passenger", "Tank": "Tanker", "Tug": "Tug"}
    return BelgianRecord(
        relative_path=f"audio/{class_name}-{index}.wav",
        class_name=class_name,
        label_index=labels[class_name],
        vessel_type=vessel_types[class_name],
        official_split=split,
        event_time=f"{date} 12:00:00+00:00",
        calendar_date=date,
        station="Grafton" if index % 2 else "Gardencity",
        distance_km=1.5,
        activity="underway-using-engine",
    )


class BelgianAISTests(unittest.TestCase):
    def test_metadata_intersection_mapping_distance_and_conflict_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "data_split"
            split_dir.mkdir()
            paths = {
                "train": ["a.wav", "b.wav", "duplicate.wav", "far.wav", "cross.wav"],
                "val": ["c.wav", "cross.wav"],
                "test": ["d.wav"],
            }
            for split, items in paths.items():
                (split_dir / f"{split}.txt").write_text(
                    "\n".join(f"{item} 1.0" for item in items) + "\n",
                    encoding="utf-8",
                )
            metadata = root / "metadata.csv"
            fields = [
                "file_location", "vessel_type", "activity", "SOG", "mmsi",
                "longitude", "latitude", "distance", "event_time", "station",
            ]
            rows = [
                ["a.wav", "Cargo", "underway", "1", "x", "0", "0", "2", "2022-01-01 00:00:00+00:00", "Grafton"],
                ["b.wav", "Towing", "underway", "1", "x", "0", "0", "3", "2022-01-02 00:00:00+00:00", "Grafton"],
                ["c.wav", "Passenger", "underway", "1", "x", "0", "0", "4", "2022-01-03 00:00:00+00:00", "Gardencity"],
                ["d.wav", "Tanker", "underway", "1", "x", "0", "0", "5", "2022-01-04 00:00:00+00:00", "Gardencity"],
                ["far.wav", "Cargo", "underway", "1", "x", "0", "0", "6", "2022-01-05 00:00:00+00:00", "Grafton"],
                ["duplicate.wav", "Cargo", "underway", "1", "x", "0", "0", "2", "2022-01-06 00:00:00+00:00", "Grafton"],
                ["duplicate.wav", "Tanker", "underway", "1", "x", "0", "0", "2", "2022-01-06 00:00:00+00:00", "Grafton"],
                ["cross.wav", "Cargo", "underway", "1", "x", "0", "0", "2", "2022-01-07 00:00:00+00:00", "Grafton"],
            ]
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            records, audit = load_belgian_records(metadata, split_dir, max_distance_km=5.0)
            self.assertEqual([item.class_name for item in records], ["Cargo", "Tug", "Passenger", "Tank"])
            self.assertEqual(audit["conflicting_duplicate_paths_count"], 1)
            self.assertEqual(audit["excluded_distance_by_class"]["Cargo"], 1)
            self.assertEqual(audit["official_split"]["cross_split_paths_count"], 1)

    def test_date_folds_are_deterministic_disjoint_and_complete(self) -> None:
        records = []
        for date_index in range(12):
            date = f"2022-01-{date_index + 1:02d}"
            for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
                records.append(record(date_index * 10 + len(records), class_name, date))
        first = assign_date_folds(records, seed=42)
        second = assign_date_folds(records, seed=42)
        self.assertEqual(first, second)
        audit = {"metadata_sha256": "abc", "filters": {"max_distance_km": 5.0}}
        manifests, index = build_fold_manifests(records, audit)
        self.assertEqual(len(manifests), 3)
        self.assertEqual(index["sealed_test_manifest"]["records"], [])
        for manifest in manifests:
            self.assertEqual(validate_fold_manifest(manifest)["status"], "passed")

    def test_development_excludes_dates_present_in_sealed_test(self) -> None:
        records = []
        for date_index in range(9):
            date = f"2022-03-{date_index + 1:02d}"
            for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
                records.append(record(date_index * 10 + len(records), class_name, date))
        test_date = "2022-03-09"
        records.append(record(999, "Cargo", test_date, split="test"))
        audit = {"metadata_sha256": "abc", "filters": {"max_distance_km": 5.0}}
        manifests, index = build_fold_manifests(records, audit)
        self.assertIn(test_date, index["sealed_test_dates"])
        self.assertEqual(index["development_records_excluded_for_test_date_isolation"], 4)
        for manifest in manifests:
            self.assertNotIn(test_date, {row["calendar_date"] for row in manifest["records"]})

    def test_sampler_balances_classes_and_spreads_dates(self) -> None:
        records = []
        for class_index, class_name in enumerate(("Cargo", "Passenger", "Tank", "Tug")):
            count = 8 if class_name in {"Cargo", "Tank"} else 4
            for index in range(count):
                records.append(record(class_index * 100 + index, class_name, f"2022-02-{index % 3 + 1:02d}"))
        sampler = ClassDateBalancedEpochSampler(records, seed=42)
        report = sampler.audit()
        self.assertEqual(report["samples"], 16)
        self.assertEqual(report["unique_files"], 16)
        self.assertEqual(report["duplicate_draws"], 0)
        self.assertEqual(set(report["by_class"].values()), {4})
        self.assertEqual(report["by_distance_km"], {"1-2": 16})
        self.assertTrue(all(value >= 2 for value in report["unique_dates_by_class"].values()))
        first = list(iter(sampler))
        sampler.set_epoch(1)
        second = list(iter(sampler))
        self.assertNotEqual(first, second)

    def test_date_balanced_metrics_and_paired_bootstrap(self) -> None:
        rows = []
        for index, (date, truth, prediction) in enumerate(
            [("d1", 0, 0), ("d1", 1, 0), ("d2", 0, 0), ("d2", 1, 1)]
        ):
            rows.append(
                {
                    "relative_path": f"{index}.wav",
                    "calendar_date": date,
                    "station": "A",
                    "distance_km": 1.0,
                    "true_label": truth,
                    "predicted_label": prediction,
                }
            )
        class_names = ["Cargo", "Passenger"]
        metrics = date_balanced_metrics(rows, class_names)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        improved = [dict(row) for row in rows]
        improved[1]["predicted_label"] = 1
        result = paired_date_cluster_bootstrap(rows, improved, class_names, resamples=100, seed=42)
        self.assertGreater(result["delta_macro_f1"], 0.0)
        self.assertGreater(result["probability_delta_gt_zero"], 0.0)


if __name__ == "__main__":
    unittest.main()
