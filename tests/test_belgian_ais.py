from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from src.data.belgian_ais import (
    BelgianMelDataset,
    BelgianRecord,
    ClassBalancedBatchEpochSampler,
    ClassDateBalancedEpochSampler,
    FullEpochShuffleSampler,
    assign_date_folds,
    build_fold_manifests,
    filter_strict_development_audio,
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
    def test_strict_audio_filter_never_reads_test_and_accepts_mono_or_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = np.zeros(480_000, dtype=np.float32)
            rows = [
                record(1, "Cargo", "2022-01-01"),
                record(2, "Passenger", "2022-01-02"),
                record(3, "Tank", "2022-01-03"),
                record(4, "Tug", "2022-01-04"),
                record(5, "Cargo", "2022-02-01", split="test"),
            ]
            for item in rows[:4]:
                path = root / item.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if item.class_name == "Passenger":
                    sf.write(path, np.column_stack([exact, exact]), 48_000)
                elif item.class_name == "Tank":
                    sf.write(path, exact[:-1], 48_000)
                else:
                    sf.write(path, exact, 48_000)
            filtered, audit = filter_strict_development_audio(rows, data_root=root)
            self.assertEqual(audit["development_admitted"], 3)
            self.assertEqual(audit["admitted_by_channels"], {"1": 2, "2": 1})
            self.assertEqual(audit["rejected_by_reason"], {"not_exact_10_seconds": 1})
            self.assertEqual(audit["sealed_test_audio_status"], "not_inspected")
            self.assertIn(rows[-1], filtered)

    def test_stereo_dataset_uses_channel_zero_not_channel_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 8_000
            samples = 800
            time = np.arange(samples, dtype=np.float32) / sample_rate
            channel_zero = 0.2 * np.sin(2 * np.pi * 440 * time)
            channel_one = 0.8 * np.sin(2 * np.pi * 1200 * time)
            stereo_record = record(10, "Cargo", "2022-01-01")
            mono_record = record(11, "Cargo", "2022-01-01")
            for item, audio in (
                (stereo_record, np.column_stack([channel_zero, channel_one])),
                (mono_record, channel_zero),
            ):
                path = root / item.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, sample_rate)
            dataset = BelgianMelDataset(
                [stereo_record, mono_record],
                data_root=root,
                sample_rate=sample_rate,
                source_sample_rate=sample_rate,
                clip_duration=0.1,
                n_fft=128,
                win_length=128,
                hop_length=64,
                n_mels=16,
            )
            stereo_mel, _ = dataset[0]
            mono_mel, _ = dataset[1]
            self.assertTrue(torch.allclose(stereo_mel, mono_mel, atol=1e-5, rtol=1e-5))

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

    def test_audio_refreeze_preserves_existing_validation_dates(self) -> None:
        records = []
        for date_index in range(12):
            date = f"2022-04-{date_index + 1:02d}"
            for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
                records.append(record(date_index * 10 + len(records), class_name, date))
        audit = {"metadata_sha256": "abc", "filters": {"max_distance_km": 5.0}}
        original, _ = build_fold_manifests(records, audit)
        assignments = {
            row["calendar_date"]: fold_index
            for fold_index, manifest in enumerate(original)
            for row in manifest["records"]
            if row["split"] == "val"
        }
        strict, index = build_fold_manifests(
            records,
            audit,
            audio_audit={"admitted_inventory_sha256": "a" * 64},
            frozen_date_assignments=assignments,
        )
        self.assertEqual(
            index["date_assignment_policy"],
            "preserved_from_pre_audio_audit_frozen_manifests",
        )
        for fold_index, manifest in enumerate(strict):
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(validate_fold_manifest(manifest)["status"], "passed")
            val_dates = {
                row["calendar_date"] for row in manifest["records"] if row["split"] == "val"
            }
            self.assertEqual(
                val_dates,
                {date for date, assigned_fold in assignments.items() if assigned_fold == fold_index},
            )

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

    def test_full_epoch_sampler_visits_every_record_once(self) -> None:
        records = [
            record(index, class_name, f"2022-05-{index % 5 + 1:02d}")
            for index, class_name in enumerate(
                ("Cargo", "Cargo", "Passenger", "Tank", "Tank", "Tank", "Tug")
            )
        ]
        sampler = FullEpochShuffleSampler(records, seed=42)
        first = list(iter(sampler))
        self.assertEqual(sorted(first), list(range(len(records))))
        self.assertEqual(len(set(first)), len(records))
        report = sampler.audit()
        self.assertEqual(report["samples"], len(records))
        self.assertEqual(report["unique_files"], len(records))
        self.assertEqual(report["duplicate_draws"], 0)
        self.assertEqual(report["by_class"]["Cargo"], 2)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(iter(sampler)))

    def test_balanced_batch_sampler_enforces_every_physical_batch(self) -> None:
        records = []
        for class_index, class_name in enumerate(("Cargo", "Passenger", "Tank", "Tug")):
            count = 12 if class_name in {"Cargo", "Tank"} else 3
            for index in range(count):
                records.append(
                    record(
                        class_index * 100 + index,
                        class_name,
                        f"2022-07-{index % 4 + 1:02d}",
                    )
                )
        sampler = ClassBalancedBatchEpochSampler(
            records,
            batch_size=8,
            samples_per_class=8,
            seed=42,
        )
        first = list(iter(sampler))
        self.assertEqual(len(first), 32)
        for start in range(0, len(first), 8):
            counts = {
                name: sum(records[index].class_name == name for index in first[start : start + 8])
                for name in ("Cargo", "Passenger", "Tank", "Tug")
            }
            self.assertEqual(counts, {"Cargo": 2, "Passenger": 2, "Tank": 2, "Tug": 2})
        report = sampler.audit()
        self.assertEqual(report["invalid_balanced_batches"], [])
        self.assertEqual(set(report["by_class"].values()), {8})
        self.assertEqual(report["unique_files_by_class"]["Cargo"], 8)
        self.assertEqual(report["unique_files_by_class"]["Tank"], 8)
        self.assertEqual(report["unique_files_by_class"]["Passenger"], 3)
        self.assertEqual(report["unique_files_by_class"]["Tug"], 3)
        sampler.set_epoch(1)
        second = list(iter(sampler))
        self.assertNotEqual(first, second)
        duplicate = ClassBalancedBatchEpochSampler(
            records,
            batch_size=8,
            samples_per_class=8,
            seed=42,
        )
        duplicate.set_epoch(1)
        self.assertEqual(second, list(iter(duplicate)))

    def test_dataset_applies_frozen_scalar_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = record(20, "Cargo", "2022-06-01")
            path = root / item.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            waveform = np.sin(np.linspace(0, 20, 800, dtype=np.float32))
            sf.write(path, waveform, 8_000)
            plain = BelgianMelDataset(
                [item],
                data_root=root,
                sample_rate=8_000,
                source_sample_rate=8_000,
                clip_duration=0.1,
                n_fft=128,
                win_length=128,
                hop_length=64,
                n_mels=16,
            )
            normalized = BelgianMelDataset(
                [item],
                data_root=root,
                sample_rate=8_000,
                source_sample_rate=8_000,
                clip_duration=0.1,
                n_fft=128,
                win_length=128,
                hop_length=64,
                n_mels=16,
                normalization_mean=2.5,
                normalization_std=4.0,
            )
            plain_mel, _ = plain[0]
            normalized_mel, _ = normalized[0]
            self.assertTrue(torch.allclose(normalized_mel, (plain_mel - 2.5) / 4.0))

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
