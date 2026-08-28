from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.embedding_diagnostics import (
    aggregate_by_recording,
    build_recording_disjoint_probe_split,
    diagnose_embeddings,
    nearest_neighbor_identity_rates,
)


class EmbeddingDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.9, 0.1],
                [0.85, 0.15],
                [0.0, 1.0],
                [0.05, 0.95],
                [0.1, 0.9],
                [0.15, 0.85],
            ],
            dtype=np.float32,
        )
        self.classes = ["Cargo"] * 4 + ["Tank"] * 4
        self.vessels = ["A", "A", "A", "A", "B", "B", "B", "B"]
        self.recordings = ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"]

    def test_aggregate_by_recording_preserves_metadata(self) -> None:
        values, metadata = aggregate_by_recording(
            self.embeddings,
            class_names=self.classes,
            vessel_keys=self.vessels,
            recording_paths=self.recordings,
        )
        self.assertEqual(values.shape, (4, 2))
        self.assertEqual(metadata["recording_paths"], ["a1", "a2", "b1", "b2"])
        self.assertEqual(metadata["segment_counts"], [2, 2, 2, 2])

    def test_probe_split_holds_out_whole_recordings(self) -> None:
        split = build_recording_disjoint_probe_split(
            class_names=self.classes,
            vessel_keys=self.vessels,
            recording_paths=self.recordings,
            seed=42,
        )
        train_recordings = {self.recordings[index] for index in split["train_indexes"]}
        test_recordings = {self.recordings[index] for index in split["test_indexes"]}
        self.assertFalse(train_recordings & test_recordings)
        self.assertEqual(split["eligible_vessels"], 2)

    def test_nearest_neighbor_rates_detect_identity_clusters(self) -> None:
        rates = nearest_neighbor_identity_rates(
            self.embeddings,
            class_names=self.classes,
            vessel_keys=self.vessels,
            max_samples=8,
            seed=42,
        )
        self.assertEqual(rates["same_vessel_rate"], 1.0)
        self.assertEqual(rates["same_class_rate"], 1.0)

    def test_nearest_neighbor_is_unavailable_for_one_recording(self) -> None:
        rates = nearest_neighbor_identity_rates(
            self.embeddings[:1],
            class_names=self.classes[:1],
            vessel_keys=self.vessels[:1],
            max_samples=1,
            seed=42,
        )
        self.assertEqual(rates, {"samples": 1, "status": "unavailable"})

    def test_diagnose_embeddings_returns_segment_and_recording_results(self) -> None:
        report = diagnose_embeddings(
            self.embeddings,
            class_names=self.classes,
            vessel_keys=self.vessels,
            recording_paths=self.recordings,
            seed=42,
            max_metric_samples=8,
        )
        self.assertEqual(report["segments"], 8)
        self.assertEqual(report["recordings"], 4)
        self.assertEqual(report["embedding_dimension"], 2)
        self.assertIn("recording_nearest_neighbor", report)


if __name__ == "__main__":
    unittest.main()
