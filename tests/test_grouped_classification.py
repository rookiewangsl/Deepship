from __future__ import annotations

import unittest

from src.evaluation.grouped_classification import (
    aggregate_recording_predictions,
    aggregate_vessel_predictions,
)


class GroupedPredictionTests(unittest.TestCase):
    def test_recording_averages_segment_probabilities(self) -> None:
        rows = [
            {
                "relative_path": "Cargo/a.wav",
                "vessel_key": "VESSEL:A",
                "true_label": 0,
                "probabilities": [0.8, 0.2],
            },
            {
                "relative_path": "Cargo/a.wav",
                "vessel_key": "VESSEL:A",
                "true_label": 0,
                "probabilities": [0.6, 0.4],
            },
        ]

        result = aggregate_recording_predictions(rows)

        self.assertEqual(result[0]["predicted_label"], 0)
        self.assertEqual(result[0]["segments"], 2)
        self.assertAlmostEqual(result[0]["probabilities"][0], 0.7)

    def test_vessel_averages_recordings_equally(self) -> None:
        recordings = [
            {
                "relative_path": "Cargo/long.wav",
                "vessel_key": "VESSEL:A",
                "true_label": 0,
                "probabilities": [0.4, 0.6],
                "segments": 100,
            },
            {
                "relative_path": "Cargo/short.wav",
                "vessel_key": "VESSEL:A",
                "true_label": 0,
                "probabilities": [0.8, 0.2],
                "segments": 1,
            },
        ]

        result = aggregate_vessel_predictions(recordings)

        self.assertEqual(result[0]["predicted_label"], 0)
        self.assertEqual(result[0]["recordings"], 2)
        self.assertAlmostEqual(result[0]["probabilities"][0], 0.6)

    def test_rejects_conflicting_vessel_labels(self) -> None:
        recordings = [
            {"vessel_key": "VESSEL:A", "true_label": 0, "probabilities": [1.0, 0.0]},
            {"vessel_key": "VESSEL:A", "true_label": 1, "probabilities": [0.0, 1.0]},
        ]

        with self.assertRaises(ValueError):
            aggregate_vessel_predictions(recordings)


if __name__ == "__main__":
    unittest.main()
