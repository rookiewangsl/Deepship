from __future__ import annotations

import unittest

from src.data.sampling_audit import build_sampling_audit


class SamplingAuditTests(unittest.TestCase):
    def test_balanced_policies_change_recording_and_vessel_exposure(self) -> None:
        manifest = {
            "protocol": "vessel_name_disjoint",
            "manifest_sha256": "abc",
            "segments": [
                {
                    "split": "train",
                    "class_name": "A",
                    "label_index": 0,
                    "relative_path": "A/a1.wav",
                    "start_frame": start,
                    "num_frames": 30,
                }
                for start in (0, 30, 60)
            ]
            + [
                {
                    "split": "train",
                    "class_name": "A",
                    "label_index": 0,
                    "relative_path": "A/a2.wav",
                    "start_frame": 0,
                    "num_frames": 30,
                },
                {
                    "split": "train",
                    "class_name": "B",
                    "label_index": 1,
                    "relative_path": "B/b1.wav",
                    "start_frame": 0,
                    "num_frames": 30,
                },
                {
                    "split": "train",
                    "class_name": "B",
                    "label_index": 1,
                    "relative_path": "B/b2.wav",
                    "start_frame": 30,
                    "num_frames": 30,
                },
            ],
        }
        inventory = [
            {
                "relative_path": path,
                "sample_rate": "10",
                "num_frames": "1000",
                "duration_seconds": "100",
            }
            for path in ("A/a1.wav", "A/a2.wav", "B/b1.wav", "B/b2.wav")
        ]
        assignments = [
            {
                "relative_path": "A/a1.wav",
                "class_name": "A",
                "vessel_key": "v1",
                "partitions": "train",
                "selected_segments": "3",
            },
            {
                "relative_path": "A/a2.wav",
                "class_name": "A",
                "vessel_key": "v1",
                "partitions": "train",
                "selected_segments": "1",
            },
            {
                "relative_path": "B/b1.wav",
                "class_name": "B",
                "vessel_key": "v2",
                "partitions": "train",
                "selected_segments": "1",
            },
            {
                "relative_path": "B/b2.wav",
                "class_name": "B",
                "vessel_key": "v3",
                "partitions": "train",
                "selected_segments": "1",
            },
        ]

        audit = build_sampling_audit(
            manifest,
            inventory,
            assignments,
            clip_duration_seconds=20,
            epoch_samples=100,
        )

        rows = {row["relative_path"]: row for row in audit["recording_rows"]}
        self.assertAlmostEqual(rows["A/a1.wav"]["s0_probability"], 0.5)
        self.assertAlmostEqual(rows["A/a1.wav"]["s1_probability"], 0.25)
        self.assertAlmostEqual(rows["A/a1.wav"]["s2_probability"], 0.25)
        self.assertAlmostEqual(rows["B/b1.wav"]["s2_probability"], 0.25)
        self.assertEqual(audit["recordings"], 4)
        self.assertEqual(audit["vessels"], 3)


if __name__ == "__main__":
    unittest.main()
