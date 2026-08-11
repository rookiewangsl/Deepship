from __future__ import annotations

import unittest

from src.data.deepship_audit import (
    build_audit_report,
    build_exclusion_rows,
    stable_json_hash,
    validate_experiment_config,
    validate_relative_path,
)


def make_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "test",
        "classes": ["Cargo", "Passenger", "Tank", "Tug"],
        "model": {"expected_num_parameters": 1},
        "features": {"clip_duration_seconds": 3.0},
        "split": {
            "target_segments_per_class": {"train": 1, "val": 1, "test": 1},
            "protocols": [
                "segment_level",
                "recording_disjoint",
                "vessel_name_disjoint",
            ],
        },
        "training": {"model_seeds": [42, 43]},
    }


class ConfigTests(unittest.TestCase):
    def test_accepts_frozen_experiment_shape(self) -> None:
        validate_experiment_config(make_config())

    def test_rejects_missing_protocol(self) -> None:
        config = make_config()
        config["split"]["protocols"] = ["segment_level"]  # type: ignore[index]

        with self.assertRaises(ValueError):
            validate_experiment_config(config)


class PathTests(unittest.TestCase):
    def test_accepts_posix_relative_path(self) -> None:
        path = validate_relative_path("Cargo/20171104-1/203623.wav")

        self.assertEqual(path.as_posix(), "Cargo/20171104-1/203623.wav")

    def test_rejects_path_escape(self) -> None:
        with self.assertRaises(ValueError):
            validate_relative_path("../Cargo/example.wav")

    def test_rejects_windows_absolute_path(self) -> None:
        with self.assertRaises(ValueError):
            validate_relative_path(r"D:\\DeepShip\\Cargo\\example.wav")


class AuditTests(unittest.TestCase):
    def test_hash_is_independent_of_dict_key_order(self) -> None:
        self.assertEqual(stable_json_hash({"a": 1, "b": 2}), stable_json_hash({"b": 2, "a": 1}))

    def test_builds_explicit_unresolved_exclusion(self) -> None:
        rows = [
            {
                "relative_path": "Tank/example.wav",
                "class_name": "Tank",
                "match_status": "unresolved",
                "match_confidence": "none",
                "vessel_key": "",
            }
        ]

        exclusions = build_exclusion_rows(rows, allowed_confidence={"high", "medium"})

        self.assertEqual(exclusions[0]["reason"], "unresolved_vessel_identity")

    def test_reports_cross_class_vessel_key(self) -> None:
        config = make_config()
        inventory = []
        identities = []
        for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
            relative_path = f"{class_name}/example.wav"
            inventory.append(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "full_segments": 3,
                }
            )
            identities.append(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "match_status": "matched",
                    "match_method": "exact",
                    "match_confidence": "high",
                    "vessel_key": "SAME" if class_name in {"Cargo", "Tank"} else class_name,
                    "ambiguous_vessel_name": False,
                    "canonical_vessel_name": class_name,
                }
            )

        report = build_audit_report(
            config,
            inventory,
            identities,
            [],
            metadata_record_count=4,
            unmatched_metadata_count=0,
            parse_issues=[],
        )

        self.assertEqual(report["vessel_keys_crossing_classes"], ["SAME"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["validation"]["vessel_keys_do_not_cross_classes"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
