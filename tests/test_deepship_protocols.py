from __future__ import annotations

import unittest

from src.data.deepship_protocols import compile_protocol
from src.data.deepship_protocol_validation import validate_protocol_manifest


def make_config() -> dict[str, object]:
    return {
        "experiment_id": "test",
        "features": {"clip_duration_seconds": 3.0},
        "split": {
            "split_seed": 42,
            "target_segments_per_class": {"train": 2, "val": 1, "test": 1},
            "protocols": [
                "segment_level",
                "recording_disjoint",
                "vessel_name_disjoint",
            ],
        },
    }


def make_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inventory = []
    identities = []
    for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
        for index in range(4):
            relative_path = f"{class_name}/{index}.wav"
            inventory.append(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "sample_rate": 10,
                    "num_frames": 60,
                    "channels": 1,
                    "duration_seconds": 6.0,
                    "file_size_bytes": 1,
                    "content_sha256": f"hash:{class_name}:{index}",
                    "full_segments": 2,
                }
            )
            identities.append(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "vessel_key": f"{class_name}:{index}",
                }
            )
    return inventory, identities


class ProtocolCompilerTests(unittest.TestCase):
    def test_recording_protocol_is_disjoint_and_exact(self) -> None:
        inventory, identities = make_rows()

        manifest, _, _, report = compile_protocol(
            "recording_disjoint",
            make_config(),
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["recordings_crossing_partitions"], [])
        self.assertEqual(len(manifest["segments"]), 16)
        self.assertEqual(report["unused_groups"], 0)

    def test_vessel_protocol_is_disjoint_and_exact(self) -> None:
        inventory, identities = make_rows()

        _, _, _, report = compile_protocol(
            "vessel_name_disjoint",
            make_config(),
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["vessel_name_groups_crossing_partitions"], [])

    def test_compiler_is_deterministic(self) -> None:
        inventory, identities = make_rows()
        first, _, _, _ = compile_protocol(
            "segment_level",
            make_config(),
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )
        second, _, _, _ = compile_protocol(
            "segment_level",
            make_config(),
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )

        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first["segments"], second["segments"])

    def test_validator_detects_manifest_tampering(self) -> None:
        inventory, identities = make_rows()
        config = make_config()
        manifest, _, _, _ = compile_protocol(
            "recording_disjoint",
            config,
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )
        manifest["segments"][0]["start_frame"] = 1
        report = validate_protocol_manifest(
            manifest,
            config,
            {
                "dataset_inventory_sha256": "inventory",
                "recording_identity_manifest_sha256": "identity",
            },
        )

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["manifest_hash_matches"])

    def test_recording_protocol_co_groups_duplicate_content(self) -> None:
        inventory, identities = make_rows()
        inventory[0]["content_sha256"] = "duplicate"
        inventory[1]["content_sha256"] = "duplicate"

        _, _, assignments, report = compile_protocol(
            "recording_disjoint",
            make_config(),
            inventory,
            identities,
            [],
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )

        duplicate_assignments = [
            row for row in assignments if row["group_key"] == "CONTENT:duplicate"
        ]
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(duplicate_assignments), 1)
        self.assertEqual(duplicate_assignments[0]["recordings"], 2)


if __name__ == "__main__":
    unittest.main()
