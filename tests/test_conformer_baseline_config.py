from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.train.train_deepship_conformer import build_parser


ROOT = Path(__file__).resolve().parents[1]


class ConformerBaselineConfigTests(unittest.TestCase):
    def test_baseline_uses_strict_manifest_and_four_classes(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "experiments" / "conformer_baseline_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["classes"], ["Cargo", "Passenger", "Tank", "Tug"])
        self.assertIn("vessel_name_disjoint", config["source_protocol"]["primary_manifest"])
        self.assertFalse(config["source_protocol"]["modify_source_manifest"])
        self.assertEqual(config["input"]["sample_rate"], 16000)
        self.assertTrue(config["model"]["load_pretrained"])
        self.assertFalse(config["model"]["apply_spec_augment"])
        self.assertEqual(config["model"]["layerdrop"], 0.0)
        self.assertFalse(config["model"]["gradient_checkpointing"])
        self.assertEqual(config["training"]["precision"], "bf16")

    def test_cli_defaults_to_verified_4070_precision_and_memory_mode(self) -> None:
        arguments = build_parser().parse_args(
            ["--output-root", "unused", "--split-manifest", "unused.json"]
        )
        self.assertEqual(arguments.precision, "bf16")
        self.assertFalse(arguments.gradient_checkpointing)

        enabled = build_parser().parse_args(
            [
                "--output-root",
                "unused",
                "--split-manifest",
                "unused.json",
                "--gradient-checkpointing",
            ]
        )
        self.assertTrue(enabled.gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
