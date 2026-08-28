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
        self.assertEqual(config["training"]["log_interval_batches"], 100)

    def test_cli_defaults_to_verified_4070_precision_and_memory_mode(self) -> None:
        arguments = build_parser().parse_args(
            ["--output-root", "unused", "--split-manifest", "unused.json"]
        )
        self.assertEqual(arguments.precision, "bf16")
        self.assertFalse(arguments.gradient_checkpointing)
        self.assertEqual(arguments.log_interval, 100)
        self.assertIsNone(arguments.eval_batch_size)
        self.assertEqual(arguments.training_sampling, "fixed_anchor")
        self.assertIsNone(arguments.train_samples_per_epoch)
        self.assertEqual(arguments.early_stopping_min_delta, 0.0)
        self.assertFalse(arguments.evaluate_test_on_completion)

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

        dynamic = build_parser().parse_args(
            [
                "--output-root",
                "unused",
                "--split-manifest",
                "unused.json",
                "--training-sampling",
                "recording_balanced_dynamic",
                "--train-samples-per-epoch",
                "14000",
                "--eval-batch-size",
                "2",
                "--early-stopping-min-delta",
                "0.005",
            ]
        )
        self.assertEqual(dynamic.training_sampling, "recording_balanced_dynamic")
        self.assertEqual(dynamic.train_samples_per_epoch, 14000)
        self.assertEqual(dynamic.eval_batch_size, 2)
        self.assertEqual(dynamic.early_stopping_min_delta, 0.005)

    def test_sampling_ablation_changes_only_training_exposure(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "experiments" / "conformer_sampling_v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(config["sampling"]["policy"], "recording_balanced_dynamic")
        self.assertEqual(config["sampling"]["samples_per_epoch"], 14000)
        self.assertEqual(config["training"]["max_epochs"], 8)
        self.assertEqual(config["training"]["early_stopping_patience"], 3)
        self.assertEqual(config["training"]["early_stopping_min_delta"], 0.005)
        self.assertFalse(config["selection"]["evaluate_test_on_completion"])
        self.assertEqual(
            config["source_protocol"]["validation_and_test_windows"],
            "fixed_manifest_anchor",
        )


if __name__ == "__main__":
    unittest.main()
