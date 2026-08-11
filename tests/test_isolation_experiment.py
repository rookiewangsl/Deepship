from __future__ import annotations

import unittest

from src.pipelines.mel_ml.isolation_experiment import enforce_training_config


def make_experiment() -> dict[str, object]:
    return {
        "features": {
            "clip_duration_seconds": 3.0,
            "n_fft": 1024,
            "hop_length": 512,
            "win_length": 1024,
            "n_mels": 64,
            "highpass_freq": None,
        },
        "split": {
            "target_segments_per_class": {"train": 3500, "val": 1000, "test": 500}
        },
        "training": {
            "model_seeds": [42, 43, 44],
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "min_learning_rate": 0.00001,
            "warmup_epochs": 10,
            "early_stopping_patience": 10,
        },
    }


def make_training() -> dict[str, object]:
    return {
        "clip_duration": 3.0,
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 1024,
        "n_mels": 64,
        "highpass_freq": None,
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "min_learning_rate": 0.00001,
        "warmup_epochs": 10,
        "early_stopping_patience": 10,
        "train_per_class": 3500,
        "val_per_class": 1000,
        "test_per_class": 500,
        "seed": 42,
        "max_train_batches": None,
        "max_eval_batches": None,
    }


class FrozenTrainingConfigTests(unittest.TestCase):
    def test_accepts_exact_formal_configuration(self) -> None:
        mismatches = enforce_training_config(
            make_training(),
            make_experiment(),
            allow_overrides=False,
        )

        self.assertEqual(mismatches, [])

    def test_rejects_formal_override(self) -> None:
        training = make_training()
        training["epochs"] = 1

        with self.assertRaises(ValueError):
            enforce_training_config(training, make_experiment(), allow_overrides=False)

    def test_allows_explicit_smoke_override_and_reports_it(self) -> None:
        training = make_training()
        training["epochs"] = 1

        mismatches = enforce_training_config(
            training,
            make_experiment(),
            allow_overrides=True,
        )

        self.assertEqual(mismatches, ["epochs: expected 100, got 1"])


if __name__ == "__main__":
    unittest.main()
