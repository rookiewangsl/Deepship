from __future__ import annotations

import unittest

from src.data.deepship_repeats import compile_repeat_vessel_split
from src.pipelines.mel_ml.train_deepship_macnna_global import (
    GlobalAttentionTrainConfig,
    _validate_g_series_experiment_config,
)


def base_protocol_config() -> dict[str, object]:
    return {
        "experiment_id": "base",
        "features": {"clip_duration_seconds": 3.0},
        "split": {
            "split_seed": 42,
            "target_segments_per_class": {"train": 2, "val": 1, "test": 1},
            "protocols": ["vessel_name_disjoint"],
        },
    }


def rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inventory = []
    identities = []
    for class_name in ("Cargo", "Passenger", "Tank", "Tug"):
        for index in range(6):
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


def repeat_g_config() -> dict[str, object]:
    return {
        "experiment_id": "macnna_global_l20_repeats_v1",
        "features": {
            "clip_duration_seconds": 20.0,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 512,
            "n_mels": 64,
            "highpass_freq": None,
        },
        "shared_adapter": {
            "d_model": 128,
            "num_heads": 4,
            "ffn_expansion": 2,
            "position_kernel_size": 9,
            "temporal_kernel_size": 15,
            "dropout": 0.1,
            "gate_init": -2.0,
        },
        "training": {
            "model_seeds": [42, 43, 44],
            "batch_size": 4,
            "eval_batch_size": 4,
            "epochs": 50,
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "momentum": 0.9,
            "weight_decay": 1e-2,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "min_learning_rate": 1e-6,
            "warmup_epochs": 5,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.005,
            "precision": "bf16",
            "num_workers": 8,
            "training_sampling": "vessel_balanced_dynamic",
            "train_samples_per_epoch": 14000,
        },
    }


class DeepShipRepeatTests(unittest.TestCase):
    def test_repeat_splits_are_deterministic_and_seed_specific(self) -> None:
        inventory, identities = rows()
        first = compile_repeat_vessel_split(
            base_protocol_config(),
            inventory,
            identities,
            [],
            split_seed=43,
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )
        second = compile_repeat_vessel_split(
            base_protocol_config(),
            inventory,
            identities,
            [],
            split_seed=43,
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )
        other = compile_repeat_vessel_split(
            base_protocol_config(),
            inventory,
            identities,
            [],
            split_seed=44,
            source_inventory_sha256="inventory",
            source_identity_sha256="identity",
        )

        self.assertEqual(first[1]["manifest_sha256"], second[1]["manifest_sha256"])
        self.assertNotEqual(first[1]["manifest_sha256"], other[1]["manifest_sha256"])
        self.assertEqual(first[4]["status"], "passed")
        self.assertEqual(other[4]["status"], "passed")
        self.assertEqual(first[4]["vessel_name_groups_crossing_partitions"], [])

    def test_repeat_g_config_accepts_only_frozen_model_seeds(self) -> None:
        valid = GlobalAttentionTrainConfig(
            model_variant="g1",
            clip_duration=20.0,
            training_sampling="vessel_balanced_dynamic",
            train_samples_per_epoch=14000,
            optimizer="adamw",
            learning_rate=3e-4,
            weight_decay=1e-2,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            min_learning_rate=1e-6,
            warmup_epochs=5,
            early_stopping_patience=8,
            early_stopping_min_delta=0.005,
            batch_size=4,
            eval_batch_size=4,
            epochs=50,
            precision="bf16",
            num_workers=8,
            seed=43,
        )
        self.assertEqual(_validate_g_series_experiment_config(valid, repeat_g_config()), [])

        invalid = GlobalAttentionTrainConfig(**{**valid.__dict__, "seed": 45})
        with self.assertRaisesRegex(ValueError, "expected one of.*42, 43, 44"):
            _validate_g_series_experiment_config(invalid, repeat_g_config())


if __name__ == "__main__":
    unittest.main()
