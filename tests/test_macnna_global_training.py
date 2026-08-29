from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.deepship import CLASS_NAMES, DeepShipMelSegmentDataset, SegmentRecord
from src.pipelines.mel_ml import train_deepship_macnna_global as pipeline


def segments() -> list[SegmentRecord]:
    return [
        SegmentRecord(
            relative_path=f"{class_name}/{index}.wav",
            class_name=class_name,
            label_index=index,
            start_frame=0,
            num_frames=48_000,
            sample_rate=16_000,
            segment_index=0,
            total_segments=1,
            group_key=f"recording-{index}",
            vessel_key=f"vessel-{index}",
        )
        for index, class_name in enumerate(CLASS_NAMES)
    ]


class SyntheticMelDataset(DeepShipMelSegmentDataset):
    def __init__(self, rows: list[SegmentRecord], *, return_index: bool) -> None:
        self.segments = rows
        self.return_index = return_index
        self.read_count = 0

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int):
        self.read_count += 1
        row = self.segments[index]
        inputs = torch.full((1, 16, 20), float(index + 1) / 4.0)
        if self.return_index:
            return inputs, row.label_index, index
        return inputs, row.label_index


class TinyMACNNA(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.model_variant = variant
        self.classifier = nn.Linear(1, 4)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def synthetic_dataloaders(
    config: pipeline.GlobalAttentionTrainConfig,
    **_kwargs,
):
    rows = segments()
    train = SyntheticMelDataset(rows, return_index=False)
    val = SyntheticMelDataset(rows, return_index=True)
    test = SyntheticMelDataset(rows, return_index=True)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(train, batch_size=2, shuffle=True, generator=generator),
        "val": DataLoader(val, batch_size=2, shuffle=False),
        "test": DataLoader(test, batch_size=2, shuffle=False),
    }
    report = {
        "source": "synthetic",
        "protocol": "vessel_name_disjoint",
        "manifest_sha256": "synthetic-manifest",
    }
    return loaders, report


def tiny_factory(num_classes: int, **kwargs):
    if num_classes != 4:
        raise AssertionError("Unexpected class count")
    return TinyMACNNA(str(kwargs["model_variant"]))


def clean_environment() -> dict[str, object]:
    return {
        "python": "test",
        "platform": "test",
        "torch": torch.__version__,
        "numpy": "test",
        "cuda_available": False,
        "torch_cuda_version": None,
        "cuda_devices": [],
        "mps_available": False,
        "git_commit": "test",
        "git_worktree_dirty": False,
        "git_status": [],
    }


def synthetic_g_series_experiment() -> dict[str, object]:
    return {
        "experiment_id": "macnna_global_v1",
        "features": {
            "clip_duration_seconds": 3.0,
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
            "seed": 42,
            "batch_size": 2,
            "eval_batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "min_learning_rate": 1e-5,
            "warmup_epochs": 0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.005,
            "precision": "fp32",
            "num_workers": 0,
        },
        "variants": {
            variant: {"expected_num_parameters": 8}
            for variant in ("g0", "g0_c", "g1")
        },
    }


class MACNNAGlobalTrainingTests(unittest.TestCase):
    def config(self, output_root: str, **overrides) -> pipeline.GlobalAttentionTrainConfig:
        values = dict(
            data_root="unused",
            output_root=output_root,
            split_manifest=None,
            experiment_config=None,
            protocol_name="vessel_name_disjoint",
            model_variant="g1",
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            warmup_epochs=0,
            early_stopping_patience=2,
            precision="fp32",
            num_workers=0,
            log_interval=1,
            allow_experiment_overrides=True,
            device="cpu",
        )
        values.update(overrides)
        return pipeline.GlobalAttentionTrainConfig(**values)

    def common_patches(self):
        return (
            patch.object(pipeline, "build_dataloaders", side_effect=synthetic_dataloaders),
            patch.object(pipeline, "build_macnna_model", side_effect=tiny_factory),
            patch.object(pipeline, "runtime_environment", side_effect=clean_environment),
            patch.object(pipeline, "plot_training_curves"),
            patch.object(pipeline, "plot_confusion_matrix"),
            patch.object(
                pipeline,
                "load_g_series_experiment_config",
                side_effect=lambda _path: synthetic_g_series_experiment(),
            ),
        )

    def test_validation_only_training_never_iterates_test_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            used_loaders = {}

            def build_and_capture(config, **_kwargs):
                loaders, report = synthetic_dataloaders(config)
                used_loaders.update(loaders)
                return loaders, report

            with patch.object(
                pipeline,
                "build_dataloaders",
                side_effect=build_and_capture,
            ), patch.object(
                pipeline,
                "build_macnna_model",
                side_effect=tiny_factory,
            ), patch.object(
                pipeline,
                "runtime_environment",
                side_effect=clean_environment,
            ), patch.object(pipeline, "plot_training_curves"), patch.object(
                pipeline,
                "plot_confusion_matrix",
            ), patch.object(
                pipeline,
                "load_g_series_experiment_config",
                side_effect=lambda _path: synthetic_g_series_experiment(),
            ):
                result = pipeline.train(self.config(directory))

            self.assertEqual(result["status"], "validation_complete")
            self.assertFalse(result["test_evaluated"])
            self.assertGreater(used_loaders["val"].dataset.read_count, 0)
            self.assertEqual(used_loaders["test"].dataset.read_count, 0)
            prediction_names = {
                path.name for path in (Path(directory) / "predictions").glob("*.csv")
            }
            self.assertTrue(prediction_names)
            self.assertTrue(all(name.startswith("validation_best_") for name in prediction_names))

    def test_resume_continues_with_same_variant_and_selection_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2)
            original_run_epoch = pipeline.run_epoch

            def interrupt_on_second_epoch(*args, **kwargs):
                if kwargs["epoch"] == 2 and kwargs["phase"] == "train":
                    raise KeyboardInterrupt("synthetic interruption")
                return original_run_epoch(*args, **kwargs)

            patches = self.common_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patch.object(
                pipeline,
                "run_epoch",
                side_effect=interrupt_on_second_epoch,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.train(config)

            checkpoint = Path(directory) / "models" / "deepship_macnna_global_last.pt"
            self.assertTrue(checkpoint.is_file())
            resume_config = self.config(directory, epochs=2, resume=True)
            patches = self.common_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                result = pipeline.train(resume_config)
            self.assertEqual(result["status"], "validation_complete")
            self.assertTrue((Path(directory) / "reports" / "run_complete.json").is_file())

    def test_rejects_test_evaluation_and_invalid_attention_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation-only"):
            pipeline.validate_config(
                pipeline.GlobalAttentionTrainConfig(evaluate_test_on_completion=True)
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            pipeline.validate_config(
                pipeline.GlobalAttentionTrainConfig(
                    attention_d_model=130,
                    attention_num_heads=4,
                )
            )
        with self.assertRaisesRegex(ValueError, "positive train_samples_per_epoch"):
            pipeline.validate_config(
                pipeline.GlobalAttentionTrainConfig(
                    training_sampling="vessel_balanced_dynamic",
                )
            )

    def test_long_context_optimizer_and_accumulation_configuration(self) -> None:
        config = pipeline.GlobalAttentionTrainConfig(
            training_sampling="vessel_balanced_dynamic",
            train_samples_per_epoch=14_000,
            clip_duration=20.0,
            optimizer="adamw",
            learning_rate=3e-4,
            weight_decay=1e-2,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
        )
        pipeline.validate_config(config)
        model = nn.Linear(4, 4)
        optimizer = pipeline._build_optimizer(model, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-2)

    def test_gradient_accumulation_steps_on_full_and_partial_windows(self) -> None:
        model = TinyMACNNA("g0")
        dataset = SyntheticMelDataset(segments(), return_index=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        config = self.config(
            "unused",
            model_variant="g0",
            batch_size=1,
            gradient_accumulation_steps=3,
            max_grad_norm=1.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            loss, accuracy, rows = pipeline.run_epoch(
                model,
                loader,
                nn.CrossEntropyLoss(),
                config,
                epoch=1,
                phase="train",
                progress=pipeline.TrainingProgress(None),
                optimizer=optimizer,
            )

        self.assertEqual(step.call_count, 2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertEqual(rows, [])
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

    def test_progress_truncates_narrow_terminal_without_wrapping(self) -> None:
        terminal = StringIO()
        progress = pipeline.TrainingProgress(terminal, terminal_width=24)
        progress.update("Epoch 1/100 | train | batch=100/875")
        rendered = terminal.getvalue().removeprefix("\r").removesuffix("\x1b[K")
        self.assertEqual(len(rendered), 24)
        self.assertTrue(rendered.endswith("…"))


if __name__ == "__main__":
    unittest.main()
