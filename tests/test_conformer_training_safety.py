from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.deepship import SegmentRecord
from src.data.deepship_waveform import (
    DeepShipWaveformSegmentDataset,
    RecordingBalancedEpochSampler,
)
from src.pipelines.waveform_conformer.train_deepship_conformer import (
    ConformerTrainConfig,
    TrainingProgress,
    _learning_rate_text,
    build_dataloaders,
    build_scheduler,
    run_epoch,
    validate_config,
    validate_trainable_gradients,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, input_values, attention_mask=None):
        del attention_mask
        return self.classifier(input_values)


class ConformerTrainingSafetyTests(unittest.TestCase):
    def test_rejects_trainable_parameters_without_gradients(self) -> None:
        model = nn.Linear(3, 2)
        with self.assertRaisesRegex(RuntimeError, "did not receive gradients"):
            validate_trainable_gradients(model)

    def test_accepts_finite_gradients(self) -> None:
        model = nn.Linear(3, 2)
        model(torch.randn(2, 3)).square().mean().backward()
        validate_trainable_gradients(model)

    def test_rejects_nonfinite_gradients(self) -> None:
        model = nn.Linear(3, 2)
        model(torch.randn(2, 3)).square().mean().backward()
        next(model.parameters()).grad.fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradients"):
            validate_trainable_gradients(model)

    def test_reports_batch_progress_and_final_partial_interval(self) -> None:
        inputs = torch.randn(6, 3)
        masks = torch.ones(6, 3, dtype=torch.long)
        targets = torch.tensor([0, 1, 0, 1, 0, 1])
        dataloader = DataLoader(TensorDataset(inputs, masks, targets), batch_size=2)
        model = TinyClassifier()
        optimizer = torch.optim.SGD(
            [{"name": "head", "params": model.parameters(), "lr": 0.01}]
        )
        config = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            batch_size=2,
            epochs=4,
            log_interval=2,
        )

        output = StringIO()
        progress = TrainingProgress(None)
        scheduler, _ = build_scheduler(
            optimizer,
            config,
            total_optimizer_steps=4,
        )
        runtime_stats: dict[str, float] = {}
        with redirect_stdout(output):
            run_epoch(
                model,
                dataloader,
                nn.CrossEntropyLoss(),
                config,
                epoch=1,
                phase="train",
                progress=progress,
                learning_rates="head:1.00e-02",
                optimizer=optimizer,
                scheduler=scheduler,
                runtime_stats=runtime_stats,
            )

        lines = output.getvalue().splitlines()
        self.assertTrue(any("batch=0/3 (0.0%)" in line for line in lines))
        self.assertTrue(any("batch=2/3 (66.7%)" in line for line in lines))
        self.assertTrue(any("batch=3/3 (100.0%)" in line for line in lines))
        self.assertTrue(any("avg_loss=" in line for line in lines))
        self.assertTrue(any("lr=head:" in line for line in lines))
        self.assertTrue(any("samples_per_sec=" in line for line in lines))
        self.assertTrue(any("gpu_peak=n/a" in line for line in lines))
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertGreater(runtime_stats["phase_seconds"], 0.0)
        self.assertGreaterEqual(runtime_stats["data_wait_seconds"], 0.0)
        self.assertGreater(runtime_stats["samples_per_second"], 0.0)

    def test_interactive_progress_overwrites_and_logs_only_epoch_summary(self) -> None:
        terminal = StringIO()
        progress = TrainingProgress(terminal)
        output = StringIO()

        with redirect_stdout(output):
            progress.update("a longer progress line")
            progress.update("short")
            progress.finish_epoch("Epoch 1/4 | done | val_acc=0.5000")
        progress.close()

        terminal_output = terminal.getvalue()
        self.assertIn("\ra longer progress line", terminal_output)
        self.assertIn("\rshort", terminal_output)
        self.assertNotIn("\n", terminal_output)
        self.assertEqual(output.getvalue(), "Epoch 1/4 | done | val_acc=0.5000\n")

    def test_interactive_progress_truncates_before_terminal_wrap(self) -> None:
        terminal = StringIO()
        progress = TrainingProgress(terminal, terminal_width=20)

        progress.update("Epoch 1/50 | train | batch=100/14000")

        rendered = terminal.getvalue().removeprefix("\r").removesuffix("\x1b[K")
        self.assertEqual(len(rendered), 20)
        self.assertTrue(rendered.endswith("…"))

    def test_formats_frozen_and_partial_finetuning_learning_rates(self) -> None:
        head = nn.Parameter(torch.zeros(1))
        frozen_optimizer = torch.optim.SGD(
            [{"name": "head", "params": [head], "lr": 3e-5}]
        )
        self.assertEqual(_learning_rate_text(frozen_optimizer), "head:3.00e-05")

        encoder = nn.Parameter(torch.zeros(1))
        partial_optimizer = torch.optim.SGD(
            [
                {"name": "encoder", "params": [encoder], "lr": 1e-6},
                {"name": "head", "params": [head], "lr": 3e-5},
            ]
        )
        self.assertEqual(
            _learning_rate_text(partial_optimizer),
            "enc:1.00e-06,head:3.00e-05",
        )

    def test_rejects_nonpositive_log_interval(self) -> None:
        config = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            log_interval=0,
        )
        with self.assertRaisesRegex(ValueError, "log_interval must be positive"):
            validate_config(config)

    def test_rejects_invalid_dataloader_parallelism(self) -> None:
        negative_workers = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            num_workers=-1,
        )
        with self.assertRaisesRegex(ValueError, "num_workers must be non-negative"):
            validate_config(negative_workers)

        zero_prefetch = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            prefetch_factor=0,
        )
        with self.assertRaisesRegex(ValueError, "prefetch_factor must be positive"):
            validate_config(zero_prefetch)

    def test_rejects_invalid_sampling_and_early_stopping_configuration(self) -> None:
        invalid_sampling = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            training_sampling="vessel_balanced_dynamic",
        )
        with self.assertRaisesRegex(ValueError, "training_sampling"):
            validate_config(invalid_sampling)

        fixed_with_budget = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            train_samples_per_epoch=100,
        )
        with self.assertRaisesRegex(ValueError, "only supported for dynamic"):
            validate_config(fixed_with_budget)

        invalid_eval_batch = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            eval_batch_size=0,
        )
        with self.assertRaisesRegex(ValueError, "eval_batch_size"):
            validate_config(invalid_eval_batch)

        invalid_min_delta = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            early_stopping_min_delta=-0.01,
        )
        with self.assertRaisesRegex(ValueError, "early_stopping_min_delta"):
            validate_config(invalid_min_delta)

    def test_dynamic_training_keeps_fixed_evaluation_and_separate_batch_size(self) -> None:
        cargo_anchor_a = SegmentRecord(
            relative_path="Cargo/a.wav",
            class_name="Cargo",
            label_index=0,
            start_frame=0,
            num_frames=30,
            sample_rate=10,
            segment_index=0,
            total_segments=2,
            group_key="cargo-vessel",
            vessel_key="cargo-vessel",
        )
        cargo_anchor_b = SegmentRecord(
            **{
                **cargo_anchor_a.__dict__,
                "start_frame": 30,
                "segment_index": 1,
            }
        )
        passenger = SegmentRecord(
            relative_path="Passenger/b.wav",
            class_name="Passenger",
            label_index=1,
            start_frame=0,
            num_frames=30,
            sample_rate=10,
            segment_index=0,
            total_segments=1,
            group_key="passenger-vessel",
            vessel_key="passenger-vessel",
        )
        split_segments = {
            "train": [cargo_anchor_a, cargo_anchor_b, passenger],
            "val": [cargo_anchor_a],
            "test": [passenger],
        }
        split_report = {
            "protocol": "vessel_name_disjoint",
            "window_rule": "fixed manifest anchor",
            "manifest_sha256": "unused",
        }
        config = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            sample_rate=16000,
            training_sampling="recording_balanced_dynamic",
            train_samples_per_epoch=8,
            batch_size=1,
            eval_batch_size=2,
            num_workers=0,
        )

        with patch(
            "src.pipelines.waveform_conformer.train_deepship_conformer."
            "load_and_validate_split",
            return_value=(split_segments, split_report),
        ):
            dataloaders, report = build_dataloaders(config)

        self.assertIsInstance(
            dataloaders["train"].sampler,
            RecordingBalancedEpochSampler,
        )
        self.assertTrue(dataloaders["train"].dataset.dynamic_crop)
        self.assertEqual(len(dataloaders["train"].dataset), 2)
        self.assertFalse(dataloaders["val"].dataset.dynamic_crop)
        self.assertTrue(dataloaders["val"].dataset.return_index)
        self.assertEqual(dataloaders["val"].batch_size, 2)
        self.assertEqual(report["training_sampling"]["id"], "S1")
        self.assertEqual(report["batch_sizes"]["validation"], 2)

    def test_step_scheduler_warms_up_then_cosine_decays(self) -> None:
        encoder = nn.Parameter(torch.zeros(1))
        head = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW(
            [
                {"name": "encoder", "params": [encoder], "lr": 5e-6},
                {"name": "head", "params": [head], "lr": 1e-4},
            ]
        )
        config = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            encoder_learning_rate=5e-6,
            head_learning_rate=1e-4,
            min_learning_rate=1e-6,
            warmup_ratio=0.1,
            warmup_start_factor=0.1,
        )
        scheduler, report = build_scheduler(
            optimizer,
            config,
            total_optimizer_steps=20,
        )

        self.assertEqual(report["warmup_steps"], 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-7)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-5)

        learning_rates = []
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            learning_rates.append([group["lr"] for group in optimizer.param_groups])

        self.assertAlmostEqual(learning_rates[1][0], 5e-6)
        self.assertAlmostEqual(learning_rates[1][1], 1e-4)
        self.assertAlmostEqual(learning_rates[-1][0], 1e-6)
        self.assertAlmostEqual(learning_rates[-1][1], 1e-6)

    def test_rejects_invalid_step_schedule_configuration(self) -> None:
        invalid_warmup = ConformerTrainConfig(
            split_manifest="unused.json",
            device="cpu",
            precision="fp32",
            warmup_ratio=1.0,
        )
        with self.assertRaisesRegex(ValueError, "warmup_ratio"):
            validate_config(invalid_warmup)

    def test_validation_collects_segment_predictions_without_second_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_root = Path(temporary_dir)
            audio_path = data_root / "Cargo" / "example.wav"
            audio_path.parent.mkdir(parents=True)
            sf.write(audio_path, np.asarray([0.1, 0.2, 0.3], dtype=np.float32), 8)
            segment = SegmentRecord(
                relative_path="Cargo/example.wav",
                class_name="Cargo",
                label_index=0,
                start_frame=0,
                num_frames=3,
                sample_rate=8,
                segment_index=0,
                total_segments=1,
                group_key="recording-1",
                vessel_key="vessel-1",
            )
            dataset = DeepShipWaveformSegmentDataset(
                [segment],
                data_root=data_root,
                sample_rate=8,
                clip_duration=0.375,
                return_index=True,
            )
            dataloader = DataLoader(dataset, batch_size=1)
            config = ConformerTrainConfig(
                split_manifest="unused.json",
                device="cpu",
                precision="fp32",
                epochs=1,
            )
            prediction_rows: list[dict[str, object]] = []

            run_epoch(
                TinyClassifier(),
                dataloader,
                nn.CrossEntropyLoss(),
                config,
                epoch=1,
                phase="val",
                progress=TrainingProgress(None),
                learning_rates="head:1.00e-02",
                prediction_rows=prediction_rows,
            )

            self.assertEqual(len(prediction_rows), 1)
            self.assertEqual(prediction_rows[0]["relative_path"], "Cargo/example.wav")
            self.assertEqual(prediction_rows[0]["vessel_key"], "vessel-1")
            self.assertEqual(len(prediction_rows[0]["probabilities"]), 2)


if __name__ == "__main__":
    unittest.main()
