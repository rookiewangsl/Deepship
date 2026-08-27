from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipelines.waveform_conformer.train_deepship_conformer import (
    ConformerTrainConfig,
    TrainingProgress,
    _learning_rate_text,
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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
            )

        lines = output.getvalue().splitlines()
        self.assertTrue(any("batch=0/3 (0.0%)" in line for line in lines))
        self.assertTrue(any("batch=2/3 (66.7%)" in line for line in lines))
        self.assertTrue(any("batch=3/3 (100.0%)" in line for line in lines))
        self.assertTrue(any("avg_loss=" in line for line in lines))
        self.assertTrue(any("lr=head:1.00e-02" in line for line in lines))
        self.assertTrue(any("samples_per_sec=" in line for line in lines))
        self.assertTrue(any("gpu_peak=n/a" in line for line in lines))

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


if __name__ == "__main__":
    unittest.main()
