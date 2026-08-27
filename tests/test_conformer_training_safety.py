from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipelines.waveform_conformer.train_deepship_conformer import (
    ConformerTrainConfig,
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
        with redirect_stdout(output):
            run_epoch(
                model,
                dataloader,
                nn.CrossEntropyLoss(),
                config,
                epoch=1,
                phase="train",
                optimizer=optimizer,
            )

        lines = output.getvalue().splitlines()
        self.assertIn("Epoch 1/4 | train | start | batches=3 | batch_size=2", lines)
        self.assertTrue(any("batch=2/3 (66.7%)" in line for line in lines))
        self.assertTrue(any("batch=3/3 (100.0%)" in line for line in lines))
        self.assertTrue(any("avg_loss=" in line for line in lines))
        self.assertTrue(any("samples_per_sec=" in line for line in lines))
        self.assertTrue(any("gpu_peak_allocated=n/a" in line for line in lines))

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
