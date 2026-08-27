from __future__ import annotations

import unittest

import torch
from torch import nn

from src.pipelines.waveform_conformer.train_deepship_conformer import (
    validate_trainable_gradients,
)


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


if __name__ == "__main__":
    unittest.main()
