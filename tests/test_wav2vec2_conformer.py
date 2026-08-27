from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.wav2vec2_conformer import (
    AttentiveStatisticsPooling,
    Wav2Vec2ConformerClassifier,
    feature_lengths_from_attention_mask,
)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.layer_norm = nn.LayerNorm(4)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            conv_kernel=[2],
            conv_stride=[2],
        )
        self.feature_extractor = nn.Linear(1, 1)
        self.encoder = FakeEncoder()

    def forward(self, input_values, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        features = input_values[:, ::2].unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=features)


class FakeCheckpointingBackbone(FakeBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs: dict[str, object] = {}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}

    def forward(self, input_values, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        features = input_values[:, ::2].unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.encoder.layers:
            if self.gradient_checkpointing and self.encoder.training:
                features = checkpoint(
                    layer,
                    features,
                    **self.gradient_checkpointing_kwargs,
                )
            else:
                features = layer(features)
        return SimpleNamespace(last_hidden_state=features)


class Wav2Vec2ConformerTests(unittest.TestCase):
    def test_feature_lengths_follow_convolution_formula(self) -> None:
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )
        lengths = feature_lengths_from_attention_mask(
            attention_mask,
            conv_kernel=[2, 2],
            conv_stride=[2, 2],
        )
        self.assertEqual(lengths.tolist(), [2, 1])

    def test_attentive_pooling_ignores_padding(self) -> None:
        pooling = AttentiveStatisticsPooling(hidden_size=2, attention_size=2)
        hidden = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [1000.0, 1000.0]]])
        mask = torch.tensor([[True, True, False]])
        result = pooling(hidden, mask)
        self.assertEqual(result.shape, (1, 4))
        self.assertLess(float(result[0, 0]), 3.0)

    def test_classifier_accepts_injected_backbone_without_download(self) -> None:
        model = Wav2Vec2ConformerClassifier(
            num_classes=4,
            backbone=FakeBackbone(),
            gradient_checkpointing=False,
            finetuning_mode="full",
            classifier_hidden_size=8,
            pooling_attention_size=4,
        )
        inputs = torch.randn(2, 8)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
        )
        logits = model(inputs, attention_mask=attention_mask)
        self.assertEqual(logits.shape, (2, 4))
        self.assertFalse(model.backbone_config.apply_spec_augment)
        self.assertEqual(model.backbone_config.layerdrop, 0.0)

    def test_frozen_backbone_stays_in_eval_mode_during_head_training(self) -> None:
        model = Wav2Vec2ConformerClassifier(
            num_classes=4,
            backbone=FakeBackbone(),
            gradient_checkpointing=False,
            finetuning_mode="frozen",
            classifier_hidden_size=8,
            pooling_attention_size=4,
        )
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.pooling.training)

    def test_last_n_mode_only_trains_selected_backbone_blocks(self) -> None:
        model = Wav2Vec2ConformerClassifier(
            num_classes=4,
            backbone=FakeBackbone(),
            gradient_checkpointing=False,
            finetuning_mode="last_n",
            train_last_n_layers=1,
            classifier_hidden_size=8,
            pooling_attention_size=4,
        )
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.backbone.encoder.training)
        self.assertFalse(model.backbone.encoder.layers[0].training)
        self.assertTrue(model.backbone.encoder.layers[-1].training)
        self.assertTrue(model.backbone.encoder.layer_norm.training)

    def test_non_reentrant_checkpointing_preserves_last_n_gradients(self) -> None:
        backbone = FakeCheckpointingBackbone()
        model = Wav2Vec2ConformerClassifier(
            num_classes=4,
            backbone=backbone,
            gradient_checkpointing=True,
            finetuning_mode="last_n",
            train_last_n_layers=1,
            classifier_hidden_size=8,
            pooling_attention_size=4,
        )
        model.train()
        logits = model(torch.randn(2, 8), attention_mask=torch.ones(2, 8, dtype=torch.long))
        logits.square().mean().backward()

        self.assertEqual(backbone.gradient_checkpointing_kwargs, {"use_reentrant": False})
        self.assertTrue(model.gradient_checkpointing_enabled)
        for layer in backbone.encoder.layers[:-1]:
            self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))
        for parameter in backbone.encoder.layers[-1].parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
