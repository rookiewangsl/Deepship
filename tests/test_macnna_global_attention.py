from __future__ import annotations

import io
import unittest

import torch
from torch import nn

from src.models.ma_cnn_a import (
    HEAD_CHANNELS,
    LocalTemporalMixer,
    MACNNAClassifier,
    MACNNATemporalClassifier,
    TemporalAxialAdapter,
    build_macnna_model,
    expand_time_padding_mask,
)


def original_forward(model: MACNNAClassifier, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(1)
    branch_outputs = []
    attention_sources = []
    for branch in model.branches:
        time_features, frequency_features, output = branch(inputs)
        attention_sources.extend([time_features, frequency_features])
        branch_outputs.append(output)
    fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
    fused = model.attention(attention_sources, fused)
    fused = model.refine_time(fused)
    fused = model.refine_freq(fused)
    return model.classifier(model.pool(fused).flatten(1))


class MACNNAGlobalAttentionTests(unittest.TestCase):
    def test_g0_refactor_preserves_original_forward_numerically(self) -> None:
        torch.manual_seed(42)
        model = MACNNAClassifier(num_classes=4).eval()
        inputs = torch.randn(2, 1, 64, 94)

        with torch.no_grad():
            expected = original_forward(model, inputs)
            actual = model(inputs)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(model.num_parameters, 532_166)

    def test_all_variants_support_multiple_batch_and_time_shapes(self) -> None:
        for variant in ("g0", "g0_c", "g1"):
            model = build_macnna_model(4, model_variant=variant, dropout=0.0).eval()
            for shape in ((1, 64, 63), (2, 1, 64, 94)):
                with self.subTest(variant=variant, shape=shape), torch.no_grad():
                    output = model(torch.randn(*shape))
                    self.assertEqual(output.shape, (shape[0], 4))
                    self.assertTrue(torch.isfinite(output).all())

    def test_capacity_control_matches_g1_parameter_budget_without_attention(self) -> None:
        g0 = build_macnna_model(4, model_variant="g0")
        g0_c = build_macnna_model(4, model_variant="g0_c")
        g1 = build_macnna_model(4, model_variant="g1")
        added_control = g0_c.num_parameters - g0.num_parameters
        added_attention = g1.num_parameters - g0.num_parameters

        self.assertLessEqual(abs(added_control - added_attention) / added_attention, 0.10)
        self.assertFalse(any(isinstance(module, nn.MultiheadAttention) for module in g0_c.modules()))
        self.assertTrue(any(isinstance(module, nn.MultiheadAttention) for module in g1.modules()))
        mixer = g0_c.temporal_adapter.temporal_block.mixer
        self.assertIsInstance(mixer, LocalTemporalMixer)
        self.assertLess(mixer.maximum_receptive_field, 50)

    def test_padding_mask_expands_to_every_frequency_position(self) -> None:
        mask = torch.tensor([[False, True, True], [False, False, True]])
        expanded = expand_time_padding_mask(
            mask,
            batch_size=2,
            frequency_bins=3,
            time_steps=3,
        )

        self.assertIsNotNone(expanded)
        self.assertEqual(expanded.shape, (6, 3))
        torch.testing.assert_close(expanded[:3], mask[0].expand(3, 3))
        torch.testing.assert_close(expanded[3:], mask[1].expand(3, 3))

    def test_masked_values_cannot_change_unmasked_g1_outputs(self) -> None:
        torch.manual_seed(7)
        adapter = TemporalAxialAdapter(variant="g1", dropout=0.0).eval()
        features = torch.randn(1, HEAD_CHANNELS, 4, 12)
        changed = features.clone()
        changed[..., 8:] = torch.randn_like(changed[..., 8:]) * 1000
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[:, 8:] = True

        with torch.no_grad():
            reference = adapter(features, mask)
            comparison = adapter(changed, mask)

        torch.testing.assert_close(reference[..., :8], comparison[..., :8])
        self.assertEqual(torch.count_nonzero(reference[..., 8:]).item(), 0)
        self.assertEqual(torch.count_nonzero(comparison[..., 8:]).item(), 0)

    def test_narrow_frequency_change_is_not_averaged_before_temporal_block(self) -> None:
        torch.manual_seed(11)
        adapter = TemporalAxialAdapter(variant="g1", dropout=0.0).eval()
        baseline = torch.zeros(1, HEAD_CHANNELS, 5, 10)
        changed = baseline.clone()
        changed[:, :, 2, 4] = 1.0
        captured: list[torch.Tensor] = []

        def capture_tokens(_module, arguments):
            captured.append(arguments[0].detach().clone())

        handle = adapter.temporal_block.register_forward_pre_hook(capture_tokens)
        try:
            with torch.no_grad():
                adapter(baseline)
                adapter(changed)
        finally:
            handle.remove()

        difference = (captured[1] - captured[0]).abs().sum(dim=(1, 2))
        changed_rows = torch.nonzero(difference > 0, as_tuple=False).flatten().tolist()
        self.assertEqual(changed_rows, [2])

    def test_gate_starts_near_g0_but_attention_receives_finite_gradients(self) -> None:
        torch.manual_seed(19)
        adapter = TemporalAxialAdapter(variant="g1", dropout=0.0, gate_init=-2.0)
        features = torch.randn(2, HEAD_CHANNELS, 4, 10, requires_grad=True)
        output = adapter(features)
        relative_change = (output - features).norm() / features.norm()

        self.assertAlmostEqual(float(adapter.gate_strength.detach()), 0.1192029, places=5)
        self.assertLess(float(relative_change.detach()), 0.5)
        output.square().mean().backward()
        attention_gradient = (
            adapter.temporal_block.mixer.attention.in_proj_weight.grad
        )
        self.assertIsNotNone(attention_gradient)
        self.assertTrue(torch.isfinite(attention_gradient).all())
        self.assertGreater(float(attention_gradient.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(adapter.gate.grad).all())

    def test_g1_model_and_optimizer_state_resume_exactly(self) -> None:
        torch.manual_seed(23)
        model = MACNNATemporalClassifier(4, model_variant="g1", dropout=0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        inputs = torch.randn(1, 1, 32, 48)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        checkpoint = io.BytesIO()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint,
        )

        checkpoint.seek(0)
        payload = torch.load(checkpoint, weights_only=False)
        restored = MACNNATemporalClassifier(4, model_variant="g1", dropout=0.0).eval()
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9)
        restored.load_state_dict(payload["model"])
        restored_optimizer.load_state_dict(payload["optimizer"])
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored(inputs), rtol=0.0, atol=0.0)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
