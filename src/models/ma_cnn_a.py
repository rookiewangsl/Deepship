from __future__ import annotations

import math

import torch
from torch import nn

THREE_BRANCH_KERNEL_SIZES = (8, 16, 32)
BRANCH_INTERMEDIATE_CHANNELS = 32
BRANCH_OUTPUT_CHANNELS = 64
HEAD_CHANNELS = 98


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AsymmetricBranch(nn.Module):
    def __init__(
        self,
        intermediate_channels: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv_time_32 = ConvBNReLU(
            1,
            intermediate_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 2),
            padding=(0, pad),
        )
        self.conv_freq_32 = ConvBNReLU(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(kernel_size, 1),
            stride=(2, 1),
            padding=(pad, 0),
        )
        self.conv_time_64 = ConvBNReLU(
            intermediate_channels,
            output_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, pad),
        )
        self.conv_freq_64 = ConvBNReLU(
            output_channels,
            output_channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(pad, 0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features_32_time = self.conv_time_32(x)
        features_32_freq = self.conv_freq_32(features_32_time)
        features_64_time = self.conv_time_64(features_32_freq)
        features_64_freq = self.conv_freq_64(features_64_time)
        return features_64_time, features_64_freq, features_64_freq


def eca_kernel_size(num_channels: int, gamma: int = 2, b: int = 1) -> int:
    # ECA uses a small odd kernel that grows slowly with channel width.
    value = int(abs((math.log2(num_channels) + b) / gamma))
    return value if value % 2 == 1 else value + 1


class PaperAttentionFusion(nn.Module):
    """Channel reweighting over the fused three-branch feature maps."""

    def __init__(
        self,
        num_channels: int,
        num_blocks: int = len(THREE_BRANCH_KERNEL_SIZES) * 2,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = eca_kernel_size(num_channels)
        padding = kernel_size // 2
        self.weight_generators = nn.ModuleList(
            [
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
                for _ in range(num_blocks)
            ]
        )
        self.activation = nn.Sigmoid()
        self.kernel_size = kernel_size

    def forward(self, feature_maps: list[torch.Tensor], fused_features: torch.Tensor) -> torch.Tensor:
        if len(feature_maps) != len(self.weight_generators):
            raise ValueError("feature_maps and weight_generators must have the same length")

        combined_weights: torch.Tensor | None = None
        for generator, feature_map in zip(self.weight_generators, feature_maps, strict=True):
            weights = self.pool(feature_map).squeeze(-1).transpose(-1, -2)
            weights = self.activation(generator(weights)).transpose(-1, -2).unsqueeze(-1)
            combined_weights = weights if combined_weights is None else combined_weights + weights
        return fused_features * combined_weights


class MACNNAClassifier(nn.Module):
    """Three-branch MA-CNN-A classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                AsymmetricBranch(
                    BRANCH_INTERMEDIATE_CHANNELS,
                    BRANCH_OUTPUT_CHANNELS,
                    kernel_size,
                )
                for kernel_size in THREE_BRANCH_KERNEL_SIZES
            ]
        )
        self.attention = PaperAttentionFusion(
            BRANCH_OUTPUT_CHANNELS,
            num_blocks=len(THREE_BRANCH_KERNEL_SIZES) * 2,
        )
        self.refine_time = ConvBNReLU(
            BRANCH_OUTPUT_CHANNELS,
            HEAD_CHANNELS,
            kernel_size=(1, 8),
            stride=(1, 1),
            padding=(0, 4),
        )
        self.refine_freq = ConvBNReLU(
            HEAD_CHANNELS,
            HEAD_CHANNELS,
            kernel_size=(8, 1),
            stride=(1, 1),
            padding=(4, 0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(HEAD_CHANNELS, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        branch_outputs: list[torch.Tensor] = []
        attention_sources: list[torch.Tensor] = []
        for branch in self.branches:
            time_features, freq_features, output = branch(x)
            attention_sources.extend([time_features, freq_features])
            branch_outputs.append(output)

        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        fused = self.attention(attention_sources, fused)
        fused = self.refine_time(fused)
        fused = self.refine_freq(fused)
        fused = self.pool(fused).flatten(1)
        return self.classifier(fused)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
