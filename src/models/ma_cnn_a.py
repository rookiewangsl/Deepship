from __future__ import annotations

import torch
from torch import nn

THREE_BRANCH_KERNEL_SIZES = (8, 16, 32)


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
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv_time = ConvBNReLU(
            1,
            channels,
            kernel_size=(1, kernel_size),
            stride=(1, 2),
            padding=(0, pad),
        )
        self.conv_freq = ConvBNReLU(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=(2, 1),
            padding=(pad, 0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_features = self.conv_time(x)
        branch_output = self.conv_freq(time_features)
        return time_features, branch_output, branch_output


class PaperAttentionFusion(nn.Module):
    """Channel reweighting over the fused three-branch feature maps."""

    def __init__(self, num_blocks: int = len(THREE_BRANCH_KERNEL_SIZES) * 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weight_generators = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(num_blocks)]
        )
        self.activation = nn.Sigmoid()

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

    def __init__(
        self,
        num_classes: int,
        branch_channels: int = 88,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                AsymmetricBranch(branch_channels, kernel_size)
                for kernel_size in THREE_BRANCH_KERNEL_SIZES
            ]
        )
        self.attention = PaperAttentionFusion(num_blocks=len(THREE_BRANCH_KERNEL_SIZES) * 2)
        self.refine_time = ConvBNReLU(
            branch_channels,
            branch_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )
        self.refine_freq = ConvBNReLU(
            branch_channels,
            branch_channels,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(branch_channels, num_classes)
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
