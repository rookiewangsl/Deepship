from __future__ import annotations

import math

import torch
from torch import nn


class ECABlock(nn.Module):
    """Efficient channel attention with adaptive 1D kernel size."""

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0) -> None:
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        kernel_size = max(kernel_size, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = self.activation(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AsymmetricConvBranch(nn.Module):
    """Approximate kxk receptive fields with 1xk then kx1 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.branch = nn.Sequential(
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
            ),
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class MultiScaleAsymmetricBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        fused_channels: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                AsymmetricConvBranch(in_channels, branch_channels, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        fused_in = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_in, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(multi_scale, dim=1))


class MACNNAClassifier(nn.Module):
    """Lightweight multi-scale asymmetric CNN with ECA for DeepShip."""

    def __init__(
        self,
        num_classes: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        stem_channels: int = 32,
        branch_channels: int = 24,
        fused_channels: int = 96,
        classifier_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(1, stem_channels, kernel_size=(3, 3), padding=(1, 1)),
            ConvNormAct(stem_channels, stem_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
        )
        self.multi_scale = MultiScaleAsymmetricBlock(
            in_channels=stem_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            fused_channels=fused_channels,
            dropout=dropout,
        )
        self.refine = nn.Sequential(
            ConvNormAct(fused_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
        )
        self.eca = ECABlock(fused_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(fused_channels, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.multi_scale(x)
        x = self.refine(x)
        x = self.eca(x)
        x = self.pool(x)
        return self.classifier(x)
