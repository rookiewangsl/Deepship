from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer conv block with optional Dropout2d after MaxPool.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    drop_rate : float
        Spatial dropout rate applied after MaxPool.
        Set to 0.0 to disable (default).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if drop_rate > 0.0:
            layers.append(nn.Dropout2d(p=drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MelCNNClassifier(nn.Module):
    """Mel-spectrogram CNN classifier with configurable depth and regularisation.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    channels : tuple[int, ...]
        Output channels for each ConvBlock.  Length determines network depth.
    drop_rates : tuple[float, ...] | float
        Dropout2d rate after each ConvBlock.  A single float is broadcast
        to every block.
    classifier_dropout : float
        Dropout rate before the fully-connected head.
    classifier_hidden : int | None
        If given, insert a hidden ``Linear → ReLU → Dropout(0.2)`` layer
        of this width before the final projection.  Set to ``None`` for a
        single-layer head.

    Default values reproduce the original 4-block architecture used for
    ShipsEar (backward compatible).
    """

    def __init__(
        self,
        num_classes: int = 5,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        drop_rates: tuple[float, ...] | float = 0.0,
        classifier_dropout: float = 0.3,
        classifier_hidden: int | None = 64,
    ) -> None:
        super().__init__()
        if isinstance(drop_rates, (int, float)):
            drop_rates = tuple(drop_rates for _ in channels)

        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch, dr in zip(channels, drop_rates):
            blocks.append(ConvBlock(in_ch, out_ch, drop_rate=dr))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = channels[-1]
        head: list[nn.Module] = [
            nn.Flatten(),
            nn.Dropout(p=classifier_dropout),
        ]
        if classifier_hidden is not None:
            head.extend([
                nn.Linear(feat_dim, classifier_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(classifier_hidden, num_classes),
            ])
        else:
            head.append(nn.Linear(feat_dim, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
