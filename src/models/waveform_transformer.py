from __future__ import annotations

import math

import torch
from torch import nn


class SpectrogramPatchEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 96,
        patch_size: tuple[int, int] = (32, 8),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class STFTTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int] = (513, 79),
        patch_size: tuple[int, int] = (32, 8),
        embed_dim: int = 96,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        padded_h = math.ceil(input_size[0] / patch_size[0]) * patch_size[0]
        padded_w = math.ceil(input_size[1] / patch_size[1]) * patch_size[1]
        self.padded_size = (padded_h, padded_w)
        self.num_patches = (padded_h // patch_size[0]) * (padded_w // patch_size[1])

        self.patch_embed = SpectrogramPatchEmbedding(
            input_channels=1,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        pad_h = self.padded_size[0] - x.size(-2)
        pad_w = self.padded_size[1] - x.size(-1)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))
