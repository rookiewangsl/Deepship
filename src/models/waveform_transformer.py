from __future__ import annotations

import math
from pathlib import Path

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


class STFTMAEPretrainer(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (128, 128),
        patch_size: tuple[int, int] = (32, 8),
        embed_dim: int = 96,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        decoder_embed_dim: int = 64,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        padded_h = math.ceil(input_size[0] / patch_size[0]) * patch_size[0]
        padded_w = math.ceil(input_size[1] / patch_size[1]) * patch_size[1]
        self.padded_size = (padded_h, padded_w)
        self.num_patches = (padded_h // patch_size[0]) * (padded_w // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1]

        self.patch_embed = SpectrogramPatchEmbedding(
            input_channels=1,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_pos_drop = nn.Dropout(dropout)
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
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(decoder_embed_dim),
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)
        pad_h = self.padded_size[0] - imgs.size(-2)
        pad_w = self.padded_size[1] - imgs.size(-1)
        if pad_h > 0 or pad_w > 0:
            imgs = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h))
        ph, pw = self.patch_size
        h = self.padded_size[0] // ph
        w = self.padded_size[1] // pw
        x = imgs.reshape(imgs.shape[0], 1, h, ph, w, pw)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, ph * pw)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = x.shape
        keep = int(num_tokens * (1.0 - mask_ratio))
        noise = torch.rand(batch_size, num_tokens, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
        )
        mask = torch.ones(batch_size, num_tokens, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor | None = None,
        mask_ratio: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(1)
        pad_h = self.padded_size[0] - noisy.size(-2)
        pad_w = self.padded_size[1] - noisy.size(-1)
        if pad_h > 0 or pad_w > 0:
            noisy = torch.nn.functional.pad(noisy, (0, pad_w, 0, pad_h))
        x = self.patch_embed(noisy)
        x = self.encoder_pos_drop(x + self.pos_embed)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = self.encoder(x)
        x = self.encoder_norm(x)

        x = self.decoder_embed(x)
        batch_size, n_visible, dim = x.shape
        n_total = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(batch_size, n_total - n_visible, -1)
        tokens = torch.cat([x, mask_tokens], dim=1)
        tokens = torch.gather(
            tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
        )
        tokens = tokens + self.decoder_pos_embed
        tokens = self.decoder(tokens)
        pred = self.decoder_pred(self.decoder_norm(tokens))

        target = self.patchify(clean if clean is not None else noisy)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, pred, mask


def load_mae_encoder_weights(
    model: STFTTransformerClassifier,
    checkpoint_path: str | Path,
    strict_positional: bool = True,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = checkpoint.get("encoder_state_dict", checkpoint)
    model_state = model.state_dict()
    loaded = 0

    for key, value in encoder_state.items():
        if key == "pos_embed":
            if strict_positional and value.shape[1] == model.pos_embed.shape[1] - 1:
                model_state["pos_embed"][:, 1:, :] = value
                loaded += 1
            continue
        target_key = key
        if key.startswith("encoder_norm."):
            target_key = key.replace("encoder_norm.", "norm.")
        if target_key in model_state and model_state[target_key].shape == value.shape:
            model_state[target_key] = value
            loaded += 1

    model.load_state_dict(model_state)
    return loaded
