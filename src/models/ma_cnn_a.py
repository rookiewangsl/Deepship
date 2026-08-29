from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

THREE_BRANCH_KERNEL_SIZES = (8, 16, 32)
BRANCH_INTERMEDIATE_CHANNELS = 32
BRANCH_OUTPUT_CHANNELS = 64
HEAD_CHANNELS = 98
MACNNA_MODEL_VARIANTS = ("g0", "g0_c", "g1")
MACNNAModelVariant = Literal["g0", "g0_c", "g1"]


def post_cnn_time_steps(input_time_steps: int | torch.Tensor) -> int | torch.Tensor:
    """Return MA-CNN-A's pre-pooling time length for a Mel-frame length.

    The first temporal convolution downsamples by two.  The later branch
    temporal convolution and the shared refinement convolution each add one
    position because the original architecture uses even kernels with
    ``padding=kernel_size // 2``.
    """

    if isinstance(input_time_steps, torch.Tensor):
        if bool((input_time_steps <= 0).any()):
            raise ValueError("input_time_steps must be positive")
        return torch.div(input_time_steps, 2, rounding_mode="floor") + 3
    if input_time_steps <= 0:
        raise ValueError("input_time_steps must be positive")
    return input_time_steps // 2 + 3


def feature_time_padding_mask(
    valid_input_time_steps: torch.Tensor,
    *,
    total_input_time_steps: int,
) -> torch.Tensor:
    """Build a post-CNN padding mask from valid Mel-frame counts."""

    if valid_input_time_steps.dim() != 1:
        raise ValueError("valid_input_time_steps must be one-dimensional")
    total_output_steps = int(post_cnn_time_steps(total_input_time_steps))
    valid_output_steps = post_cnn_time_steps(valid_input_time_steps.to(dtype=torch.long))
    assert isinstance(valid_output_steps, torch.Tensor)
    valid_output_steps = valid_output_steps.clamp(min=0, max=total_output_steps)
    positions = torch.arange(total_output_steps, device=valid_output_steps.device)
    return positions.unsqueeze(0) >= valid_output_steps.unsqueeze(1)


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


class MACNNABaseClassifier(nn.Module):
    """Reusable MA-CNN-A feature extractor and unchanged classification head."""

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

    @staticmethod
    def _with_channel_dimension(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected [B,F,T] or [B,1,F,T], got {tuple(x.shape)}")
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the complete pre-pooling MA-CNN-A feature map [B,98,F',T']."""

        x = self._with_channel_dimension(x)
        branch_outputs: list[torch.Tensor] = []
        attention_sources: list[torch.Tensor] = []
        for branch in self.branches:
            time_features, freq_features, output = branch(x)
            attention_sources.extend([time_features, freq_features])
            branch_outputs.append(output)

        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        fused = self.attention(attention_sources, fused)
        fused = self.refine_time(fused)
        return self.refine_freq(fused)

    def classify_features(
        self,
        features: torch.Tensor,
        time_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if time_padding_mask is None:
            pooled = self.pool(features).flatten(1)
        else:
            expected = (features.size(0), features.size(-1))
            if time_padding_mask.shape != expected:
                raise ValueError(
                    "time_padding_mask must match pre-pooling features: "
                    f"expected {expected}, got {tuple(time_padding_mask.shape)}"
                )
            valid = (~time_padding_mask.bool())[:, None, None, :].to(features.dtype)
            denominator = valid.sum(dim=(2, 3)).clamp_min(1.0) * features.size(2)
            pooled = (features * valid).sum(dim=(2, 3)) / denominator
        return self.classifier(pooled)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MACNNAClassifier(MACNNABaseClassifier):
    """Original three-branch MA-CNN-A classifier (G0)."""

    model_variant: MACNNAModelVariant = "g0"

    def forward(
        self,
        x: torch.Tensor,
        time_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classify_features(
            self.extract_features(x),
            time_padding_mask=time_padding_mask,
        )


class FrequencyCoordinateEmbedding(nn.Module):
    """Learned embedding of normalized absolute Mel-frequency coordinates."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, embedding_dim)

    def forward(self, frequency_bins: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")
        coordinates = torch.linspace(-1.0, 1.0, frequency_bins, device=device, dtype=dtype)
        return self.projection(coordinates.view(1, frequency_bins, 1))


def expand_time_padding_mask(
    time_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    frequency_bins: int,
    time_steps: int,
) -> torch.Tensor | None:
    """Copy a [B,T] mask to the shared axial layout [B*F,T]."""

    if time_padding_mask is None:
        return None
    if time_padding_mask.shape != (batch_size, time_steps):
        raise ValueError(
            "time_padding_mask must match the post-CNN feature time axis: "
            f"expected {(batch_size, time_steps)}, got {tuple(time_padding_mask.shape)}"
        )
    mask = time_padding_mask.to(dtype=torch.bool)
    return mask[:, None, :].expand(batch_size, frequency_bins, time_steps).reshape(
        batch_size * frequency_bins,
        time_steps,
    )


class TemporalPositionEncoding(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Temporal position kernel size must be a positive odd integer")
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        positioned = tokens
        if mask is not None:
            positioned = positioned.masked_fill(mask.unsqueeze(-1), 0.0)
        positioned = self.convolution(positioned.transpose(1, 2)).transpose(1, 2)
        positioned = self.dropout(self.activation(positioned))
        if mask is not None:
            positioned = positioned.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens + positioned


class GlobalTemporalMixer(nn.Module):
    """Content-dependent interaction over the complete post-CNN time axis (G1)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.normalization(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=mask,
            need_weights=False,
        )
        attended = self.dropout(attended)
        if mask is not None:
            attended = attended.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens + attended


class LocalTemporalMixer(nn.Module):
    """Parameter-matched local-only temporal mixer for the G0-C control."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(d_model)
        branch_specs = ((3, 1), (5, 2), (7, 3))
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation * (kernel_size // 2),
                    groups=d_model,
                )
                for kernel_size, dilation in branch_specs
            ]
        )
        self.projection = nn.Conv1d(len(branch_specs) * d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.maximum_receptive_field = max(
            1 + (kernel_size - 1) * dilation for kernel_size, dilation in branch_specs
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.normalization(tokens)
        if mask is not None:
            normalized = normalized.masked_fill(mask.unsqueeze(-1), 0.0)
        channels_first = normalized.transpose(1, 2)
        mixed = torch.cat([branch(channels_first) for branch in self.branches], dim=1)
        mixed = self.projection(mixed).transpose(1, 2)
        mixed = self.dropout(self.activation(mixed))
        if mask is not None:
            mixed = mixed.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens + mixed


class TemporalConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Temporal convolution kernel size must be a positive odd integer")
        self.normalization = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.normalization(tokens)
        if mask is not None:
            normalized = normalized.masked_fill(mask.unsqueeze(-1), 0.0)
        convolved = self.depthwise(normalized.transpose(1, 2)).transpose(1, 2)
        convolved = self.dropout(self.activation(convolved))
        if mask is not None:
            convolved = convolved.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens + convolved


class TemporalFeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden_size = d_model * expansion
        self.normalization = nn.LayerNorm(d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        forwarded = self.network(self.normalization(tokens))
        if mask is not None:
            forwarded = forwarded.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens + forwarded


class SharedTemporalAxialBlock(nn.Module):
    def __init__(
        self,
        *,
        variant: Literal["g0_c", "g1"],
        d_model: int,
        num_heads: int,
        ffn_expansion: int,
        temporal_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if variant == "g1":
            self.mixer: nn.Module = GlobalTemporalMixer(d_model, num_heads, dropout)
        elif variant == "g0_c":
            self.mixer = LocalTemporalMixer(d_model, dropout)
        else:
            raise ValueError(f"Unsupported temporal block variant: {variant}")
        self.temporal_convolution = TemporalConvolutionModule(
            d_model,
            temporal_kernel_size,
            dropout,
        )
        self.feed_forward = TemporalFeedForward(d_model, ffn_expansion, dropout)
        self.output_normalization = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        tokens = self.mixer(tokens, mask)
        tokens = self.temporal_convolution(tokens, mask)
        tokens = self.feed_forward(tokens, mask)
        tokens = self.output_normalization(tokens)
        if mask is not None:
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        return tokens


class TemporalAxialAdapter(nn.Module):
    """Add shared time-axis context without averaging frequency before attention."""

    def __init__(
        self,
        *,
        variant: Literal["g0_c", "g1"],
        input_channels: int = HEAD_CHANNELS,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_expansion: int = 2,
        position_kernel_size: int = 9,
        temporal_kernel_size: int = 15,
        dropout: float = 0.1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if ffn_expansion <= 0:
            raise ValueError("ffn_expansion must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.variant = variant
        self.d_model = d_model
        self.input_projection = nn.Conv2d(input_channels, d_model, kernel_size=1)
        self.frequency_embedding = FrequencyCoordinateEmbedding(d_model)
        self.position_encoding = TemporalPositionEncoding(
            d_model,
            position_kernel_size,
            dropout,
        )
        self.temporal_block = SharedTemporalAxialBlock(
            variant=variant,
            d_model=d_model,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.output_projection = nn.Conv2d(d_model, input_channels, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def gate_strength(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def forward(
        self,
        features: torch.Tensor,
        time_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(f"Expected [B,C,F,T] features, got {tuple(features.shape)}")
        projected = self.input_projection(features)
        batch_size, channels, frequency_bins, time_steps = projected.shape
        frequency_embedding = self.frequency_embedding(
            frequency_bins,
            device=projected.device,
            dtype=projected.dtype,
        )
        tokens = projected.permute(0, 2, 3, 1)
        tokens = tokens + frequency_embedding.unsqueeze(2)
        tokens = tokens.reshape(batch_size * frequency_bins, time_steps, channels)
        expanded_mask = expand_time_padding_mask(
            time_padding_mask,
            batch_size=batch_size,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
        )
        tokens = self.position_encoding(tokens, expanded_mask)
        tokens = self.temporal_block(tokens, expanded_mask)
        contextualized = tokens.reshape(batch_size, frequency_bins, time_steps, channels)
        contextualized = contextualized.permute(0, 3, 1, 2).contiguous()
        residual = self.output_projection(contextualized)
        output = features + self.gate_strength * residual
        if time_padding_mask is not None:
            output = output.masked_fill(time_padding_mask[:, None, None, :].bool(), 0.0)
        return output


class MACNNATemporalClassifier(MACNNABaseClassifier):
    """MA-CNN-A plus a local capacity control (G0-C) or global time attention (G1)."""

    def __init__(
        self,
        num_classes: int,
        *,
        model_variant: Literal["g0_c", "g1"],
        d_model: int = 128,
        num_heads: int = 4,
        ffn_expansion: int = 2,
        position_kernel_size: int = 9,
        temporal_kernel_size: int = 15,
        dropout: float = 0.1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__(num_classes)
        self.model_variant = model_variant
        self.temporal_adapter = TemporalAxialAdapter(
            variant=model_variant,
            d_model=d_model,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            position_kernel_size=position_kernel_size,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
            gate_init=gate_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.extract_features(x)
        features = self.temporal_adapter(features, time_padding_mask=time_padding_mask)
        return self.classify_features(features, time_padding_mask=time_padding_mask)


def build_macnna_model(
    num_classes: int,
    *,
    model_variant: MACNNAModelVariant = "g0",
    d_model: int = 128,
    num_heads: int = 4,
    ffn_expansion: int = 2,
    position_kernel_size: int = 9,
    temporal_kernel_size: int = 15,
    dropout: float = 0.1,
    gate_init: float = -2.0,
) -> MACNNABaseClassifier:
    if model_variant == "g0":
        return MACNNAClassifier(num_classes)
    if model_variant not in {"g0_c", "g1"}:
        raise ValueError(
            f"Unsupported model_variant={model_variant!r}; expected one of {MACNNA_MODEL_VARIANTS}"
        )
    return MACNNATemporalClassifier(
        num_classes,
        model_variant=model_variant,
        d_model=d_model,
        num_heads=num_heads,
        ffn_expansion=ffn_expansion,
        position_kernel_size=position_kernel_size,
        temporal_kernel_size=temporal_kernel_size,
        dropout=dropout,
        gate_init=gate_init,
    )
