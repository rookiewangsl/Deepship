from __future__ import annotations

from typing import Any

import torch
from torch import nn


DEFAULT_PRETRAINED_MODEL = "facebook/wav2vec2-conformer-rel-pos-large"


def feature_lengths_from_attention_mask(
    attention_mask: torch.Tensor,
    conv_kernel: list[int] | tuple[int, ...],
    conv_stride: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Compute Wav2Vec2 feature lengths without relying on a private HF API."""

    if len(conv_kernel) != len(conv_stride):
        raise ValueError("conv_kernel and conv_stride must have the same length")
    lengths = attention_mask.long().sum(dim=-1)
    for kernel_size, stride in zip(conv_kernel, conv_stride, strict=True):
        lengths = torch.div(lengths - int(kernel_size), int(stride), rounding_mode="floor") + 1
        lengths = lengths.clamp_min(0)
    return lengths


class AttentiveStatisticsPooling(nn.Module):
    """Pool a variable-length sequence into weighted mean and deviation."""

    def __init__(self, hidden_size: int, attention_size: int = 256) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )
        self.output_size = hidden_size * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.attention(self.input_norm(hidden_states)).squeeze(-1)
        if valid_mask is not None:
            if valid_mask.shape != scores.shape:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                    f"sequence shape {tuple(scores.shape)}"
                )
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mean = (weights * hidden_states).sum(dim=1)
        second_moment = (weights * hidden_states.square()).sum(dim=1)
        deviation = (second_moment - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat([mean, deviation], dim=-1)


class Wav2Vec2ConformerClassifier(nn.Module):
    """Raw-waveform four-class model built on a HF Wav2Vec2-Conformer encoder."""

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_MODEL,
        pretrained_revision: str | None = None,
        load_pretrained: bool = True,
        pooling_attention_size: int = 256,
        classifier_hidden_size: int = 256,
        dropout: float = 0.2,
        apply_spec_augment: bool = False,
        layerdrop: float = 0.0,
        gradient_checkpointing: bool = False,
        finetuning_mode: str = "last_n",
        train_last_n_layers: int = 4,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not 0.0 <= layerdrop < 1.0:
            raise ValueError("layerdrop must be in [0, 1)")

        if backbone is None:
            try:
                from transformers import AutoConfig, Wav2Vec2ConformerModel
            except ImportError as error:  # pragma: no cover - depends on server environment
                raise ImportError(
                    "Wav2Vec2ConformerClassifier requires transformers. "
                    "Install requirements-conformer.txt on the training server."
                ) from error

            config_kwargs: dict[str, object] = {"trust_remote_code": False}
            if pretrained_revision is not None:
                config_kwargs["revision"] = pretrained_revision
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
            )
            if getattr(config, "model_type", None) != "wav2vec2-conformer":
                raise ValueError(
                    f"Expected a wav2vec2-conformer checkpoint, got {config.model_type!r}"
                )
            # Keep the first baseline augmentation-free. Hugging Face's
            # Wav2Vec2-Conformer config enables latent SpecAugment by default.
            config.apply_spec_augment = apply_spec_augment
            config.layerdrop = layerdrop
            if load_pretrained:
                backbone = Wav2Vec2ConformerModel.from_pretrained(
                    pretrained_model_name_or_path,
                    config=config,
                    revision=pretrained_revision,
                )
            else:
                backbone = Wav2Vec2ConformerModel(config)

        self.backbone = backbone
        self.backbone_config: Any = backbone.config
        self.backbone_config.apply_spec_augment = apply_spec_augment
        self.backbone_config.layerdrop = layerdrop
        if hasattr(self.backbone.encoder, "layerdrop"):
            self.backbone.encoder.layerdrop = layerdrop
        hidden_size = int(self.backbone_config.hidden_size)
        self.pooling = AttentiveStatisticsPooling(hidden_size, pooling_attention_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.pooling.output_size),
            nn.Dropout(dropout),
            nn.Linear(self.pooling.output_size, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

        self.set_finetuning_mode(finetuning_mode, train_last_n_layers=train_last_n_layers)
        if (
            gradient_checkpointing
            and finetuning_mode != "frozen"
            and hasattr(self.backbone, "gradient_checkpointing_enable")
        ):
            # Re-entrant checkpointing drops the graph when the frozen prefix
            # produces inputs without ``requires_grad``. Non-reentrant
            # checkpointing supports that partial-finetuning boundary and lets
            # the selected late encoder blocks receive gradients.
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.gradient_checkpointing_enabled = True
        else:
            self.gradient_checkpointing_enabled = False
        self.gradient_checkpointing_use_reentrant = False

    def set_finetuning_mode(self, mode: str, *, train_last_n_layers: int = 4) -> None:
        allowed = {"frozen", "feature_encoder", "last_n", "full"}
        if mode not in allowed:
            raise ValueError(f"finetuning_mode must be one of {sorted(allowed)}, got {mode!r}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"

        if mode == "feature_encoder":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            feature_extractor = getattr(self.backbone, "feature_extractor", None)
            if feature_extractor is None:
                raise AttributeError("Backbone does not expose feature_extractor")
            for parameter in feature_extractor.parameters():
                parameter.requires_grad = False
        elif mode == "last_n":
            if train_last_n_layers <= 0:
                raise ValueError("train_last_n_layers must be positive for last_n mode")
            encoder = getattr(self.backbone, "encoder", None)
            layers = getattr(encoder, "layers", None)
            if layers is None:
                raise AttributeError("Backbone does not expose encoder.layers")
            if train_last_n_layers > len(layers):
                raise ValueError(
                    f"Requested {train_last_n_layers} trainable layers, backbone has {len(layers)}"
                )
            for layer in layers[-train_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            final_layer_norm = getattr(encoder, "layer_norm", None)
            if final_layer_norm is not None:
                for parameter in final_layer_norm.parameters():
                    parameter.requires_grad = True

        # Pooling and classification layers are always task-trainable.
        for module in (self.pooling, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.finetuning_mode = mode
        self.train_last_n_layers = train_last_n_layers

    def train(self, mode: bool = True) -> Wav2Vec2ConformerClassifier:
        super().train(mode)
        if not mode:
            return self

        finetuning_mode = getattr(self, "finetuning_mode", None)
        if finetuning_mode == "frozen":
            self.backbone.eval()
        elif finetuning_mode == "feature_encoder":
            # In this mode only the convolutional feature extractor is frozen.
            self.backbone.feature_extractor.eval()
        elif finetuning_mode == "last_n":
            # Keep every frozen component deterministic and activate train mode
            # only for the late blocks and final norm that receive gradients.
            self.backbone.eval()
            encoder = self.backbone.encoder
            # Gradient-checkpointed HF encoders gate checkpointing on this
            # parent flag. Set only the parent flag without recursively
            # re-enabling dropout in frozen child modules.
            encoder.training = True
            for layer in encoder.layers[-self.train_last_n_layers :]:
                layer.train(True)
            final_layer_norm = getattr(encoder, "layer_norm", None)
            if final_layer_norm is not None:
                final_layer_norm.train(True)
        return self

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        feature_mask = None
        if attention_mask is not None:
            lengths = feature_lengths_from_attention_mask(
                attention_mask,
                self.backbone_config.conv_kernel,
                self.backbone_config.conv_stride,
            ).clamp_max(hidden_states.size(1))
            positions = torch.arange(hidden_states.size(1), device=hidden_states.device)
            feature_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        embedding = self.pooling(hidden_states, feature_mask)
        return self.classifier(embedding)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
