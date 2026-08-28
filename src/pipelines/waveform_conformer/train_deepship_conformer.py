from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from src.data.deepship import CLASS_NAMES, SegmentRecord, segment_record_from_dict
from src.data.deepship_audit import load_experiment_config
from src.data.deepship_protocol_validation import load_split_manifest, validate_protocol_manifest
from src.data.deepship_waveform import (
    DeepShipWaveformSegmentDataset,
    RecordingBalancedEpochSampler,
    VesselBalancedEpochSampler,
    recording_representatives,
)
from src.evaluation.classification import (
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.evaluation.grouped_classification import (
    compute_grouped_metrics,
    save_prediction_rows,
)
from src.evaluation.model_selection import (
    SELECTION_SCHEMA_VERSION,
    build_validation_selection,
    primary_metric_improves,
    selection_is_better,
    should_stop_early,
    validate_resume_selection_state,
    validation_selection_rule,
)
from src.models.wav2vec2_conformer import (
    DEFAULT_PRETRAINED_MODEL,
    Wav2Vec2ConformerClassifier,
)
from src.utils.pathing import resolve_path


TARGET_SAMPLE_RATE = 16000


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def runtime_environment() -> dict[str, object]:
    root = resolve_path(".")
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    git_status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else "unknown",
        "git_worktree_dirty": bool(git_status_result.stdout.strip()),
        "git_status": git_status_result.stdout.splitlines(),
    }


@dataclass
class ConformerTrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "runs/deepship_conformer_baseline"
    split_manifest: str = ""
    isolation_experiment_config: str = "configs/experiments/isolation_comparison_v1.json"
    protocol_name: str | None = None
    pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_MODEL
    pretrained_revision: str | None = None
    load_pretrained: bool = True
    sample_rate: int = TARGET_SAMPLE_RATE
    clip_duration: float = 20.0
    normalize_waveform: bool = True
    remove_dc: bool = True
    pooling_attention_size: int = 256
    classifier_hidden_size: int = 256
    dropout: float = 0.2
    apply_spec_augment: bool = False
    layerdrop: float = 0.0
    finetuning_mode: str = "last_n"
    train_last_n_layers: int = 4
    gradient_checkpointing: bool = False
    batch_size: int = 1
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 8
    training_sampling: str = "fixed_anchor"
    train_samples_per_epoch: int | None = None
    epochs: int = 30
    encoder_learning_rate: float = 5e-6
    head_learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    warmup_start_factor: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    precision: str = "bf16"
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2
    log_interval: int = 100
    resume: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    evaluate_test_on_completion: bool = False
    device: str = "cuda"


def validate_config(config: ConformerTrainConfig) -> None:
    if not config.split_manifest:
        raise ValueError("split_manifest is required; do not build a new split at training time")
    if config.sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("The selected Wav2Vec2-Conformer baseline requires 16 kHz input")
    if config.clip_duration <= 0:
        raise ValueError("clip_duration must be positive")
    if config.batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if config.eval_batch_size is not None and config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive when provided")
    if config.training_sampling not in {
        "fixed_anchor",
        "recording_balanced_dynamic",
        "vessel_balanced_dynamic",
    }:
        raise ValueError("Unsupported training_sampling policy")
    if config.train_samples_per_epoch is not None and config.train_samples_per_epoch <= 0:
        raise ValueError("train_samples_per_epoch must be positive when provided")
    if (
        config.training_sampling == "fixed_anchor"
        and config.train_samples_per_epoch is not None
    ):
        raise ValueError(
            "train_samples_per_epoch is only supported for dynamic training sampling"
        )
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.encoder_learning_rate <= 0 or config.head_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")
    if config.min_learning_rate > min(
        config.encoder_learning_rate, config.head_learning_rate
    ):
        raise ValueError("min_learning_rate cannot exceed a parameter-group learning rate")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 < config.warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if (
        not math.isfinite(config.early_stopping_min_delta)
        or config.early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta must be finite and non-negative")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if config.max_train_batches is not None and config.max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided")
    if config.max_eval_batches is not None and config.max_eval_batches <= 0:
        raise ValueError("max_eval_batches must be positive when provided")
    if config.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config.finetuning_mode not in {"frozen", "feature_encoder", "last_n", "full"}:
        raise ValueError("Unsupported finetuning_mode")
    if config.finetuning_mode == "last_n" and config.train_last_n_layers <= 0:
        raise ValueError("train_last_n_layers must be positive for last_n mode")
    if not 0.0 <= config.layerdrop < 1.0:
        raise ValueError("layerdrop must be in [0, 1)")
    uses_cuda = str(config.device).startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {config.device}")
    if config.precision != "fp32" and not uses_cuda:
        raise ValueError("fp16/bf16 precision currently requires a CUDA device")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 precision requested but the CUDA device does not support bf16")


def load_and_validate_split(
    config: ConformerTrainConfig,
) -> tuple[dict[str, list[SegmentRecord]], dict[str, object]]:
    manifest_path = resolve_path(config.split_manifest)
    isolation_config = load_experiment_config(resolve_path(config.isolation_experiment_config))
    protocol_root = manifest_path.parent.parent
    audit_report_path = protocol_root / "audit" / "identity_audit.json"
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    manifest = load_split_manifest(manifest_path)
    validation = validate_protocol_manifest(
        manifest,
        isolation_config,
        audit_report,
        data_root=config.data_root,
    )
    if validation["status"] != "passed":
        raise ValueError(f"Frozen split manifest validation failed: {validation}")
    manifest_protocol = str(manifest["protocol"])
    if config.protocol_name is not None and config.protocol_name != manifest_protocol:
        raise ValueError(
            f"protocol_name={config.protocol_name!r} does not match manifest "
            f"protocol={manifest_protocol!r}"
        )

    split_segments = {split: [] for split in ("train", "val", "test")}
    for row in manifest["segments"]:
        split_segments[str(row["split"])].append(segment_record_from_dict(row))
    split_report = {
        "source": "frozen_isolation_manifest",
        "protocol": manifest_protocol,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "validation": validation,
        "waveform_context_seconds": config.clip_duration,
        "window_rule": "requested waveform window centered on each frozen three-second anchor",
        "samples_by_split": {name: len(rows) for name, rows in split_segments.items()},
    }
    return split_segments, split_report


def build_dataloaders(
    config: ConformerTrainConfig,
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    split_segments, split_report = load_and_validate_split(config)
    protocol = str(split_report["protocol"])
    train_sampler: RecordingBalancedEpochSampler | None = None
    if config.training_sampling in {
        "recording_balanced_dynamic",
        "vessel_balanced_dynamic",
    }:
        if protocol == "segment_level":
            raise ValueError(
                "Dynamic recording sampling requires a recording- or vessel-disjoint protocol"
            )
        train_recordings = recording_representatives(split_segments["train"])
        train_dataset = DeepShipWaveformSegmentDataset(
            train_recordings,
            data_root=config.data_root,
            sample_rate=config.sample_rate,
            clip_duration=config.clip_duration,
            normalize=config.normalize_waveform,
            remove_dc=config.remove_dc,
            dynamic_crop=True,
        )
        epoch_samples = config.train_samples_per_epoch or len(split_segments["train"])
        sampler_class = (
            RecordingBalancedEpochSampler
            if config.training_sampling == "recording_balanced_dynamic"
            else VesselBalancedEpochSampler
        )
        train_sampler = sampler_class(
            train_recordings,
            epoch_samples=epoch_samples,
            seed=config.seed,
        )
        sampling_id = (
            "S1" if config.training_sampling == "recording_balanced_dynamic" else "S2"
        )
        sampling_policy = (
            "class_then_recording_balanced_dynamic_crop"
            if config.training_sampling == "recording_balanced_dynamic"
            else "class_then_vessel_then_recording_balanced_dynamic_crop"
        )
        split_report["training_sampling"] = {
            "id": sampling_id,
            "policy": sampling_policy,
            "epoch_samples": epoch_samples,
            "recordings": len(train_recordings),
            "classes": len({row.label_index for row in train_recordings}),
            "crop_seed_rule": "deterministic from model seed, epoch, and draw",
        }
        split_report["training_window_rule"] = (
            "uniform random frame-aligned crop within the selected training recording"
        )
    else:
        train_dataset = DeepShipWaveformSegmentDataset(
            split_segments["train"],
            data_root=config.data_root,
            sample_rate=config.sample_rate,
            clip_duration=config.clip_duration,
            normalize=config.normalize_waveform,
            remove_dc=config.remove_dc,
        )
        split_report["training_sampling"] = {
            "id": "S0",
            "policy": "fixed_manifest_anchor",
            "epoch_samples": len(split_segments["train"]),
        }
        split_report["training_window_rule"] = split_report["window_rule"]

    datasets = {
        "train": train_dataset,
        **{
            split: DeepShipWaveformSegmentDataset(
                split_segments[split],
                data_root=config.data_root,
                sample_rate=config.sample_rate,
                clip_duration=config.clip_duration,
                normalize=config.normalize_waveform,
                remove_dc=config.remove_dc,
                return_index=True,
            )
            for split in ("val", "test")
        },
    }
    split_report["evaluation_window_rule"] = split_report["window_rule"]
    pin_memory = str(config.device).startswith("cuda")
    worker_options: dict[str, object] = {}
    if config.num_workers > 0:
        worker_options = {
            "persistent_workers": True,
            "prefetch_factor": config.prefetch_factor,
            "worker_init_fn": _configure_dataloader_worker,
        }
    eval_batch_size = config.eval_batch_size or config.batch_size
    dataloaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
    }
    for split in ("val", "test"):
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        )
    split_report["batch_sizes"] = {
        "train": config.batch_size,
        "validation": eval_batch_size,
        "test": eval_batch_size,
    }
    return dataloaders, split_report


def _configure_dataloader_worker(_worker_id: int) -> None:
    # Each process handles one audio item at a time. Keeping Torch kernels
    # single-threaded avoids num_workers × BLAS-thread oversubscription.
    torch.set_num_threads(1)


def build_optimizer(
    model: Wav2Vec2ConformerClassifier,
    config: ConformerTrainConfig,
) -> torch.optim.Optimizer:
    encoder_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder_parameters.append(parameter)
        else:
            task_parameters.append(parameter)
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append(
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": config.encoder_learning_rate,
            }
        )
    if task_parameters:
        parameter_groups.append(
            {"name": "head", "params": task_parameters, "lr": config.head_learning_rate}
        )
    if not parameter_groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ConformerTrainConfig,
    *,
    total_optimizer_steps: int,
) -> tuple[LambdaLR, dict[str, object]]:
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    warmup_steps = min(
        total_optimizer_steps - 1,
        int(round(total_optimizer_steps * config.warmup_ratio)),
    )
    decay_steps = max(1, total_optimizer_steps - warmup_steps)

    def schedule_for(base_learning_rate: float):
        minimum_factor = config.min_learning_rate / base_learning_rate

        def learning_rate_multiplier(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                progress = current_step / warmup_steps
                return config.warmup_start_factor + (
                    1.0 - config.warmup_start_factor
                ) * progress
            decay_progress = min(
                max(current_step - warmup_steps, 0) / decay_steps,
                1.0,
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine_factor

        return learning_rate_multiplier

    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[schedule_for(rate) for rate in base_learning_rates],
    )
    report = {
        "name": "optimizer_step_linear_warmup_cosine_decay",
        "scheduler_step_unit": "optimizer_step",
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": warmup_steps,
        "warmup_start_factor": config.warmup_start_factor,
        "decay_steps": decay_steps,
        "base_learning_rates": base_learning_rates,
        "minimum_learning_rate": config.min_learning_rate,
    }
    return scheduler, report


def _autocast_context(config: ConformerTrainConfig):
    if not str(config.device).startswith("cuda") or config.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _move_batch(batch, device: str):
    input_values, attention_mask, targets = batch[:3]
    use_cuda = str(device).startswith("cuda")
    return (
        input_values.to(device, non_blocking=use_cuda),
        attention_mask.to(device, non_blocking=use_cuda),
        targets.to(device, non_blocking=use_cuda),
    )


def validate_trainable_gradients(model: nn.Module) -> None:
    """Fail fast when a nominally trainable parameter has no usable gradient."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            "Trainable parameters did not receive gradients. "
            f"Examples: {examples}. Check the frozen/checkpointing boundary."
        )

    nonfinite = [
        name
        for name, parameter in trainable
        if not bool(torch.isfinite(parameter.grad).all())
    ]
    if nonfinite:
        examples = ", ".join(nonfinite[:5])
        raise FloatingPointError(
            "Non-finite gradients detected after precision unscaling. "
            f"Examples: {examples}. Prefer BF16 on supported CUDA devices."
        )


def _cuda_device(device: str) -> torch.device | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.device(device)


def _start_phase_timer(device: str) -> tuple[float, torch.device | None]:
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    return time.perf_counter(), cuda_device


class TrainingProgress:
    """Render one live terminal line while keeping piped logs readable."""

    def __init__(
        self,
        interactive_stream: TextIO | None,
        *,
        owns_stream: bool = False,
        terminal_width: int | None = None,
    ) -> None:
        self.interactive_stream = interactive_stream
        self.owns_stream = owns_stream
        self.terminal_width = terminal_width
        self.rendered_width = 0

    @classmethod
    def auto(cls) -> TrainingProgress:
        for stream in (sys.stdout, sys.stderr):
            if stream.isatty():
                return cls(stream)

        terminal_path = "CONOUT$" if platform.system() == "Windows" else "/dev/tty"
        try:
            terminal_stream = open(terminal_path, "w", encoding="utf-8", buffering=1)
        except OSError:
            return cls(None)
        return cls(terminal_stream, owns_stream=True)

    def _available_width(self) -> int | None:
        if self.terminal_width is not None:
            return self.terminal_width
        if self.interactive_stream is None:
            return None
        try:
            return max(20, os.get_terminal_size(self.interactive_stream.fileno()).columns - 1)
        except (AttributeError, OSError):
            return None

    def update(self, line: str) -> None:
        if self.interactive_stream is None:
            print(line, flush=True)
            return
        available_width = self._available_width()
        if available_width is not None and len(line) > available_width:
            line = line[: max(1, available_width - 1)] + "…"
        self.interactive_stream.write(f"\r{line}\x1b[K")
        self.interactive_stream.flush()
        self.rendered_width = len(line)

    def clear(self) -> None:
        if self.interactive_stream is None or self.rendered_width == 0:
            return
        self.interactive_stream.write("\r\x1b[2K\r")
        self.interactive_stream.flush()
        self.rendered_width = 0

    def finish_epoch(self, summary: str) -> None:
        self.clear()
        print(summary, flush=True)

    def close(self) -> None:
        self.clear()
        if self.owns_stream and self.interactive_stream is not None:
            self.interactive_stream.close()
        self.interactive_stream = None


def _learning_rate_text(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", index))
        if name == "encoder":
            name = "enc"
        parts.append(f"{name}:{float(group['lr']):.2e}")
    return ",".join(parts)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_line(
    *,
    epoch: int,
    epochs: int,
    phase: str,
    batch: int,
    batches: int,
    samples: int,
    total_loss: float,
    total_correct: int,
    started_at: float,
    cuda_device: torch.device | None,
    learning_rates: str,
) -> str:
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        peak_gib = torch.cuda.max_memory_allocated(cuda_device) / (1024**3)
        gpu_peak = f"{peak_gib:.2f}GiB"
    else:
        gpu_peak = "n/a"
    progress = 100.0 * batch / batches
    if samples:
        elapsed_seconds = max(time.perf_counter() - started_at, 1e-12)
        average_loss = f"{total_loss / samples:.4f}"
        average_accuracy = f"{total_correct / samples:.4f}"
        samples_per_second = f"{samples / elapsed_seconds:.2f}"
    else:
        average_loss = "--"
        average_accuracy = "--"
        samples_per_second = "--"
    return (
        f"Epoch {epoch}/{epochs} | {phase} | batch={batch}/{batches} "
        f"({progress:.1f}%) | avg_loss={average_loss} | avg_acc={average_accuracy} "
        f"| lr={learning_rates} | samples_per_sec={samples_per_second} "
        f"| gpu_peak={gpu_peak}"
    )


def _append_prediction_rows(
    rows: list[dict[str, object]],
    dataset: DeepShipWaveformSegmentDataset,
    targets: torch.Tensor,
    logits: torch.Tensor,
    indexes: torch.Tensor,
) -> None:
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()
    predictions = logits.argmax(dim=1).detach().cpu().tolist()
    for target, prediction, probability, index in zip(
        targets.detach().cpu().tolist(),
        predictions,
        probabilities,
        indexes.tolist(),
        strict=True,
    ):
        segment = dataset.segments[index]
        rows.append(
            {
                "relative_path": segment.relative_path,
                "group_key": segment.group_key,
                "vessel_key": segment.vessel_key,
                "segment_index": segment.segment_index,
                "true_label": target,
                "predicted_label": prediction,
                "probabilities": probability,
            }
        )


def run_epoch(
    model: Wav2Vec2ConformerClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: ConformerTrainConfig,
    *,
    epoch: int,
    phase: str,
    progress: TrainingProgress,
    learning_rates: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
    prediction_rows: list[dict[str, object]] | None = None,
    runtime_stats: dict[str, float] | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    effective_batches = (
        len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    )
    if effective_batches == 0:
        raise ValueError("Dataloader produced no batches")
    if phase not in {"train", "val"}:
        raise ValueError("phase must be train or val")
    if (phase == "train") != is_train:
        raise ValueError("train phase requires an optimizer; val phase must not receive one")
    if scheduler is not None and not is_train:
        raise ValueError("A learning-rate scheduler is only valid during training")
    prediction_dataset = None
    if prediction_rows is not None:
        if is_train:
            raise ValueError("Prediction collection is only supported during validation")
        if not isinstance(dataloader.dataset, DeepShipWaveformSegmentDataset):
            raise ValueError("Prediction DataLoader must use DeepShip waveform segments")
        if not dataloader.dataset.return_index:
            raise ValueError("Prediction DataLoader must expose DeepShip segment indexes")
        prediction_dataset = dataloader.dataset
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    gradients_validated = False
    started_at, cuda_device = _start_phase_timer(config.device)
    progress.update(
        _progress_line(
            epoch=epoch,
            epochs=config.epochs,
            phase=phase,
            batch=0,
            batches=effective_batches,
            samples=0,
            total_loss=0.0,
            total_correct=0,
            started_at=started_at,
            cuda_device=cuda_device,
            learning_rates=(
                _learning_rate_text(optimizer) if optimizer is not None else learning_rates
            ),
        )
    )
    data_wait_seconds = 0.0
    last_batch_finished_at = time.perf_counter()

    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= effective_batches:
                break
            data_wait_seconds += time.perf_counter() - last_batch_finished_at
            input_values, attention_mask, targets = _move_batch(batch, config.device)
            with _autocast_context(config):
                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, targets)

            if prediction_rows is not None and prediction_dataset is not None:
                _append_prediction_rows(
                    prediction_rows,
                    prediction_dataset,
                    targets,
                    logits,
                    batch[3],
                )

            if is_train:
                scaled_loss = loss / config.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0
                    or batch_index + 1 == effective_batches
                )
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if not gradients_validated:
                        validate_trainable_gradients(model)
                        gradients_validated = True
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * input_values.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += input_values.size(0)
            completed_batches = batch_index + 1
            if (
                completed_batches % config.log_interval == 0
                or completed_batches == effective_batches
            ):
                progress.update(
                    _progress_line(
                        epoch=epoch,
                        epochs=config.epochs,
                        phase=phase,
                        batch=completed_batches,
                        batches=effective_batches,
                        samples=total_samples,
                        total_loss=total_loss,
                        total_correct=total_correct,
                        started_at=started_at,
                        cuda_device=cuda_device,
                        learning_rates=(
                            _learning_rate_text(optimizer)
                            if optimizer is not None
                            else learning_rates
                        ),
                    ),
                )
            last_batch_finished_at = time.perf_counter()
    phase_seconds = max(time.perf_counter() - started_at, 1e-12)
    if runtime_stats is not None:
        runtime_stats.update(
            {
                "phase_seconds": phase_seconds,
                "data_wait_seconds": data_wait_seconds,
                "data_wait_fraction": data_wait_seconds / phase_seconds,
                "samples_per_second": total_samples / phase_seconds,
            }
        )
    return total_loss / total_samples, total_correct / total_samples


def collect_prediction_rows(
    model: Wav2Vec2ConformerClassifier,
    dataloader: DataLoader,
    config: ConformerTrainConfig,
    *,
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    model.eval()
    dataset = dataloader.dataset
    if not isinstance(dataset, DeepShipWaveformSegmentDataset) or not dataset.return_index:
        raise ValueError("Prediction DataLoader must expose DeepShip segment indexes")
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_values, attention_mask, targets = _move_batch(batch, config.device)
            indexes = batch[3]
            with _autocast_context(config):
                logits = model(input_values, attention_mask=attention_mask)
            _append_prediction_rows(rows, dataset, targets, logits, indexes)
    return rows


def train(config: ConformerTrainConfig) -> dict[str, object]:
    validate_config(config)
    set_seed(config.seed)
    output_root = resolve_path(config.output_root)
    if config.resume and (output_root / "reports" / "run_complete.json").is_file():
        raise RuntimeError(f"Run is already complete and must not be resumed: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not config.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use a new directory or --resume."
        )
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    predictions_dir = output_root / "predictions"
    for directory in (metrics_dir, figures_dir, models_dir, reports_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    environment = runtime_environment()
    (reports_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataloaders, split_report = build_dataloaders(config)
    protocol = str(split_report["protocol"])
    shutil.copyfile(
        resolve_path(config.split_manifest),
        reports_dir / "frozen_split_manifest.json",
    )
    (reports_dir / "split_validation.json").write_text(
        json.dumps(split_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model = Wav2Vec2ConformerClassifier(
        num_classes=len(CLASS_NAMES),
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        pretrained_revision=config.pretrained_revision,
        load_pretrained=config.load_pretrained,
        pooling_attention_size=config.pooling_attention_size,
        classifier_hidden_size=config.classifier_hidden_size,
        dropout=config.dropout,
        apply_spec_augment=config.apply_spec_augment,
        layerdrop=config.layerdrop,
        gradient_checkpointing=config.gradient_checkpointing,
        finetuning_mode=config.finetuning_mode,
        train_last_n_layers=config.train_last_n_layers,
    ).to(config.device)
    backbone_config = model.backbone_config.to_dict()
    model_report = {
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "requested_revision": config.pretrained_revision,
        "resolved_commit_hash": getattr(model.backbone_config, "_commit_hash", None),
        "load_pretrained": config.load_pretrained,
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": model.num_trainable_parameters,
        "finetuning_mode": config.finetuning_mode,
        "train_last_n_layers": config.train_last_n_layers,
        "apply_spec_augment": config.apply_spec_augment,
        "layerdrop": config.layerdrop,
        "gradient_checkpointing": model.gradient_checkpointing_enabled,
        "gradient_checkpointing_use_reentrant": (
            model.gradient_checkpointing_use_reentrant
            if model.gradient_checkpointing_enabled
            else None
        ),
        "backbone_config": backbone_config,
    }
    (reports_dir / "model_report.json").write_text(
        json.dumps(model_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    train_batches_per_epoch = (
        len(dataloaders["train"])
        if config.max_train_batches is None
        else min(len(dataloaders["train"]), config.max_train_batches)
    )
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch / config.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    scheduler, schedule_report = build_scheduler(
        optimizer,
        config,
        total_optimizer_steps=total_optimizer_steps,
    )
    schedule_report["train_batches_per_epoch"] = train_batches_per_epoch
    schedule_report["optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
    (reports_dir / "training_schedule.json").write_text(
        json.dumps(schedule_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    use_scaler = str(config.device).startswith("cuda") and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_segment_macro_f1": [],
        "val_recording_acc": [],
        "val_recording_macro_f1": [],
        "val_vessel_acc": [],
        "val_vessel_macro_f1": [],
        "selection_primary_value": [],
        "optimizer_steps_completed": [],
        "encoder_learning_rate": [],
        "head_learning_rate": [],
    }
    history_path = reports_dir / "deepship_conformer_history.json"
    best_model_path = models_dir / "deepship_conformer_best.pt"
    last_model_path = models_dir / "deepship_conformer_last.pt"
    best_selection: dict[str, object] | None = None
    best_validation_metrics: dict[str, dict[str, object] | None] | None = None
    best_epoch = -1
    early_stopping_primary_value: float | None = None
    early_stopping_reference_epoch = -1
    start_epoch = 1
    sampling_exposure_path = reports_dir / "training_sampling_exposure.json"
    sampling_exposure_history: list[dict[str, object]] = []

    if config.resume:
        if not last_model_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is unavailable: {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=config.device, weights_only=False)
        validate_resume_selection_state(
            checkpoint,
            manifest_sha256=str(split_report["manifest_sha256"]),
            protocol=protocol,
        )
        if checkpoint.get("schedule_report") != schedule_report:
            raise ValueError("Resume checkpoint uses a different optimizer schedule")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_selection = checkpoint["best_selection"]
        best_validation_metrics = checkpoint["best_validation_metrics"]
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        stored_primary_value = checkpoint.get("early_stopping_primary_value")
        early_stopping_primary_value = (
            float(stored_primary_value) if stored_primary_value is not None else None
        )
        early_stopping_reference_epoch = int(
            checkpoint.get("early_stopping_reference_epoch", -1)
        )
        if early_stopping_primary_value is None or early_stopping_reference_epoch < 1:
            early_stopping_primary_value = None
            early_stopping_reference_epoch = -1
            for completed_epoch, primary_value in enumerate(
                history["selection_primary_value"], start=1
            ):
                if (
                    early_stopping_primary_value is None
                    or float(primary_value)
                    > early_stopping_primary_value
                    + config.early_stopping_min_delta
                    + 1e-12
                ):
                    early_stopping_primary_value = float(primary_value)
                    early_stopping_reference_epoch = completed_epoch
        if sampling_exposure_path.is_file():
            stored_exposure = json.loads(
                sampling_exposure_path.read_text(encoding="utf-8")
            )
            if isinstance(stored_exposure, list):
                sampling_exposure_history = [
                    row
                    for row in stored_exposure
                    if int(row.get("epoch", 0)) < start_epoch
                ]
        restore_rng_state(checkpoint["rng_state"])

    progress = TrainingProgress.auto()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        epoch_sampling_exposure = None
        train_sampler = dataloaders["train"].sampler
        if isinstance(train_sampler, RecordingBalancedEpochSampler):
            train_sampler.set_epoch(epoch)
            epoch_sampling_exposure = train_sampler.exposure_report()
        progress_learning_rates = _learning_rate_text(optimizer)
        train_runtime_stats: dict[str, float] = {}
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            config,
            epoch=epoch,
            phase="train",
            progress=progress,
            learning_rates=progress_learning_rates,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            max_batches=config.max_train_batches,
            runtime_stats=train_runtime_stats,
        )
        if epoch_sampling_exposure is not None:
            epoch_sampling_exposure["runtime"] = train_runtime_stats
        validation_learning_rates = _learning_rate_text(optimizer)
        validation_segment_predictions: list[dict[str, object]] = []
        val_loss, val_acc = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            config,
            epoch=epoch,
            phase="val",
            progress=progress,
            learning_rates=validation_learning_rates,
            max_batches=config.max_eval_batches,
            prediction_rows=validation_segment_predictions,
        )
        validation_metrics, validation_recording_predictions, validation_vessel_predictions = (
            compute_grouped_metrics(validation_segment_predictions, CLASS_NAMES)
        )
        selection = build_validation_selection(protocol, validation_metrics, val_loss)
        segment_validation_metrics = validation_metrics["segment"]
        recording_validation_metrics = validation_metrics["recording"]
        vessel_validation_metrics = validation_metrics["vessel"]
        if segment_validation_metrics is None or recording_validation_metrics is None:
            raise RuntimeError("Validation aggregation did not produce required metrics")
        learning_rates = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        encoder_lr = learning_rates.get("encoder", 0.0)
        head_lr = learning_rates["head"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_segment_macro_f1"].append(
            float(segment_validation_metrics["macro_f1"])
        )
        history["val_recording_acc"].append(
            float(recording_validation_metrics["accuracy"])
        )
        history["val_recording_macro_f1"].append(
            float(recording_validation_metrics["macro_f1"])
        )
        history["val_vessel_acc"].append(
            float(vessel_validation_metrics["accuracy"])
            if vessel_validation_metrics is not None
            else None
        )
        history["val_vessel_macro_f1"].append(
            float(vessel_validation_metrics["macro_f1"])
            if vessel_validation_metrics is not None
            else None
        )
        history["selection_primary_value"].append(float(selection["primary_value"]))
        history["optimizer_steps_completed"].append(epoch * optimizer_steps_per_epoch)
        history["encoder_learning_rate"].append(encoder_lr)
        history["head_learning_rate"].append(head_lr)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if epoch_sampling_exposure is not None:
            sampling_exposure_history.append(epoch_sampling_exposure)
            sampling_exposure_path.write_text(
                json.dumps(sampling_exposure_history, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

        checkpoint_improved = selection_is_better(selection, best_selection)
        meaningful_improvement = primary_metric_improves(
            selection,
            early_stopping_primary_value,
            min_delta=config.early_stopping_min_delta,
        )
        if meaningful_improvement:
            early_stopping_primary_value = float(selection["primary_value"])
            early_stopping_reference_epoch = epoch
        if checkpoint_improved:
            best_selection = selection
            best_validation_metrics = validation_metrics
            best_epoch = epoch
            save_prediction_rows(
                validation_segment_predictions,
                predictions_dir / "validation_best_segment_predictions.csv",
                CLASS_NAMES,
            )
            save_prediction_rows(
                validation_recording_predictions,
                predictions_dir / "validation_best_recording_predictions.csv",
                CLASS_NAMES,
            )
            if validation_vessel_predictions:
                save_prediction_rows(
                    validation_vessel_predictions,
                    predictions_dir / "validation_best_vessel_predictions.csv",
                    CLASS_NAMES,
                )
            save_metrics(
                segment_validation_metrics,
                metrics_dir / "validation_best_segment_metrics.json",
            )
            save_metrics(
                recording_validation_metrics,
                metrics_dir / "validation_best_recording_metrics.json",
            )
            if vessel_validation_metrics is not None:
                save_metrics(
                    vessel_validation_metrics,
                    metrics_dir / "validation_best_vessel_metrics.json",
                )
            (reports_dir / "validation_best_selection.json").write_text(
                json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "selection_schema_version": SELECTION_SCHEMA_VERSION,
                    "selection": selection,
                    "validation_metrics": validation_metrics,
                    "schedule_report": schedule_report,
                    "num_parameters": model.num_parameters,
                    "num_trainable_parameters": model.num_trainable_parameters,
                    "split_report": split_report,
                },
                best_model_path,
            )
        should_stop = should_stop_early(
            improved=meaningful_improvement,
            epoch=epoch,
            best_epoch=early_stopping_reference_epoch,
            patience=config.early_stopping_patience,
        )
        if best_selection is None or best_validation_metrics is None:
            raise RuntimeError("Best validation selection was not initialized")
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "schedule_report": schedule_report,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "config": asdict(config),
                "epoch": epoch,
                "history": history,
                "selection_schema_version": SELECTION_SCHEMA_VERSION,
                "selection_rule": validation_selection_rule(protocol),
                "best_selection": best_selection,
                "best_validation_metrics": best_validation_metrics,
                "best_epoch": best_epoch,
                "early_stopping_primary_value": early_stopping_primary_value,
                "early_stopping_reference_epoch": early_stopping_reference_epoch,
                "split_manifest_sha256": split_report["manifest_sha256"],
                "rng_state": capture_rng_state(),
            },
            last_model_path,
        )
        selection_name = str(selection["rule"]["name"])
        best_selection_value = float(best_selection["primary_value"])
        recording_macro_f1 = float(recording_validation_metrics["macro_f1"])
        vessel_macro_f1 = (
            float(vessel_validation_metrics["macro_f1"])
            if vessel_validation_metrics is not None
            else None
        )
        vessel_summary = (
            f"{vessel_macro_f1:.4f}" if vessel_macro_f1 is not None else "n/a"
        )
        summary = (
            f"Epoch {epoch}/{config.epochs} | done | train_loss={train_loss:.4f} "
            f"| train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
            f"| val_acc={val_acc:.4f} | val_recording_f1={recording_macro_f1:.4f} "
            f"| val_vessel_f1={vessel_summary} "
            f"| select={selection_name}:{float(selection['primary_value']):.4f} "
            f"| best_select={best_selection_value:.4f} "
            f"| early_stop_wait={epoch - early_stopping_reference_epoch}/"
            f"{config.early_stopping_patience} "
            f"| time={_format_duration(time.perf_counter() - epoch_started_at)}"
        )
        if should_stop:
            summary += f" | early_stop=true | best_epoch={best_epoch}"
        progress.finish_epoch(summary)
        if should_stop:
            break
    progress.close()

    segment_metrics = None
    recording_metrics = None
    vessel_metrics = None
    if config.evaluate_test_on_completion:
        checkpoint = torch.load(
            best_model_path,
            map_location=config.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        segment_predictions = collect_prediction_rows(
            model,
            dataloaders["test"],
            config,
            max_batches=config.max_eval_batches,
        )
        test_metrics, recording_predictions, vessel_predictions = (
            compute_grouped_metrics(segment_predictions, CLASS_NAMES)
        )
        segment_metrics = test_metrics["segment"]
        recording_metrics = test_metrics["recording"]
        vessel_metrics = test_metrics["vessel"]
        if segment_metrics is None or recording_metrics is None:
            raise RuntimeError("Test aggregation did not produce required metrics")
        save_metrics(segment_metrics, metrics_dir / "segment_metrics.json")
        save_metrics(recording_metrics, metrics_dir / "recording_metrics.json")
        if vessel_metrics is not None:
            save_metrics(vessel_metrics, metrics_dir / "vessel_metrics.json")
        save_prediction_rows(
            segment_predictions,
            predictions_dir / "test_segment_predictions.csv",
            CLASS_NAMES,
        )
        save_prediction_rows(
            recording_predictions,
            predictions_dir / "test_recording_predictions.csv",
            CLASS_NAMES,
        )
        if vessel_predictions:
            save_prediction_rows(
                vessel_predictions,
                predictions_dir / "test_vessel_predictions.csv",
                CLASS_NAMES,
            )

        plot_confusion_matrix(
            segment_metrics["confusion_matrix"],
            CLASS_NAMES,
            figures_dir / "segment_confusion_matrix.png",
            title=f"DeepShip {protocol} Conformer Segment Confusion Matrix",
        )
        plot_confusion_matrix(
            recording_metrics["confusion_matrix"],
            CLASS_NAMES,
            figures_dir / "recording_confusion_matrix.png",
            title=f"DeepShip {protocol} Conformer Recording Confusion Matrix",
        )
        if vessel_metrics is not None:
            plot_confusion_matrix(
                vessel_metrics["confusion_matrix"],
                CLASS_NAMES,
                figures_dir / "vessel_confusion_matrix.png",
                title=(
                    f"DeepShip {protocol} Conformer Vessel-group Confusion Matrix"
                ),
            )
    curve_history = {
        key: history[key] for key in ("train_loss", "val_loss", "train_acc", "val_acc")
    }
    plot_training_curves(
        curve_history,
        figures_dir / "deepship_conformer_training_curves.png",
        title=f"DeepShip {protocol} Wav2Vec2-Conformer Training Curves",
    )

    if best_selection is None or best_validation_metrics is None:
        raise RuntimeError("Best validation selection is unavailable")
    best_segment_metrics = best_validation_metrics["segment"]
    if best_segment_metrics is None:
        raise RuntimeError("Best validation segment metrics are unavailable")
    result = {
        "status": (
            "complete" if config.evaluate_test_on_completion else "validation_complete"
        ),
        "test_evaluated": config.evaluate_test_on_completion,
        "protocol": protocol,
        "segment_metrics": segment_metrics,
        "recording_metrics": recording_metrics,
        "vessel_metrics": vessel_metrics,
        "best_epoch": best_epoch,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_rule": validation_selection_rule(protocol),
        "best_selection": best_selection,
        "best_validation_metrics": best_validation_metrics,
        "best_val_acc": float(best_segment_metrics["accuracy"]),
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": model.num_trainable_parameters,
    }
    (reports_dir / "run_complete.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
