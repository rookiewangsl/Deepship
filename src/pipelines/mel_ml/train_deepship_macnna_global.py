from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import shutil
import sys
import time
from typing import TextIO

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipMelSegmentDataset,
    build_paper_split,
    save_segment_split_manifest,
    scan_deepship,
    segment_record_from_dict,
)
from src.data.deepship_audit import load_experiment_config
from src.data.deepship_waveform import (
    DeepShipMelWindowDataset,
    VesselBalancedEpochSampler,
    recording_representatives,
)
from src.data.deepship_protocol_validation import (
    load_split_manifest,
    validate_protocol_manifest,
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
from src.models.ma_cnn_a import (
    MACNNA_MODEL_VARIANTS,
    MACNNABaseClassifier,
    THREE_BRANCH_KERNEL_SIZES,
    build_macnna_model,
    feature_time_padding_mask,
)
from src.pipelines.mel_ml.isolation_experiment import enforce_training_config
from src.pipelines.mel_ml.train_deepship_macnna import (
    atomic_torch_save,
    capture_rng_state,
    get_default_device,
    restore_rng_state,
    runtime_environment,
    set_seed,
)
from src.utils.pathing import resolve_path


TARGET_SAMPLE_RATE = 16_000


@dataclass
class GlobalAttentionTrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs/deepship_macnna_global"
    split_manifest: str | None = None
    experiment_config: str | None = None
    g_series_config: str = "configs/experiments/macnna_global_v1.json"
    protocol_name: str | None = None
    model_variant: str = "g0"
    attention_d_model: int = 128
    attention_num_heads: int = 4
    attention_ffn_expansion: int = 2
    attention_position_kernel_size: int = 9
    attention_temporal_kernel_size: int = 15
    attention_dropout: float = 0.1
    attention_gate_init: float = -2.0
    training_sampling: str = "fixed_anchor"
    train_samples_per_epoch: int | None = None
    clip_duration: float = 3.0
    samples_per_class: int = 5000
    train_per_class: int = 3500
    val_per_class: int = 1000
    test_per_class: int = 500
    seed: int = 42
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    n_mels: int = 64
    highpass_freq: float | None = None
    batch_size: int = 16
    eval_batch_size: int | None = None
    epochs: int = 100
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = None
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 10
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.005
    precision: str = "bf16"
    num_workers: int = 0
    prefetch_factor: int = 2
    log_interval: int = 100
    resume: bool = False
    allow_experiment_overrides: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    evaluate_test_on_completion: bool = False
    device: str = get_default_device()


def validate_config(config: GlobalAttentionTrainConfig) -> None:
    if config.model_variant not in MACNNA_MODEL_VARIANTS:
        raise ValueError(
            f"model_variant must be one of {MACNNA_MODEL_VARIANTS}, got {config.model_variant!r}"
        )
    if config.attention_d_model <= 0:
        raise ValueError("attention_d_model must be positive")
    if (
        config.attention_num_heads <= 0
        or config.attention_d_model % config.attention_num_heads != 0
    ):
        raise ValueError("attention_d_model must be divisible by attention_num_heads")
    if config.attention_ffn_expansion <= 0:
        raise ValueError("attention_ffn_expansion must be positive")
    if config.training_sampling not in {"fixed_anchor", "vessel_balanced_dynamic"}:
        raise ValueError("training_sampling must be fixed_anchor or vessel_balanced_dynamic")
    if config.training_sampling == "fixed_anchor" and config.train_samples_per_epoch is not None:
        raise ValueError("train_samples_per_epoch is only supported for dynamic sampling")
    if config.training_sampling != "fixed_anchor":
        if config.train_samples_per_epoch is None or config.train_samples_per_epoch <= 0:
            raise ValueError("Dynamic sampling requires a positive train_samples_per_epoch")
        if config.highpass_freq is not None:
            raise ValueError("Long-context dynamic Mel windows do not support highpass filtering")
    for name, value in (
        ("attention_position_kernel_size", config.attention_position_kernel_size),
        ("attention_temporal_kernel_size", config.attention_temporal_kernel_size),
    ):
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    if not 0.0 <= config.attention_dropout < 1.0:
        raise ValueError("attention_dropout must be in [0, 1)")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.eval_batch_size is not None and config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.optimizer not in {"sgd", "adamw"}:
        raise ValueError("optimizer must be sgd or adamw")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.max_grad_norm is not None and (
        not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be positive and finite")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be finite and non-negative")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if config.evaluate_test_on_completion:
        raise ValueError(
            "G-series development runs are validation-only; sealed test evaluation is disabled"
        )


def _validate_g_series_experiment_config(
    config: GlobalAttentionTrainConfig,
    experiment: dict[str, object],
) -> list[str]:
    if experiment.get("experiment_id") not in {
        "macnna_global_v1",
        "macnna_global_l20_v1",
        "macnna_global_l20_repeats_v1",
    }:
        raise ValueError("Unexpected G-series experiment config")
    features = experiment["features"]
    adapter = experiment["shared_adapter"]
    training = experiment["training"]
    if not isinstance(features, dict) or not isinstance(adapter, dict) or not isinstance(training, dict):
        raise TypeError("G-series experiment config has invalid sections")
    expected = {
        "clip_duration": features["clip_duration_seconds"],
        "n_fft": features["n_fft"],
        "win_length": features["win_length"],
        "hop_length": features["hop_length"],
        "n_mels": features["n_mels"],
        "highpass_freq": features["highpass_freq"],
        "attention_d_model": adapter["d_model"],
        "attention_num_heads": adapter["num_heads"],
        "attention_ffn_expansion": adapter["ffn_expansion"],
        "attention_position_kernel_size": adapter["position_kernel_size"],
        "attention_temporal_kernel_size": adapter["temporal_kernel_size"],
        "attention_dropout": adapter["dropout"],
        "attention_gate_init": adapter["gate_init"],
        "training_sampling": training.get("training_sampling", "fixed_anchor"),
        "train_samples_per_epoch": training.get("train_samples_per_epoch"),
        "batch_size": training["batch_size"],
        "eval_batch_size": training["eval_batch_size"],
        "epochs": training["epochs"],
        "optimizer": training.get("optimizer", "sgd"),
        "learning_rate": training["learning_rate"],
        "momentum": training["momentum"],
        "weight_decay": training.get("weight_decay", 0.0),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 1),
        "max_grad_norm": training.get("max_grad_norm"),
        "min_learning_rate": training["min_learning_rate"],
        "warmup_epochs": training["warmup_epochs"],
        "early_stopping_patience": training["early_stopping_patience"],
        "early_stopping_min_delta": training["early_stopping_min_delta"],
        "precision": training["precision"],
        "num_workers": training["num_workers"],
    }
    actual = asdict(config)
    mismatches = [
        f"{field}: expected {expected_value!r}, got {actual.get(field)!r}"
        for field, expected_value in expected.items()
        if actual.get(field) != expected_value
    ]
    allowed_seeds = training.get("model_seeds")
    if allowed_seeds is None:
        allowed_seeds = [training["seed"]]
    if config.seed not in {int(seed) for seed in allowed_seeds}:
        mismatches.append(
            f"seed: expected one of {[int(seed) for seed in allowed_seeds]!r}, "
            f"got {config.seed!r}"
        )
    for field in ("max_train_batches", "max_eval_batches"):
        if actual.get(field) is not None:
            mismatches.append(f"{field}: expected None, got {actual.get(field)!r}")
    if mismatches and not config.allow_experiment_overrides:
        raise ValueError(
            "Training configuration differs from the frozen G-series experiment. "
            "Use --allow-experiment-overrides only for smoke/debug runs:\n- "
            + "\n- ".join(mismatches)
        )
    return mismatches


def load_g_series_experiment_config(path: str) -> dict[str, object]:
    data = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("G-series experiment config must contain a JSON object")
    return data


def _configure_dataloader_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def build_dataloaders(
    config: GlobalAttentionTrainConfig,
    *,
    allow_protocol_overrides: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    if config.split_manifest is not None:
        if config.experiment_config is None:
            raise ValueError("experiment_config is required when split_manifest is used")
        manifest_path = resolve_path(config.split_manifest)
        experiment = load_experiment_config(config.experiment_config)
        enforce_training_config(
            asdict(config),
            experiment,
            allow_overrides=config.allow_experiment_overrides or allow_protocol_overrides,
        )
        protocol_root = manifest_path.parent.parent
        audit_report = json.loads(
            (protocol_root / "audit" / "identity_audit.json").read_text(encoding="utf-8")
        )
        manifest = load_split_manifest(manifest_path)
        validation = validate_protocol_manifest(
            manifest,
            experiment,
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
        split_report: dict[str, object] = {
            "source": "frozen_manifest",
            "protocol": manifest_protocol,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(manifest_path),
            "validation": validation,
        }
    else:
        records = scan_deepship(config.data_root)
        split_segments, split_report = build_paper_split(
            records,
            clip_duration=config.clip_duration,
            samples_per_class=config.samples_per_class,
            train_per_class=config.train_per_class,
            val_per_class=config.val_per_class,
            test_per_class=config.test_per_class,
            seed=config.seed,
        )
        split_report["protocol"] = "segment_level"

    train_sampler = None
    if config.training_sampling == "fixed_anchor":
        dataset_kwargs = dict(
            data_root=config.data_root,
            sample_rate=TARGET_SAMPLE_RATE,
            clip_duration=config.clip_duration,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            highpass_freq=config.highpass_freq,
        )
        datasets = {
            split: DeepShipMelSegmentDataset(
                segments,
                return_index=(split != "train"),
                **dataset_kwargs,
            )
            for split, segments in split_segments.items()
        }
        split_report["training_sampling"] = {"id": "S0", "name": "fixed_anchor"}
    else:
        if config.split_manifest is None:
            raise ValueError("Dynamic sampling requires a frozen split manifest")
        assert config.train_samples_per_epoch is not None
        window_kwargs = dict(
            data_root=config.data_root,
            sample_rate=TARGET_SAMPLE_RATE,
            clip_duration=config.clip_duration,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
        )
        train_recordings = recording_representatives(split_segments["train"])
        datasets = {
            "train": DeepShipMelWindowDataset(
                train_recordings,
                return_index=False,
                dynamic_crop=True,
                **window_kwargs,
            ),
            "val": DeepShipMelWindowDataset(
                split_segments["val"],
                return_index=True,
                dynamic_crop=False,
                **window_kwargs,
            ),
            "test": DeepShipMelWindowDataset(
                split_segments["test"],
                return_index=True,
                dynamic_crop=False,
                **window_kwargs,
            ),
        }
        train_sampler = VesselBalancedEpochSampler(
            train_recordings,
            epoch_samples=config.train_samples_per_epoch,
            seed=config.seed,
        )
        split_report["training_sampling"] = {
            "id": "S2",
            "name": "vessel_balanced_dynamic",
            "samples_per_epoch": config.train_samples_per_epoch,
            "recording_representatives": len(train_recordings),
            "initial_exposure": train_sampler.exposure_report(),
        }
    pin_memory = str(config.device).startswith("cuda")
    worker_options: dict[str, object] = {}
    if config.num_workers > 0:
        worker_options = {
            "persistent_workers": True,
            "prefetch_factor": config.prefetch_factor,
            "worker_init_fn": _configure_dataloader_worker,
        }
    eval_batch_size = config.eval_batch_size or config.batch_size
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=train_generator,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
    }
    split_report["batch_sizes"] = {
        "train": config.batch_size,
        "validation": eval_batch_size,
        "test": eval_batch_size,
    }
    return dataloaders, split_report


class TrainingProgress:
    """Render one live terminal line while keeping redirected logs epoch-oriented."""

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
        width = self._available_width()
        if width is not None and len(line) > width:
            line = line[: max(1, width - 1)] + "…"
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


def _amp_context(config: GlobalAttentionTrainConfig):
    if not str(config.device).startswith("cuda") or config.precision == "fp32":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def validate_trainable_gradients(model: nn.Module) -> None:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(
            "Trainable parameters did not receive gradients. Examples: " + ", ".join(missing[:5])
        )
    nonfinite = [
        name
        for name, parameter in trainable
        if not bool(torch.isfinite(parameter.grad).all())
    ]
    if nonfinite:
        raise FloatingPointError(
            "Non-finite gradients detected. Examples: " + ", ".join(nonfinite[:5])
        )


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
    learning_rate: float,
    device: str,
) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    loss = f"{total_loss / samples:.4f}" if samples else "--"
    accuracy = f"{total_correct / samples:.4f}" if samples else "--"
    rate = f"{samples / elapsed:.2f}" if samples else "--"
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak = f"{torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3:.2f}GiB"
    else:
        peak = "n/a"
    percent = 100.0 * batch / max(1, batches)
    return (
        f"Epoch {epoch}/{epochs} | {phase} | batch={batch}/{batches} ({percent:.1f}%) "
        f"| avg_loss={loss} | avg_acc={accuracy} | lr={learning_rate:.2e} "
        f"| samples_per_sec={rate} | gpu_peak={peak}"
    )


def _prediction_row(
    dataset: DeepShipMelSegmentDataset | DeepShipMelWindowDataset,
    *,
    index: int,
    target: int,
    prediction: int,
    probabilities: list[float],
) -> dict[str, object]:
    segment = dataset.segments[index]
    return {
        "relative_path": segment.relative_path,
        "group_key": segment.group_key,
        "vessel_key": segment.vessel_key,
        "segment_index": segment.segment_index,
        "true_label": target,
        "predicted_label": prediction,
        "probabilities": probabilities,
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: GlobalAttentionTrainConfig,
    *,
    epoch: int,
    phase: str,
    progress: TrainingProgress,
    optimizer: torch.optim.Optimizer | None = None,
    learning_rate: float | None = None,
    max_batches: int | None = None,
    validate_gradients: bool = False,
) -> tuple[float, float, list[dict[str, object]]]:
    is_train = optimizer is not None
    model.train(is_train)
    batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    prediction_rows: list[dict[str, object]] = []
    dataset = dataloader.dataset
    if not is_train and not isinstance(
        dataset,
        (DeepShipMelSegmentDataset, DeepShipMelWindowDataset),
    ):
        raise TypeError("Validation dataset must be a DeepShip Mel dataset")
    use_cuda = str(config.device).startswith("cuda")
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(config.device))
        torch.cuda.reset_peak_memory_stats(torch.device(config.device))
    started_at = time.perf_counter()
    if optimizer is not None:
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
    elif learning_rate is None:
        learning_rate = 0.0
    progress.update(
        _progress_line(
            epoch=epoch,
            epochs=config.epochs,
            phase=phase,
            batch=0,
            batches=batches,
            samples=0,
            total_loss=0.0,
            total_correct=0,
            started_at=started_at,
            learning_rate=learning_rate,
            device=config.device,
        )
    )
    gradients_checked = False
    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= batches:
                break
            inputs, targets = batch[:2]
            valid_mel_frames = None
            if isinstance(dataset, DeepShipMelWindowDataset):
                indexes = batch[2] if not is_train else None
                valid_mel_frames = batch[-1]
            else:
                indexes = batch[2] if len(batch) > 2 else None
            inputs = inputs.to(config.device, non_blocking=use_cuda)
            targets = targets.to(config.device, non_blocking=use_cuda)
            time_padding_mask = None
            if valid_mel_frames is not None:
                time_padding_mask = feature_time_padding_mask(
                    valid_mel_frames.to(config.device, non_blocking=use_cuda),
                    total_input_time_steps=inputs.size(-1),
                )
            with _amp_context(config):
                logits = (
                    model(inputs)
                    if time_padding_mask is None
                    else model(inputs, time_padding_mask=time_padding_mask)
                )
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {phase} loss at batch {batch_index + 1}")
            if is_train:
                window_start = (batch_index // config.gradient_accumulation_steps) * (
                    config.gradient_accumulation_steps
                )
                accumulation_window = min(
                    config.gradient_accumulation_steps,
                    batches - window_start,
                )
                (loss / accumulation_window).backward()
                if validate_gradients and not gradients_checked:
                    validate_trainable_gradients(model)
                    gradients_checked = True
                should_step = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0
                    or batch_index + 1 == batches
                )
                if should_step:
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())
            total_samples += batch_size
            if not is_train:
                if indexes is None:
                    raise ValueError("Validation batches must expose segment indexes")
                probabilities = torch.softmax(logits.float(), dim=1).cpu().tolist()
                for target, prediction, probability, index in zip(
                    targets.cpu().tolist(),
                    predictions.cpu().tolist(),
                    probabilities,
                    indexes.tolist(),
                    strict=True,
                ):
                    prediction_rows.append(
                        _prediction_row(
                            dataset,
                            index=index,
                            target=target,
                            prediction=prediction,
                            probabilities=probability,
                        )
                    )
            completed = batch_index + 1
            if completed % config.log_interval == 0 or completed == batches:
                progress.update(
                    _progress_line(
                        epoch=epoch,
                        epochs=config.epochs,
                        phase=phase,
                        batch=completed,
                        batches=batches,
                        samples=total_samples,
                        total_loss=total_loss,
                        total_correct=total_correct,
                        started_at=started_at,
                        learning_rate=learning_rate,
                        device=config.device,
                    )
                )
    if total_samples == 0:
        raise RuntimeError(f"No samples were processed during {phase}")
    return total_loss / total_samples, total_correct / total_samples, prediction_rows


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: GlobalAttentionTrainConfig,
):
    if config.epochs <= 1:
        return None
    warmup_epochs = max(0, min(config.warmup_epochs, config.epochs - 1))
    cosine_epochs = config.epochs - warmup_epochs
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cosine_epochs),
        eta_min=config.min_learning_rate,
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _build_optimizer(
    model: nn.Module,
    config: GlobalAttentionTrainConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _model_kwargs(config: GlobalAttentionTrainConfig) -> dict[str, object]:
    return {
        "model_variant": config.model_variant,
        "d_model": config.attention_d_model,
        "num_heads": config.attention_num_heads,
        "ffn_expansion": config.attention_ffn_expansion,
        "position_kernel_size": config.attention_position_kernel_size,
        "temporal_kernel_size": config.attention_temporal_kernel_size,
        "dropout": config.attention_dropout,
        "gate_init": config.attention_gate_init,
    }


def _write_best_validation_artifacts(
    *,
    selection: dict[str, object],
    metrics: dict[str, dict[str, object] | None],
    segment_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
    vessel_rows: list[dict[str, object]],
    metrics_dir,
    predictions_dir,
    reports_dir,
) -> None:
    save_prediction_rows(
        segment_rows,
        predictions_dir / "validation_best_segment_predictions.csv",
        CLASS_NAMES,
    )
    save_prediction_rows(
        recording_rows,
        predictions_dir / "validation_best_recording_predictions.csv",
        CLASS_NAMES,
    )
    if vessel_rows:
        save_prediction_rows(
            vessel_rows,
            predictions_dir / "validation_best_vessel_predictions.csv",
            CLASS_NAMES,
        )
    for level in ("segment", "recording", "vessel"):
        level_metrics = metrics[level]
        if level_metrics is not None:
            save_metrics(level_metrics, metrics_dir / f"validation_best_{level}_metrics.json")
    (reports_dir / "validation_best_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _plot_best_validation(
    *,
    metrics: dict[str, dict[str, object] | None],
    protocol: str,
    variant: str,
    figures_dir,
) -> None:
    for level, title_level in (
        ("segment", "Segment"),
        ("recording", "Recording"),
        ("vessel", "Vessel-group"),
    ):
        level_metrics = metrics[level]
        if level_metrics is None:
            continue
        plot_confusion_matrix(
            level_metrics["confusion_matrix"],
            CLASS_NAMES,
            figures_dir / f"validation_best_{level}_confusion_matrix.png",
            title=f"DeepShip {protocol} {variant} Validation {title_level} Confusion Matrix",
        )


def train(config: GlobalAttentionTrainConfig) -> dict[str, object]:
    validate_config(config)
    set_seed(config.seed)
    g_series_experiment = load_g_series_experiment_config(config.g_series_config)
    g_series_config_mismatches = _validate_g_series_experiment_config(
        config,
        g_series_experiment,
    )
    base_protocol = g_series_experiment.get("base_protocol", {})
    allow_protocol_overrides = bool(
        isinstance(base_protocol, dict)
        and base_protocol.get("allow_training_overrides", False)
    )
    experiment = (
        load_experiment_config(config.experiment_config)
        if config.experiment_config is not None
        else None
    )
    config_mismatches = (
        enforce_training_config(
            asdict(config),
            experiment,
            allow_overrides=config.allow_experiment_overrides or allow_protocol_overrides,
        )
        if experiment is not None
        else []
    )
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
    predictions_dir = output_root / "predictions"
    reports_dir = output_root / "reports"
    for directory in (metrics_dir, figures_dir, models_dir, predictions_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    environment = runtime_environment()
    if (
        config.split_manifest is not None
        and not config.allow_experiment_overrides
        and environment["git_worktree_dirty"]
    ):
        details = "\n".join(str(item) for item in environment["git_status"])
        raise RuntimeError(
            "Formal G-series training requires a clean git worktree. Commit changes first:\n"
            + details
        )
    (reports_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataloaders, split_report = build_dataloaders(
        config,
        allow_protocol_overrides=allow_protocol_overrides,
    )
    protocol = str(split_report["protocol"])
    if config.split_manifest is None:
        split_segments = {
            split: list(dataloader.dataset.segments)
            for split, dataloader in dataloaders.items()
        }
        save_segment_split_manifest(reports_dir, split_segments, split_report)
    else:
        shutil.copyfile(
            resolve_path(config.split_manifest),
            reports_dir / "frozen_split_manifest.json",
        )
        (reports_dir / "split_validation.json").write_text(
            json.dumps(split_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    model: MACNNABaseClassifier = build_macnna_model(
        len(CLASS_NAMES),
        **_model_kwargs(config),
    ).to(config.device)
    if config.model_variant == "g0" and experiment is not None:
        expected_parameters = int(experiment["model"]["expected_num_parameters"])
        if model.num_parameters != expected_parameters:
            raise ValueError(
                f"G0 parameter count changed: expected {expected_parameters}, got {model.num_parameters}"
            )
    variants = g_series_experiment["variants"]
    if not isinstance(variants, dict) or not isinstance(variants.get(config.model_variant), dict):
        raise ValueError("G-series experiment config does not define the selected variant")
    expected_variant_parameters = int(
        variants[config.model_variant]["expected_num_parameters"]
    )
    if model.num_parameters != expected_variant_parameters:
        raise ValueError(
            f"{config.model_variant} parameter count changed: expected "
            f"{expected_variant_parameters}, got {model.num_parameters}"
        )
    model.eval()
    mel_frames = int(config.clip_duration * TARGET_SAMPLE_RATE // config.hop_length) + 1
    with torch.no_grad():
        example = torch.zeros(1, 1, config.n_mels, mel_frames, device=config.device)
        feature_shape = list(model.extract_features(example).shape)
        output_shape = list(model(example).shape)
    model.train()
    base_parameters = build_macnna_model(len(CLASS_NAMES), model_variant="g0").num_parameters
    model_report = {
        "model_variant": config.model_variant,
        "model_kwargs": _model_kwargs(config),
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "base_g0_parameters": base_parameters,
        "added_parameters": model.num_parameters - base_parameters,
        "example_input_shape": [1, 1, config.n_mels, mel_frames],
        "pre_pool_feature_shape": feature_shape,
        "post_cnn_time_steps": feature_shape[-1],
        "output_shape": output_shape,
        "input_context_seconds": config.clip_duration,
        "training_sampling": config.training_sampling,
        "optimizer": config.optimizer,
        "physical_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
    }
    (reports_dir / "model_report.json").write_text(
        json.dumps(model_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    history: dict[str, list[float | None]] = {
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
        "learning_rate": [],
    }
    history_path = reports_dir / "deepship_macnna_global_history.json"
    best_model_path = models_dir / "deepship_macnna_global_best.pt"
    last_model_path = models_dir / "deepship_macnna_global_last.pt"
    best_selection: dict[str, object] | None = None
    best_validation_metrics: dict[str, dict[str, object] | None] | None = None
    best_epoch = -1
    early_stopping_primary_value: float | None = None
    early_stopping_reference_epoch = -1
    start_epoch = 1

    if config.resume:
        if not last_model_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is unavailable: {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=config.device, weights_only=False)
        validate_resume_selection_state(
            checkpoint,
            manifest_sha256=str(split_report["manifest_sha256"]),
            protocol=protocol,
        )
        if checkpoint.get("model_kwargs") != _model_kwargs(config):
            raise ValueError("Resume checkpoint uses a different G-series model configuration")
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Resume checkpoint is missing a valid training configuration")
        resume_fields = (
            "training_sampling",
            "train_samples_per_epoch",
            "clip_duration",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation_steps",
            "max_grad_norm",
        )
        resume_mismatches = [
            field
            for field in resume_fields
            if checkpoint_config.get(field) != getattr(config, field)
        ]
        if resume_mismatches:
            raise ValueError(
                "Resume checkpoint uses a different training configuration: "
                + ", ".join(resume_mismatches)
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_selection = checkpoint["best_selection"]
        best_validation_metrics = checkpoint["best_validation_metrics"]
        best_epoch = int(checkpoint["best_epoch"])
        early_stopping_primary_value = float(checkpoint["early_stopping_primary_value"])
        early_stopping_reference_epoch = int(checkpoint["early_stopping_reference_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        print(f"Resuming from epoch {start_epoch}", flush=True)

    progress = TrainingProgress.auto()
    gradients_validated = False
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            epoch_started_at = time.perf_counter()
            if dataloaders["train"].generator is None:
                raise RuntimeError("Training DataLoader is missing its deterministic generator")
            dataloaders["train"].generator.manual_seed(config.seed + epoch)
            train_sampler = dataloaders["train"].sampler
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if hasattr(train_sampler, "exposure_report"):
                sampling_dir = reports_dir / "sampling_audits"
                sampling_dir.mkdir(parents=True, exist_ok=True)
                (sampling_dir / f"epoch_{epoch:03d}.json").write_text(
                    json.dumps(train_sampler.exposure_report(), indent=2) + "\n",
                    encoding="utf-8",
                )
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_acc, _ = run_epoch(
                model,
                dataloaders["train"],
                criterion,
                config,
                epoch=epoch,
                phase="train",
                progress=progress,
                optimizer=optimizer,
                max_batches=config.max_train_batches,
                validate_gradients=not gradients_validated,
            )
            gradients_validated = True
            val_loss, val_acc, segment_rows = run_epoch(
                model,
                dataloaders["val"],
                criterion,
                config,
                epoch=epoch,
                phase="val",
                progress=progress,
                learning_rate=current_lr,
                max_batches=config.max_eval_batches,
            )
            validation_metrics, recording_rows, vessel_rows = compute_grouped_metrics(
                segment_rows,
                CLASS_NAMES,
            )
            selection = build_validation_selection(protocol, validation_metrics, val_loss)
            segment_metrics = validation_metrics["segment"]
            recording_metrics = validation_metrics["recording"]
            vessel_metrics = validation_metrics["vessel"]
            if segment_metrics is None or recording_metrics is None:
                raise RuntimeError("Validation aggregation did not produce required metrics")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_segment_macro_f1"].append(float(segment_metrics["macro_f1"]))
            history["val_recording_acc"].append(float(recording_metrics["accuracy"]))
            history["val_recording_macro_f1"].append(float(recording_metrics["macro_f1"]))
            history["val_vessel_acc"].append(
                float(vessel_metrics["accuracy"]) if vessel_metrics is not None else None
            )
            history["val_vessel_macro_f1"].append(
                float(vessel_metrics["macro_f1"]) if vessel_metrics is not None else None
            )
            history["selection_primary_value"].append(float(selection["primary_value"]))
            history["learning_rate"].append(current_lr)
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

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
                _write_best_validation_artifacts(
                    selection=selection,
                    metrics=validation_metrics,
                    segment_rows=segment_rows,
                    recording_rows=recording_rows,
                    vessel_rows=vessel_rows,
                    metrics_dir=metrics_dir,
                    predictions_dir=predictions_dir,
                    reports_dir=reports_dir,
                )
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "model_kwargs": _model_kwargs(config),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "selection_schema_version": SELECTION_SCHEMA_VERSION,
                        "selection": selection,
                        "validation_metrics": validation_metrics,
                        "num_parameters": model.num_parameters,
                        "split_report": split_report,
                    },
                    best_model_path,
                )
            if best_selection is None or best_validation_metrics is None:
                raise RuntimeError("Best validation selection was not initialized")
            should_stop = should_stop_early(
                improved=meaningful_improvement,
                epoch=epoch,
                best_epoch=early_stopping_reference_epoch,
                patience=config.early_stopping_patience,
            )
            if scheduler is not None:
                scheduler.step()
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "config": asdict(config),
                    "model_kwargs": _model_kwargs(config),
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
            vessel_f1 = (
                f"{float(vessel_metrics['macro_f1']):.4f}" if vessel_metrics is not None else "n/a"
            )
            summary = (
                f"Epoch {epoch}/{config.epochs} | done | train_loss={train_loss:.4f} "
                f"| train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.4f} "
                f"| val_recording_f1={float(recording_metrics['macro_f1']):.4f} "
                f"| val_vessel_f1={vessel_f1} "
                f"| select={selection['rule']['name']}:{float(selection['primary_value']):.4f} "
                f"| best_select={float(best_selection['primary_value']):.4f} "
                f"| early_stop_wait={epoch - early_stopping_reference_epoch}/"
                f"{config.early_stopping_patience} "
                f"| time={time.perf_counter() - epoch_started_at:.1f}s"
            )
            if should_stop:
                summary += f" | early_stop=true | best_epoch={best_epoch}"
            progress.finish_epoch(summary)
            if should_stop:
                break
    finally:
        progress.close()

    if best_selection is None or best_validation_metrics is None:
        raise RuntimeError("Best validation result is unavailable")
    curve_history = {
        key: history[key] for key in ("train_loss", "val_loss", "train_acc", "val_acc")
    }
    plot_training_curves(
        curve_history,
        figures_dir / "deepship_macnna_global_training_curves.png",
        title=f"DeepShip {protocol} {config.model_variant} Training Curves",
    )
    _plot_best_validation(
        metrics=best_validation_metrics,
        protocol=protocol,
        variant=config.model_variant,
        figures_dir=figures_dir,
    )
    best_segment_metrics = best_validation_metrics["segment"]
    if best_segment_metrics is None:
        raise RuntimeError("Best validation segment metrics are unavailable")
    result = {
        "status": "validation_complete",
        "test_evaluated": False,
        "protocol": protocol,
        "model_variant": config.model_variant,
        "best_epoch": best_epoch,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_rule": validation_selection_rule(protocol),
        "best_selection": best_selection,
        "best_validation_metrics": best_validation_metrics,
        "best_val_acc": float(best_segment_metrics["accuracy"]),
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "fixed_architecture": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "clip_duration_seconds": config.clip_duration,
            "n_mels": config.n_mels,
            "mel_frames": mel_frames,
            "post_cnn_time_steps": feature_shape[-1],
            "kernel_sizes": list(THREE_BRANCH_KERNEL_SIZES),
            "model_kwargs": _model_kwargs(config),
        },
        "training_protocol": {
            "sampling": config.training_sampling,
            "train_samples_per_epoch": config.train_samples_per_epoch,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "physical_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        },
        "experiment_config_mismatches": config_mismatches,
        "g_series_config_mismatches": g_series_config_mismatches,
    }
    (reports_dir / "run_complete.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
