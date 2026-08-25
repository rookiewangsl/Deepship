from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.deepship import CLASS_NAMES, SegmentRecord, segment_record_from_dict
from src.data.deepship_audit import load_experiment_config
from src.data.deepship_protocol_validation import load_split_manifest, validate_protocol_manifest
from src.data.deepship_waveform import DeepShipWaveformSegmentDataset
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.evaluation.grouped_classification import (
    aggregate_recording_predictions,
    aggregate_vessel_predictions,
    save_prediction_rows,
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
    gradient_checkpointing: bool = True
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 50
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 8
    precision: str = "fp16"
    seed: int = 42
    num_workers: int = 4
    resume: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
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
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
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
    datasets = {
        split: DeepShipWaveformSegmentDataset(
            segments,
            data_root=config.data_root,
            sample_rate=config.sample_rate,
            clip_duration=config.clip_duration,
            normalize=config.normalize_waveform,
            remove_dc=config.remove_dc,
            return_index=(split == "test"),
        )
        for split, segments in split_segments.items()
    }
    pin_memory = str(config.device).startswith("cuda")
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return dataloaders, split_report


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
) -> SequentialLR | CosineAnnealingLR | None:
    if config.epochs <= 1:
        return None
    warmup_epochs = max(0, min(config.warmup_epochs, config.epochs - 1))
    cosine_epochs = config.epochs - warmup_epochs
    if warmup_epochs == 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_epochs),
            eta_min=config.min_learning_rate,
        )
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, cosine_epochs),
                eta_min=config.min_learning_rate,
            ),
        ],
        milestones=[warmup_epochs],
    )


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


def run_epoch(
    model: Wav2Vec2ConformerClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: ConformerTrainConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
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
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= effective_batches:
                break
            input_values, attention_mask, targets = _move_batch(batch, config.device)
            with _autocast_context(config):
                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, targets)

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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * input_values.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += input_values.size(0)
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
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            for target, prediction, probability, index in zip(
                targets.cpu().tolist(),
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
    shutil.copyfile(resolve_path(config.split_manifest), reports_dir / "frozen_split_manifest.json")
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
        "backbone_config": backbone_config,
    }
    (reports_dir / "model_report.json").write_text(
        json.dumps(model_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_scaler = str(config.device).startswith("cuda") and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "encoder_learning_rate": [],
        "head_learning_rate": [],
    }
    history_path = reports_dir / "deepship_conformer_history.json"
    best_model_path = models_dir / "deepship_conformer_best.pt"
    last_model_path = models_dir / "deepship_conformer_last.pt"
    best_val_acc = -1.0
    best_epoch = -1
    start_epoch = 1

    if config.resume:
        if not last_model_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is unavailable: {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=config.device, weights_only=False)
        if checkpoint.get("split_manifest_sha256") != split_report["manifest_sha256"]:
            raise ValueError("Resume checkpoint uses a different split manifest")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_val_acc = float(checkpoint["best_val_acc"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            config,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.max_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            config,
            max_batches=config.max_eval_batches,
        )
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
        history["encoder_learning_rate"].append(encoder_lr)
        history["head_learning_rate"].append(head_lr)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "num_parameters": model.num_parameters,
                    "num_trainable_parameters": model.num_trainable_parameters,
                    "split_report": split_report,
                },
                best_model_path,
            )
        should_stop = not improved and epoch - best_epoch >= config.early_stopping_patience
        if scheduler is not None:
            scheduler.step()
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "config": asdict(config),
                "epoch": epoch,
                "history": history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "split_manifest_sha256": split_report["manifest_sha256"],
                "rng_state": capture_rng_state(),
            },
            last_model_path,
        )
        print(
            f"Epoch {epoch}/{config.epochs} | encoder_lr={encoder_lr:.3g} "
            f"head_lr={head_lr:.3g} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} | best_val_acc={best_val_acc:.4f}",
            flush=True,
        )
        if should_stop:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})", flush=True)
            break

    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    segment_predictions = collect_prediction_rows(
        model,
        dataloaders["test"],
        config,
        max_batches=config.max_eval_batches,
    )
    recording_predictions = aggregate_recording_predictions(segment_predictions)
    vessel_predictions = aggregate_vessel_predictions(recording_predictions)
    segment_metrics = compute_metrics(
        [int(row["true_label"]) for row in segment_predictions],
        [int(row["predicted_label"]) for row in segment_predictions],
        CLASS_NAMES,
    )
    recording_metrics = compute_metrics(
        [int(row["true_label"]) for row in recording_predictions],
        [int(row["predicted_label"]) for row in recording_predictions],
        CLASS_NAMES,
    )
    vessel_metrics = (
        compute_metrics(
            [int(row["true_label"]) for row in vessel_predictions],
            [int(row["predicted_label"]) for row in vessel_predictions],
            CLASS_NAMES,
        )
        if vessel_predictions
        else None
    )
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

    protocol = str(split_report["protocol"])
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
            title=f"DeepShip {protocol} Conformer Vessel-group Confusion Matrix",
        )
    curve_history = {
        key: history[key] for key in ("train_loss", "val_loss", "train_acc", "val_acc")
    }
    plot_training_curves(
        curve_history,
        figures_dir / "deepship_conformer_training_curves.png",
        title=f"DeepShip {protocol} Wav2Vec2-Conformer Training Curves",
    )

    result = {
        "protocol": protocol,
        "segment_metrics": segment_metrics,
        "recording_metrics": recording_metrics,
        "vessel_metrics": vessel_metrics,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": model.num_trainable_parameters,
    }
    (reports_dir / "run_complete.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
