from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.belgian_ais import (
    BelgianMelDataset,
    BelgianRecord,
    ClassDateBalancedEpochSampler,
    STRICT_AUDIO_POLICY,
    STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION,
    record_from_dict,
    validate_fold_manifest,
)
from src.data.deepship import CLASS_NAMES
from src.evaluation.belgian_attention import date_balanced_metrics, stratified_metrics
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.evaluation.grouped_classification import save_prediction_rows
from src.models.ma_cnn_a import MACNNABaseClassifier, build_macnna_model
from src.pipelines.mel_ml.train_deepship_macnna import (
    atomic_torch_save,
    capture_rng_state,
    get_default_device,
    restore_rng_state,
    runtime_environment,
    set_seed,
)
from src.pipelines.mel_ml.train_deepship_macnna_global import (
    TrainingProgress,
    validate_trainable_gradients,
)
from src.utils.pathing import resolve_path


TARGET_SAMPLE_RATE = 16_000


@dataclass
class BelgianTrainConfig:
    data_root: str
    output_root: str
    split_manifest: str
    experiment_config: str = "configs/experiments/belgian_attention_v1.json"
    model_variant: str = "g0"
    seed: int = 42
    clip_duration: float = 10.0
    source_sample_rate: int = 48_000
    source_channel_policy: str = "fixed_channel_0"
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    attention_d_model: int = 128
    attention_num_heads: int = 4
    attention_ffn_expansion: int = 2
    attention_position_kernel_size: int = 9
    attention_temporal_kernel_size: int = 15
    attention_dropout: float = 0.1
    attention_gate_init: float = -2.0
    batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.005
    precision: str = "bf16"
    num_workers: int = 8
    prefetch_factor: int = 2
    log_interval: int = 100
    resume: bool = False
    allow_experiment_overrides: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    device: str = get_default_device()


def load_experiment(path: str) -> dict[str, object]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("experiment_id") != "belgian_attention_v1":
        raise ValueError("Unexpected Belgian experiment config")
    return payload


def _model_kwargs(config: BelgianTrainConfig) -> dict[str, object]:
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


def validate_config(config: BelgianTrainConfig, experiment: dict[str, object]) -> list[str]:
    if config.model_variant not in {"g0", "g1"}:
        raise ValueError("Belgian model_variant must be g0 or g1; G0-C is not in this protocol")
    if config.seed not in {42, 43, 44}:
        raise ValueError("Belgian model seed must be one of 42, 43, 44")
    if config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.epochs <= 0 or config.early_stopping_patience <= 0:
        raise ValueError("Epoch and patience values must be positive")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    features = experiment["features"]
    training = experiment["training"]
    adapter = experiment["shared_adapter"]
    if not all(isinstance(section, dict) for section in (features, training, adapter)):
        raise TypeError("Belgian experiment config sections are invalid")
    expected = {
        "clip_duration": features["clip_duration_seconds"],
        "source_sample_rate": features["source_sample_rate"],
        "source_channel_policy": features["source_channel_policy"],
        "n_fft": features["n_fft"],
        "win_length": features["win_length"],
        "hop_length": features["hop_length"],
        "n_mels": features["n_mels"],
        "attention_d_model": adapter["d_model"],
        "attention_num_heads": adapter["num_heads"],
        "attention_ffn_expansion": adapter["ffn_expansion"],
        "attention_position_kernel_size": adapter["position_kernel_size"],
        "attention_temporal_kernel_size": adapter["temporal_kernel_size"],
        "attention_dropout": adapter["dropout"],
        "attention_gate_init": adapter["gate_init"],
        "batch_size": training["batch_size"],
        "eval_batch_size": training["eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "max_grad_norm": training["max_grad_norm"],
        "min_learning_rate": training["min_learning_rate"],
        "warmup_epochs": training["warmup_epochs"],
        "early_stopping_patience": training["early_stopping_patience"],
        "early_stopping_min_delta": training["early_stopping_min_delta"],
        "precision": training["precision"],
        "num_workers": training["num_workers"],
    }
    mismatches = [
        f"{name}: expected {value!r}, got {getattr(config, name)!r}"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    for debug_field in ("max_train_batches", "max_eval_batches"):
        if getattr(config, debug_field) is not None:
            mismatches.append(f"{debug_field}: expected None, got {getattr(config, debug_field)!r}")
    if mismatches and not config.allow_experiment_overrides:
        raise ValueError(
            "Training configuration differs from frozen Belgian experiment:\n- "
            + "\n- ".join(mismatches)
        )
    return mismatches


def load_manifest(
    path: str,
    *,
    require_strict_audio: bool = True,
) -> tuple[dict[str, object], list[BelgianRecord], list[BelgianRecord]]:
    manifest_path = resolve_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_fold_manifest(manifest)
    if validation["status"] != "passed":
        raise ValueError(f"Frozen Belgian manifest validation failed: {validation}")
    if require_strict_audio and (
        manifest.get("schema_version") != STRICT_AUDIO_PROTOCOL_SCHEMA_VERSION
        or manifest.get("audio_policy") != STRICT_AUDIO_POLICY
    ):
        raise ValueError(
            "Formal Belgian runs require a schema-v2 manifest with the frozen exact-10s "
            "fixed-channel-0 audio policy"
        )
    train_records: list[BelgianRecord] = []
    val_records: list[BelgianRecord] = []
    for row in manifest["records"]:
        record = record_from_dict(row)
        if row["split"] == "train":
            train_records.append(record)
        elif row["split"] == "val":
            val_records.append(record)
        else:
            raise ValueError(f"Development manifest contains unsupported split: {row['split']}")
    return manifest, train_records, val_records


def _configure_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def build_dataloaders(config: BelgianTrainConfig):
    manifest, train_records, val_records = load_manifest(
        config.split_manifest,
        require_strict_audio=not config.allow_experiment_overrides,
    )
    dataset_kwargs = {
        "data_root": config.data_root,
        "sample_rate": TARGET_SAMPLE_RATE,
        "clip_duration": config.clip_duration,
        "source_sample_rate": config.source_sample_rate,
        "channel_policy": config.source_channel_policy,
        "n_fft": config.n_fft,
        "win_length": config.win_length,
        "hop_length": config.hop_length,
        "n_mels": config.n_mels,
    }
    train_dataset = BelgianMelDataset(train_records, return_index=False, **dataset_kwargs)
    val_dataset = BelgianMelDataset(val_records, return_index=True, **dataset_kwargs)
    sampler = ClassDateBalancedEpochSampler(train_records, seed=config.seed)
    worker_options: dict[str, object] = {}
    if config.num_workers > 0:
        worker_options = {
            "persistent_workers": True,
            "prefetch_factor": config.prefetch_factor,
            "worker_init_fn": _configure_worker,
        }
    pin_memory = str(config.device).startswith("cuda")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            **worker_options,
        ),
    }
    report = {
        "protocol": manifest["protocol"],
        "fold": manifest["fold"],
        "manifest_sha256": manifest["manifest_sha256"],
        "audio_policy": manifest.get("audio_policy"),
        "development_audio_inventory_sha256": manifest.get(
            "development_audio_inventory_sha256"
        ),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_samples_per_epoch": len(sampler),
        "initial_sampling_audit": sampler.audit(),
        "test_evaluated": False,
    }
    return loaders, report


def _amp_context(config: BelgianTrainConfig):
    if not str(config.device).startswith("cuda") or config.precision == "fp32":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _progress_line(
    config: BelgianTrainConfig,
    *,
    epoch: int,
    phase: str,
    batch: int,
    batches: int,
    samples: int,
    total_loss: float,
    total_correct: int,
    started_at: float,
    learning_rate: float,
) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    loss = f"{total_loss / samples:.4f}" if samples else "--"
    accuracy = f"{total_correct / samples:.4f}" if samples else "--"
    rate = f"{samples / elapsed:.2f}" if samples else "--"
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        peak = f"{torch.cuda.max_memory_allocated(torch.device(config.device)) / 1024**3:.2f}GiB"
    else:
        peak = "n/a"
    percent = 100.0 * batch / max(1, batches)
    return (
        f"Epoch {epoch}/{config.epochs} | {phase} | batch={batch}/{batches} ({percent:.1f}%) "
        f"| avg_loss={loss} | avg_acc={accuracy} | lr={learning_rate:.2e} "
        f"| samples_per_sec={rate} | gpu_peak={peak}"
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: BelgianTrainConfig,
    *,
    epoch: int,
    phase: str,
    progress: TrainingProgress,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    validate_gradients: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    rows: list[dict[str, object]] = []
    dataset = dataloader.dataset
    if not isinstance(dataset, BelgianMelDataset):
        raise TypeError("Belgian epoch requires BelgianMelDataset")
    use_cuda = str(config.device).startswith("cuda")
    if use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(config.device))
    started_at = time.perf_counter()
    learning_rate = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    gradients_checked = False
    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= batches:
                break
            inputs, targets = batch[:2]
            indexes = batch[2] if not is_train else None
            inputs = inputs.to(config.device, non_blocking=use_cuda)
            targets = targets.to(config.device, non_blocking=use_cuda)
            with _amp_context(config):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {phase} loss at batch {batch_index + 1}")
            if is_train:
                window_start = (batch_index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            predictions = logits.argmax(dim=1)
            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == targets).sum().item())
            total_samples += batch_size
            if not is_train:
                if indexes is None:
                    raise RuntimeError("Belgian validation requires dataset indexes")
                probabilities = torch.softmax(logits.float(), dim=1).cpu().tolist()
                for target, prediction, probability, index in zip(
                    targets.cpu().tolist(),
                    predictions.cpu().tolist(),
                    probabilities,
                    indexes.tolist(),
                    strict=True,
                ):
                    record = dataset.records[index]
                    rows.append(
                        {
                            "relative_path": record.relative_path,
                            "calendar_date": record.calendar_date,
                            "station": record.station,
                            "distance_km": record.distance_km,
                            "official_split": record.official_split,
                            "true_label": target,
                            "predicted_label": prediction,
                            "probabilities": probability,
                        }
                    )
            completed = batch_index + 1
            if completed % config.log_interval == 0 or completed == batches:
                progress.update(
                    _progress_line(
                        config,
                        epoch=epoch,
                        phase=phase,
                        batch=completed,
                        batches=batches,
                        samples=total_samples,
                        total_loss=total_loss,
                        total_correct=total_correct,
                        started_at=started_at,
                        learning_rate=learning_rate,
                    )
                )
    if total_samples == 0:
        raise RuntimeError(f"No Belgian samples processed during {phase}")
    return total_loss / total_samples, total_correct / total_samples, rows


def _scheduler(optimizer: torch.optim.Optimizer, config: BelgianTrainConfig):
    warmup_epochs = max(0, min(config.warmup_epochs, config.epochs - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - warmup_epochs),
        eta_min=config.min_learning_rate,
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _optimizer(model: nn.Module, config: BelgianTrainConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _validation_metrics(rows: list[dict[str, object]]):
    clip = compute_metrics(
        [int(row["true_label"]) for row in rows],
        [int(row["predicted_label"]) for row in rows],
        CLASS_NAMES,
    )
    date = date_balanced_metrics(rows, CLASS_NAMES)
    slices = stratified_metrics(rows, CLASS_NAMES)
    return {"clip": clip, "date_balanced": date, "stratified": slices}


def train(config: BelgianTrainConfig) -> dict[str, object]:
    experiment = load_experiment(config.experiment_config)
    mismatches = validate_config(config, experiment)
    set_seed(config.seed)
    output_root = resolve_path(config.output_root)
    if config.resume and (output_root / "reports" / "run_complete.json").is_file():
        raise RuntimeError(f"Belgian run is already complete: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not config.resume:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    directories = {
        name: output_root / name
        for name in ("metrics", "figures", "models", "predictions", "reports")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment()
    if not config.allow_experiment_overrides and environment["git_worktree_dirty"]:
        raise RuntimeError("Formal Belgian training requires a clean git worktree")
    (directories["reports"] / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directories["reports"] / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dataloaders, split_report = build_dataloaders(config)
    shutil.copyfile(
        resolve_path(config.split_manifest),
        directories["reports"] / "frozen_split_manifest.json",
    )
    (directories["reports"] / "split_validation.json").write_text(
        json.dumps(split_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model: MACNNABaseClassifier = build_macnna_model(
        len(CLASS_NAMES), **_model_kwargs(config)
    ).to(config.device)
    variants = experiment["variants"]
    expected_parameters = int(variants[config.model_variant]["expected_num_parameters"])
    if model.num_parameters != expected_parameters:
        raise ValueError(
            f"{config.model_variant} parameter count changed: "
            f"expected {expected_parameters}, got {model.num_parameters}"
        )
    mel_frames = int(config.clip_duration * TARGET_SAMPLE_RATE // config.hop_length) + 1
    model.eval()
    with torch.no_grad():
        example = torch.zeros(1, 1, config.n_mels, mel_frames, device=config.device)
        feature_shape = list(model.extract_features(example).shape)
        output_shape = list(model(example).shape)
    model.train()
    model_report = {
        "model_variant": config.model_variant,
        "model_kwargs": _model_kwargs(config),
        "num_parameters": model.num_parameters,
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "example_input_shape": [1, 1, config.n_mels, mel_frames],
        "pre_pool_feature_shape": feature_shape,
        "output_shape": output_shape,
        "test_evaluated": False,
    }
    (directories["reports"] / "model_report.json").write_text(
        json.dumps(model_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_clip_macro_f1": [],
        "val_date_macro_f1": [],
        "learning_rate": [],
    }
    history_path = directories["reports"] / "belgian_history.json"
    best_path = directories["models"] / "belgian_best.pt"
    last_path = directories["models"] / "belgian_last.pt"
    best_epoch = -1
    best_value = -math.inf
    best_val_loss = math.inf
    early_reference = -math.inf
    early_reference_epoch = 0
    best_metrics: dict[str, object] | None = None
    start_epoch = 1
    if config.resume:
        checkpoint = torch.load(last_path, map_location=config.device, weights_only=False)
        if checkpoint["manifest_sha256"] != split_report["manifest_sha256"]:
            raise ValueError("Resume checkpoint uses another Belgian manifest")
        if checkpoint["model_kwargs"] != _model_kwargs(config):
            raise ValueError("Resume checkpoint uses another Belgian model")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_epoch = int(checkpoint["best_epoch"])
        best_value = float(checkpoint["best_value"])
        best_val_loss = float(checkpoint["best_val_loss"])
        early_reference = float(checkpoint["early_reference"])
        early_reference_epoch = int(checkpoint["early_reference_epoch"])
        best_metrics = checkpoint["best_metrics"]
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
    progress = TrainingProgress.auto()
    gradients_validated = False
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            sampler = dataloaders["train"].sampler
            if not isinstance(sampler, ClassDateBalancedEpochSampler):
                raise TypeError("Belgian training sampler is invalid")
            sampler.set_epoch(epoch)
            sampling_dir = directories["reports"] / "sampling_audits"
            sampling_dir.mkdir(parents=True, exist_ok=True)
            (sampling_dir / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(sampler.audit(), indent=2) + "\n", encoding="utf-8"
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
            val_loss, val_acc, rows = run_epoch(
                model,
                dataloaders["val"],
                criterion,
                config,
                epoch=epoch,
                phase="val",
                progress=progress,
                max_batches=config.max_eval_batches,
            )
            metrics = _validation_metrics(rows)
            date_value = float(metrics["date_balanced"]["macro_f1"])
            clip_value = float(metrics["clip"]["macro_f1"])
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_clip_macro_f1"].append(clip_value)
            history["val_date_macro_f1"].append(date_value)
            history["learning_rate"].append(current_lr)
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            checkpoint_improved = date_value > best_value or (
                date_value == best_value and val_loss < best_val_loss
            )
            meaningful = date_value >= early_reference + config.early_stopping_min_delta
            if meaningful:
                early_reference = date_value
                early_reference_epoch = epoch
            if checkpoint_improved:
                best_epoch = epoch
                best_value = date_value
                best_val_loss = val_loss
                best_metrics = metrics
                save_prediction_rows(
                    rows,
                    directories["predictions"] / "validation_best_predictions.csv",
                    CLASS_NAMES,
                )
                save_metrics(metrics["clip"], directories["metrics"] / "validation_best_clip_metrics.json")
                save_metrics(
                    metrics["date_balanced"],
                    directories["metrics"] / "validation_best_date_balanced_metrics.json",
                )
                (directories["metrics"] / "validation_best_stratified_metrics.json").write_text(
                    json.dumps(metrics["stratified"], indent=2) + "\n", encoding="utf-8"
                )
                (directories["reports"] / "validation_best_selection.json").write_text(
                    json.dumps(
                        {
                            "metric": "validation date-balanced macro-F1",
                            "primary_value": date_value,
                            "val_loss_tiebreaker": val_loss,
                            "epoch": epoch,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "model_kwargs": _model_kwargs(config),
                        "manifest_sha256": split_report["manifest_sha256"],
                        "epoch": epoch,
                        "validation_metrics": metrics,
                        "test_evaluated": False,
                    },
                    best_path,
                )
            if best_metrics is None:
                raise RuntimeError("Belgian best validation metrics were not initialized")
            scheduler.step()
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(config),
                    "model_kwargs": _model_kwargs(config),
                    "manifest_sha256": split_report["manifest_sha256"],
                    "epoch": epoch,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_value": best_value,
                    "best_val_loss": best_val_loss,
                    "early_reference": early_reference,
                    "early_reference_epoch": early_reference_epoch,
                    "best_metrics": best_metrics,
                    "rng_state": capture_rng_state(),
                    "test_evaluated": False,
                },
                last_path,
            )
            wait = epoch - early_reference_epoch
            stop = wait >= config.early_stopping_patience
            summary = (
                f"Epoch {epoch}/{config.epochs} | done | train_loss={train_loss:.4f} "
                f"| train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.4f} | val_clip_f1={clip_value:.4f} "
                f"| val_date_f1={date_value:.4f} | best_date_f1={best_value:.4f} "
                f"| early_stop_wait={wait}/{config.early_stopping_patience} "
                f"| time={time.perf_counter() - started:.1f}s"
            )
            if stop:
                summary += f" | early_stop=true | best_epoch={best_epoch}"
            progress.finish_epoch(summary)
            if stop:
                break
    finally:
        progress.close()
    if best_metrics is None:
        raise RuntimeError("Belgian best validation result is unavailable")
    plot_training_curves(
        {key: history[key] for key in ("train_loss", "val_loss", "train_acc", "val_acc")},
        directories["figures"] / "belgian_training_curves.png",
        title=f"Belgian fold {split_report['fold']} {config.model_variant}",
    )
    plot_confusion_matrix(
        best_metrics["clip"]["confusion_matrix"],
        CLASS_NAMES,
        directories["figures"] / "validation_best_clip_confusion_matrix.png",
        title=f"Belgian fold {split_report['fold']} {config.model_variant} Validation",
    )
    result = {
        "status": "validation_complete",
        "test_evaluated": False,
        "protocol": split_report["protocol"],
        "fold": split_report["fold"],
        "model_seed": config.seed,
        "model_variant": config.model_variant,
        "best_epoch": best_epoch,
        "best_selection": {
            "metric": "validation date-balanced macro-F1",
            "primary_value": best_value,
            "val_loss_tiebreaker": best_val_loss,
        },
        "best_validation_metrics": best_metrics,
        "num_parameters": model.num_parameters,
        "manifest_sha256": split_report["manifest_sha256"],
        "training_samples_per_epoch": split_report["train_samples_per_epoch"],
        "experiment_config_mismatches": mismatches,
    }
    (directories["reports"] / "run_complete.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
