from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipMelDataset,
    DeepShipRandomCropDataset,
    build_segment_records,
    save_split_manifest,
    scan_deepship,
    stratified_split,
    summarize_records,
    summarize_segments,
)
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.models.ma_cnn_a import MACNNAClassifier
from src.pipelines.mel_ml.train_shipsear_cnn import (
    collect_predictions,
    get_default_device,
    run_epoch,
    set_seed,
)
from src.utils.optim import EpochWarmupCosineScheduler
from src.utils.pathing import resolve_path


@dataclass
class TrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs"
    sample_rate: int = 4000
    clip_duration: float = 3.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 2000.0
    highpass_freq: float | None = 50.0
    lowpass_freq: float | None = 2000.0
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_lr: float = 1e-5
    num_workers: int = 0
    use_augmentation: bool = True
    use_weighted_sampler: bool = True
    use_class_weights: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    time_shift_frames: int = 8
    time_mask_param: int = 16
    freq_mask_param: int = 6
    use_random_crop: bool = True
    max_segments_per_recording: int = 12
    kernel_sizes: tuple[int, ...] = (8, 16, 32)
    stem_channels: int = 32
    branch_channels: int = 24
    fused_channels: int = 96
    classifier_hidden: int = 64
    dropout: float = 0.2
    device: str = get_default_device()


def build_class_weights(stats: dict[str, object], device: str) -> torch.Tensor | None:
    train_segment_stats = stats.get("train_segments")
    if not isinstance(train_segment_stats, dict):
        return None
    class_counts = train_segment_stats.get("class_counts")
    if not isinstance(class_counts, dict):
        return None

    counts = torch.tensor(
        [float(class_counts.get(name, 0)) for name in CLASS_NAMES],
        dtype=torch.float32,
    ).clamp_min(1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights.to(device)


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler | None:
    labels: list[int] | None = None
    if hasattr(dataset, "segments"):
        labels = [int(seg.label_index) for seg in getattr(dataset, "segments")]
    elif hasattr(dataset, "_items") and hasattr(dataset, "records"):
        items = getattr(dataset, "_items")
        records = getattr(dataset, "records")
        labels = [int(records[idx].label_index) for idx in items]
    if not labels:
        return None

    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=len(CLASS_NAMES)).float()
    counts = counts.clamp_min(1.0)
    sample_weights = [float(1.0 / counts[label].item()) for label in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(config: TrainConfig) -> tuple[dict[str, DataLoader], dict[str, object]]:
    records = scan_deepship(config.data_root)
    train_records, val_records, test_records = stratified_split(
        records=records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    mel_kwargs = dict(
        sample_rate=config.sample_rate,
        clip_duration=config.clip_duration,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        highpass_freq=config.highpass_freq,
        lowpass_freq=config.lowpass_freq,
        time_shift_frames=config.time_shift_frames,
        time_mask_param=config.time_mask_param,
        freq_mask_param=config.freq_mask_param,
    )
    if config.use_random_crop:
        train_dataset: Dataset = DeepShipRandomCropDataset(
            train_records,
            max_segments_per_recording=config.max_segments_per_recording,
            augment=config.use_augmentation,
            **mel_kwargs,
        )
    else:
        train_segments = build_segment_records(train_records, clip_duration=config.clip_duration)
        train_dataset = DeepShipMelDataset(
            train_segments,
            augment=config.use_augmentation,
            cache_features=False,
            **mel_kwargs,
        )

    val_segments = build_segment_records(val_records, clip_duration=config.clip_duration)
    test_segments = build_segment_records(test_records, clip_duration=config.clip_duration)
    val_dataset = DeepShipMelDataset(val_segments, augment=False, cache_features=False, **mel_kwargs)
    test_dataset = DeepShipMelDataset(test_segments, augment=False, cache_features=False, **mel_kwargs)

    pin_memory = str(config.device).startswith("cuda")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_sampler = build_weighted_sampler(train_dataset) if config.use_weighted_sampler else None
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        ),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }

    split_dir = resolve_path(config.output_root) / "reports"
    save_split_manifest(split_dir, train_records, val_records, test_records)

    all_train_segments = build_segment_records(train_records, clip_duration=config.clip_duration)
    all_val_segments = build_segment_records(val_records, clip_duration=config.clip_duration)
    all_test_segments = build_segment_records(test_records, clip_duration=config.clip_duration)
    stats = {
        "full_recordings": summarize_records(records),
        "train_recordings": summarize_records(train_records),
        "val_recordings": summarize_records(val_records),
        "test_recordings": summarize_records(test_records),
        "train_segments": summarize_segments(all_train_segments),
        "val_segments": summarize_segments(all_val_segments),
        "test_segments": summarize_segments(all_test_segments),
        "train_effective_samples": len(train_dataset),
    }
    if config.use_random_crop:
        stats["random_crop"] = {
            "max_segments_per_recording": config.max_segments_per_recording,
        }
    (split_dir / "deepship_macnna_split_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return dataloaders, stats


def train(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    print("Preparing datasets and dataloaders...")
    dataloaders, stats = build_dataloaders(config)
    output_root = resolve_path(config.output_root)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    for directory in [metrics_dir, figures_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / "deepship_macnna_history.json"

    model = MACNNAClassifier(
        num_classes=len(CLASS_NAMES),
        kernel_sizes=config.kernel_sizes,
        stem_channels=config.stem_channels,
        branch_channels=config.branch_channels,
        fused_channels=config.fused_channels,
        classifier_hidden=config.classifier_hidden,
        dropout=config.dropout,
    ).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MA-CNN-A style CNN, {n_params:,} parameters")

    class_weights = build_class_weights(stats, config.device) if config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = EpochWarmupCosineScheduler(
        optimizer,
        total_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        warmup_start_factor=config.warmup_start_factor,
        min_lr=config.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda") if str(config.device).startswith("cuda") else None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = -1
    best_model_path = models_dir / "deepship_macnna_best.pt"

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            config.device,
            desc="train",
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            config.device,
            desc="val",
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"lr={current_lr:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}"
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_acc > best_val_acc + config.early_stopping_min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                best_model_path,
            )
        elif epoch - best_epoch >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = collect_predictions(model, dataloaders["test"], config.device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    save_metrics(metrics, metrics_dir / "deepship_macnna_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "deepship_macnna_confusion_matrix.png",
        title="DeepShip MA-CNN-A Style Confusion Matrix",
    )
    plot_training_curves(
        history,
        figures_dir / "deepship_macnna_training_curves.png",
        title="DeepShip MA-CNN-A Style Training Curves",
    )

    run_config = asdict(config)
    run_config["num_parameters"] = n_params
    run_config["best_val_acc"] = best_val_acc
    run_config["best_epoch"] = best_epoch
    (reports_dir / "deepship_macnna_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    print(
        f"Test accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    return metrics
