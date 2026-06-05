from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from torch.utils.data import DataLoader

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
from src.data.shipsear import PrecomputedMelDataset
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.models.mel_cnn import MelCNNClassifier
from src.pipelines.mel_ml.train_shipsear_cnn import (
    collect_predictions,
    get_default_device,
    run_epoch,
    set_seed,
)
import torch
from torch import nn


@dataclass
class TrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs"
    sample_rate: int = 4000
    clip_duration: float = 5.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 2000.0
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_threshold: float = 1e-3
    scheduler_min_lr: float = 1e-6
    num_workers: int = 0
    use_augmentation: bool = False
    cache_features: bool = False
    precomputed_root: str | None = "outputs/precomputed/deepship_mel"
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    time_shift_frames: int = 8
    time_mask_param: int = 30
    freq_mask_param: int = 8
    use_random_crop: bool = True
    max_segments_per_recording: int = 12
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
    )
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights.to(device)


def build_precomputed_dataloaders(
    config: TrainConfig,
) -> tuple[dict[str, DataLoader], dict[str, object]] | None:
    if not config.precomputed_root:
        return None
    precomputed_root = Path(config.precomputed_root)
    required_files = {
        split: precomputed_root / f"{split}.pt" for split in ["train", "val", "test"]
    }
    if not all(path.exists() for path in required_files.values()):
        return None

    datasets = {
        split: PrecomputedMelDataset(
            path,
            augment=(split == "train" and config.use_augmentation),
            time_shift_frames=config.time_shift_frames,
            time_mask_param=config.time_mask_param,
            freq_mask_param=config.freq_mask_param,
        )
        for split, path in required_files.items()
    }
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
        )
        for split, dataset in datasets.items()
    }

    stats_path = precomputed_root / "deepship_split_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {
            split: {"num_samples": len(dataset)}
            for split, dataset in datasets.items()
        }
    print(f"Using precomputed Mel features from {precomputed_root}")
    return dataloaders, stats


def build_dataloaders(config: TrainConfig) -> tuple[dict[str, DataLoader], dict[str, object]]:
    # When random_crop is disabled, try to use fully precomputed features.
    if not config.use_random_crop:
        maybe_precomputed = build_precomputed_dataloaders(config)
        if maybe_precomputed is not None:
            return maybe_precomputed

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
        time_shift_frames=config.time_shift_frames,
        time_mask_param=config.time_mask_param,
        freq_mask_param=config.freq_mask_param,
    )

    # --- Training dataset ---
    if config.use_random_crop:
        train_dataset = DeepShipRandomCropDataset(
            train_records,
            max_segments_per_recording=config.max_segments_per_recording,
            augment=config.use_augmentation,
            **mel_kwargs,
        )
        print(
            f"Random-crop training: {len(train_records)} recordings "
            f"→ {len(train_dataset)} samples "
            f"(max {config.max_segments_per_recording} segments/recording)"
        )
    else:
        train_segments = build_segment_records(
            train_records, clip_duration=config.clip_duration,
        )
        train_dataset = DeepShipMelDataset(
            train_segments,
            augment=config.use_augmentation,
            cache_features=config.cache_features and not config.use_augmentation,
            **mel_kwargs,
        )

    # --- Val / Test datasets (always deterministic segments) ---
    # Try precomputed features for val/test first.
    val_test_precomputed = _try_precomputed_val_test(config)
    if val_test_precomputed is not None:
        val_dataset, test_dataset = val_test_precomputed
    else:
        val_segments = build_segment_records(
            val_records, clip_duration=config.clip_duration,
        )
        test_segments = build_segment_records(
            test_records, clip_duration=config.clip_duration,
        )
        val_dataset = DeepShipMelDataset(
            val_segments,
            augment=False,
            cache_features=config.cache_features,
            **mel_kwargs,
        )
        test_dataset = DeepShipMelDataset(
            test_segments,
            augment=False,
            cache_features=config.cache_features,
            **mel_kwargs,
        )

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
        )
        for split, dataset in datasets.items()
    }

    split_dir = Path(config.output_root) / "reports"
    save_split_manifest(split_dir, train_records, val_records, test_records)

    # Build segment records for statistics only (not for training data).
    all_train_segments = build_segment_records(
        train_records, clip_duration=config.clip_duration,
    )
    all_val_segments = build_segment_records(
        val_records, clip_duration=config.clip_duration,
    )
    all_test_segments = build_segment_records(
        test_records, clip_duration=config.clip_duration,
    )
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
    (split_dir / "deepship_split_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return dataloaders, stats


def _try_precomputed_val_test(
    config: TrainConfig,
) -> tuple[Dataset, Dataset] | None:
    """Load precomputed val/test bundles if they exist."""
    if not config.precomputed_root:
        return None
    precomputed_root = Path(config.precomputed_root)
    val_path = precomputed_root / "val.pt"
    test_path = precomputed_root / "test.pt"
    if not (val_path.exists() and test_path.exists()):
        return None
    from src.data.shipsear import PrecomputedMelDataset
    val_dataset = PrecomputedMelDataset(val_path, augment=False)
    test_dataset = PrecomputedMelDataset(test_path, augment=False)
    print(f"Using precomputed val/test features from {precomputed_root}")
    return val_dataset, test_dataset


def train(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    print("Preparing datasets and dataloaders...")
    dataloaders, stats = build_dataloaders(config)
    output_root = Path(config.output_root)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    for directory in [metrics_dir, figures_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / "deepship_mel_cnn_history.json"

    model = MelCNNClassifier(
        num_classes=len(CLASS_NAMES),
        channels=(16, 32, 64, 128),
        drop_rates=(0.0, 0.05, 0.1, 0.1),
        classifier_dropout=0.35,
        classifier_hidden=64,
    ).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 4-block CNN, {n_params:,} parameters")
    class_weights = build_class_weights(stats, config.device)
    if class_weights is not None:
        print(f"Using class-weighted CrossEntropyLoss with weights={class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        min_lr=config.scheduler_min_lr,
    )

    best_val_loss = float("inf")
    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = models_dir / "deepship_mel_cnn_best.pt"
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    print(f"Starting training on device={config.device} for {config.epochs} epochs...")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=config.device,
            desc=f"Epoch {epoch:03d}/{config.epochs:03d} train",
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=config.device,
            desc=f"Epoch {epoch:03d}/{config.epochs:03d} val",
            optimizer=None,
        )
        val_y_true, val_y_pred = collect_predictions(model, dataloaders["val"], config.device)
        val_metrics = compute_metrics(val_y_true, val_y_pred, CLASS_NAMES)
        val_macro_f1 = val_metrics["macro_f1"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_training_curves(
            history,
            figures_dir / "deepship_mel_cnn_training_curves.png",
            title="DeepShip Mel+CNN Training Curves",
        )
        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.6g}"
        )
        improved = val_macro_f1 > (best_val_macro_f1 + config.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            print(
                f"New best model saved at epoch {epoch:03d} "
                f"with val_macro_f1={val_macro_f1:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "class_names": CLASS_NAMES,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation macro-F1 improvement for {epochs_without_improvement} epoch(s) "
                f"(best epoch: {best_epoch:03d}, best val_macro_f1: {best_val_macro_f1:.4f})"
            )
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch:03d}. "
                    f"Best epoch was {best_epoch:03d}."
                )
                break

    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = collect_predictions(model, dataloaders["test"], config.device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["split_stats"] = stats
    metrics["config"] = asdict(config)

    save_metrics(metrics, metrics_dir / "deepship_mel_cnn_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "deepship_mel_cnn_confusion_matrix.png",
        title="DeepShip Mel+CNN Confusion Matrix",
    )
    (reports_dir / "deepship_mel_cnn_run_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    print(
        f"Final test metrics: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}"
    )
    return metrics
