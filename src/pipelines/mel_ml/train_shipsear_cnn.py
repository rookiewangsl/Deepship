from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.shipsear import (
    CLASS_NAMES,
    PrecomputedMelDataset,
    ShipsearMelDataset,
    save_split_manifest,
    scan_shipsear,
    stratified_split,
    summarize_records,
)
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.models.mel_cnn import MelCNNClassifier


def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    data_root: str = "ShipsEar"
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
    weight_decay: float = 1e-4
    num_workers: int = 0
    use_augmentation: bool = False
    cache_features: bool = True
    precomputed_root: str | None = "outputs/precomputed/shipsear_mel"
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    time_shift_frames: int = 8
    time_mask_param: int = 12
    freq_mask_param: int = 8
    device: str = get_default_device()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    stats_path = precomputed_root / "shipsear_split_stats.json"
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
    maybe_precomputed = build_precomputed_dataloaders(config)
    if maybe_precomputed is not None:
        return maybe_precomputed

    records = scan_shipsear(config.data_root)
    train_records, val_records, test_records = stratified_split(
        records=records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    dataset_kwargs = dict(
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
    datasets = {
        "train": ShipsearMelDataset(
            train_records,
            augment=config.use_augmentation,
            cache_features=config.cache_features and not config.use_augmentation,
            **dataset_kwargs,
        ),
        "val": ShipsearMelDataset(
            val_records,
            augment=False,
            cache_features=config.cache_features,
            **dataset_kwargs,
        ),
        "test": ShipsearMelDataset(
            test_records,
            augment=False,
            cache_features=config.cache_features,
            **dataset_kwargs,
        ),
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

    split_dir = Path(config.output_root) / "reports"
    save_split_manifest(split_dir, train_records, val_records, test_records)
    stats = {
        "full": summarize_records(records),
        "train": summarize_records(train_records),
        "val": summarize_records(val_records),
        "test": summarize_records(test_records),
    }
    (split_dir / "shipsear_split_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return dataloaders, stats


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    with torch.set_grad_enabled(is_train):
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
            )

    progress_bar.close()

    return total_loss / total_samples, total_correct / total_samples


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.tolist())
    return y_true, y_pred


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
    history_path = reports_dir / "shipsear_mel_cnn_history.json"

    model = MelCNNClassifier(num_classes=len(CLASS_NAMES)).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = models_dir / "shipsear_mel_cnn_best.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_training_curves(
            history,
            figures_dir / "shipsear_mel_cnn_training_curves.png",
            title="ShipsEar Mel+CNN Training Curves",
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        improved = val_loss < (best_val_loss - config.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            print(
                f"New best model saved at epoch {epoch:03d} "
                f"with val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
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
                f"No validation-loss improvement for {epochs_without_improvement} epoch(s) "
                f"(best epoch: {best_epoch:03d}, best val_loss: {best_val_loss:.4f})"
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

    save_metrics(metrics, metrics_dir / "shipsear_mel_cnn_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "shipsear_mel_cnn_confusion_matrix.png",
        title="ShipsEar Mel+CNN Confusion Matrix",
    )
    (reports_dir / "shipsear_mel_cnn_run_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    print(
        f"Final test metrics: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}"
    )
    return metrics
