from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipMelSegmentDataset,
    build_paper_split,
    save_segment_split_manifest,
    scan_deepship,
)
from src.evaluation.classification import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.models.ma_cnn_a import MACNNAClassifier, THREE_BRANCH_KERNEL_SIZES
from src.utils.pathing import resolve_path


TARGET_SAMPLE_RATE = 16000


def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs/deepship_macnna_paper"
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
    epochs: int = 100
    learning_rate: float = 1e-2
    early_stopping_patience: int = 10
    branch_channels: int = 88
    device: str = get_default_device()


def build_dataloaders(config: TrainConfig) -> tuple[dict[str, DataLoader], dict[str, object]]:
    records = scan_deepship(config.data_root)
    split_segments, split_stats = build_paper_split(
        records,
        clip_duration=config.clip_duration,
        samples_per_class=config.samples_per_class,
        train_per_class=config.train_per_class,
        val_per_class=config.val_per_class,
        test_per_class=config.test_per_class,
        seed=config.seed,
    )
    dataset_kwargs = dict(
        sample_rate=TARGET_SAMPLE_RATE,
        clip_duration=config.clip_duration,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        highpass_freq=config.highpass_freq,
    )
    datasets = {
        split: DeepShipMelSegmentDataset(segments, **dataset_kwargs)
        for split, segments in split_segments.items()
    }
    pin_memory = str(config.device).startswith("cuda")
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }
    save_segment_split_manifest(resolve_path(config.output_root) / "reports", split_segments, split_stats)
    return dataloaders, split_stats


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
    use_amp = str(device).startswith("cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()

    progress = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(is_train):
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=use_amp)
            targets = targets.to(device, non_blocking=use_amp)

            with amp_ctx:
                logits = model(inputs)
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    progress.close()
    return total_loss / total_samples, total_correct / total_samples


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    use_amp = str(device).startswith("cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="test", leave=False, dynamic_ncols=True):
            inputs = inputs.to(device, non_blocking=use_amp)
            with amp_ctx:
                logits = model(inputs)
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def train(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    output_root = resolve_path(config.output_root)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    for directory in [metrics_dir, figures_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataloaders, split_stats = build_dataloaders(config)
    model = MACNNAClassifier(
        num_classes=len(CLASS_NAMES),
        branch_channels=config.branch_channels,
    ).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    history_path = reports_dir / "deepship_macnna_history.json"
    best_model_path = models_dir / "deepship_macnna_best.pt"
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            config.device,
            desc="train",
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            config.device,
            desc="val",
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "num_parameters": model.num_parameters,
                    "split_stats": split_stats,
                },
                best_model_path,
            )
        elif epoch - best_epoch >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = collect_predictions(model, dataloaders["test"], config.device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    save_metrics(metrics, metrics_dir / "deepship_macnna_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "deepship_macnna_confusion_matrix.png",
        title="DeepShip Paper Reproduction Confusion Matrix",
    )
    plot_training_curves(
        history,
        figures_dir / "deepship_macnna_training_curves.png",
        title="DeepShip Paper Reproduction Training Curves",
    )

    run_config = asdict(config)
    run_config["num_parameters"] = model.num_parameters
    run_config["best_val_acc"] = best_val_acc
    run_config["best_epoch"] = best_epoch
    run_config["paper_alignment"] = {
        "matches_paper": [
            "16 kHz resampling",
            "3-second non-overlapping segments",
            "20,000 total segments",
            "per-class split: 3500/1000/500",
            "64x94 Mel spectrogram input",
            "batch size 16",
            "initial learning rate 1e-2",
            "cross-entropy loss",
            "early stopping patience 10",
            "no data augmentation",
        ],
        "inferred_from_paper_text": [
            "segment-level random sampling is used to emulate the paper's 20,000-sample subset",
            "branch channel width is set to 88 for the fixed three-branch MA-CNN-A variant",
        ],
        "fixed_architecture": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "kernel_sizes": list(THREE_BRANCH_KERNEL_SIZES),
        },
    }
    (reports_dir / "deepship_macnna_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    return metrics
