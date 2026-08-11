from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import platform
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipMelSegmentDataset,
    build_paper_split,
    save_segment_split_manifest,
    scan_deepship,
    segment_record_from_dict,
)
from src.data.deepship_audit import load_experiment_config
from src.data.deepship_protocol_validation import (
    load_split_manifest,
    validate_protocol_manifest,
)
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
from src.models.ma_cnn_a import MACNNAClassifier, THREE_BRANCH_KERNEL_SIZES
from src.pipelines.mel_ml.isolation_experiment import enforce_training_config
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


def runtime_environment() -> dict[str, object]:
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolve_path("."),
        text=True,
        capture_output=True,
        check=False,
    )
    git_status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=resolve_path("."),
        text=True,
        capture_output=True,
        check=False,
    )
    git_status = (
        git_status_result.stdout.splitlines()
        if git_status_result.returncode == 0
        else ["git status unavailable"]
    )
    cuda_devices = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_devices": cuda_devices,
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else "unknown",
        "git_worktree_dirty": bool(git_status),
        "git_status": git_status,
    }


def atomic_torch_save(payload: dict[str, object], path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


@dataclass
class TrainConfig:
    data_root: str = "DeepShip"
    output_root: str = "outputs/deepship_macnna_paper"
    cache_root: str | None = None
    split_manifest: str | None = None
    experiment_config: str | None = None
    protocol_name: str | None = None
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
    momentum: float = 0.9
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 10
    early_stopping_patience: int = 10
    num_workers: int = 0
    resume: bool = False
    allow_experiment_overrides: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    device: str = get_default_device()


def build_dataloaders(config: TrainConfig) -> tuple[dict[str, DataLoader], dict[str, object]]:
    if config.split_manifest is not None:
        if config.experiment_config is None:
            raise ValueError("experiment_config is required when split_manifest is used")
        manifest_path = resolve_path(config.split_manifest)
        experiment = load_experiment_config(config.experiment_config)
        enforce_training_config(
            asdict(config),
            experiment,
            allow_overrides=config.allow_experiment_overrides,
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
            raise ValueError("Frozen split manifest validation failed")
        manifest_protocol = str(manifest["protocol"])
        if config.protocol_name is not None and config.protocol_name != manifest_protocol:
            raise ValueError(
                f"protocol_name={config.protocol_name!r} does not match manifest "
                f"protocol={manifest_protocol!r}"
            )
        split_segments = {split: [] for split in ("train", "val", "test")}
        for row in manifest["segments"]:
            split_segments[str(row["split"])].append(segment_record_from_dict(row))
        split_stats = {
            "source": "frozen_manifest",
            "protocol": manifest_protocol,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(manifest_path),
            "validation": validation,
        }
    else:
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
            return_index=(split == "test"),
            **dataset_kwargs,
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
    reports_dir = resolve_path(config.output_root) / "reports"
    if config.split_manifest is None:
        save_segment_split_manifest(reports_dir, split_segments, split_stats)
    else:
        reports_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve_path(config.split_manifest), reports_dir / "frozen_split_manifest.json")
        (reports_dir / "split_validation.json").write_text(
            json.dumps(split_stats["validation"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dataloaders, split_stats


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
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
        for batch_index, (inputs, targets) in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
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


def collect_prediction_rows(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    dataset = dataloader.dataset
    if not isinstance(dataset, DeepShipMelSegmentDataset) or not dataset.return_index:
        raise ValueError("Prediction DataLoader must expose DeepShip segment indexes")
    use_amp = str(device).startswith("cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.no_grad():
        for batch_index, (inputs, targets, indexes) in enumerate(tqdm(
            dataloader,
            desc="test",
            leave=False,
            dynamic_ncols=True,
        )):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=use_amp)
            with amp_ctx:
                logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            for target, prediction, probability, index in zip(
                targets.tolist(),
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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
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

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cosine_epochs),
        eta_min=config.min_learning_rate,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def train(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    experiment = (
        load_experiment_config(config.experiment_config)
        if config.experiment_config is not None
        else None
    )
    config_mismatches = (
        enforce_training_config(
            asdict(config),
            experiment,
            allow_overrides=config.allow_experiment_overrides,
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
    cache_root = resolve_path(config.cache_root) if config.cache_root is not None else None
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    for directory in [metrics_dir, figures_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    environment = runtime_environment()
    if (
        config.split_manifest is not None
        and not config.allow_experiment_overrides
        and environment["git_worktree_dirty"]
    ):
        details = "\n".join(str(item) for item in environment["git_status"])
        raise RuntimeError(
            "Formal isolation training requires a clean git worktree. Commit or remove these "
            f"changes before training:\n{details}"
        )
    (reports_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataloaders, split_stats = build_dataloaders(config)
    model = MACNNAClassifier(num_classes=len(CLASS_NAMES)).to(config.device)
    if experiment is not None:
        expected_parameters = int(experiment["model"]["expected_num_parameters"])
        if model.num_parameters != expected_parameters:
            raise ValueError(
                f"Model parameter count changed: expected {expected_parameters}, "
                f"got {model.num_parameters}"
            )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
    )
    scheduler = build_scheduler(optimizer, config)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "learning_rate": []}
    history_path = reports_dir / "deepship_macnna_history.json"
    best_model_path = models_dir / "deepship_macnna_best.pt"
    last_model_path = models_dir / "deepship_macnna_last.pt"
    best_val_acc = -1.0
    best_epoch = -1
    start_epoch = 1

    if config.resume:
        if not last_model_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is unavailable: {last_model_path}")
        resume_checkpoint = torch.load(
            last_model_path,
            map_location=config.device,
            weights_only=False,
        )
        current_manifest_hash = split_stats.get("manifest_sha256")
        if resume_checkpoint.get("split_manifest_sha256") != current_manifest_hash:
            raise ValueError("Resume checkpoint uses a different split manifest")
        if resume_checkpoint["config"]["seed"] != config.seed:
            raise ValueError("Resume checkpoint uses a different model seed")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None and resume_checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        history = resume_checkpoint["history"]
        best_val_acc = float(resume_checkpoint["best_val_acc"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        restore_rng_state(resume_checkpoint["rng_state"])
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            config.device,
            desc="train",
            optimizer=optimizer,
            max_batches=config.max_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            config.device,
            desc="val",
            max_batches=config.max_eval_batches,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)
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
                    "split_stats": split_stats,
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
                "config": asdict(config),
                "epoch": epoch,
                "history": history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "split_manifest_sha256": split_stats.get("manifest_sha256"),
                "rng_state": capture_rng_state(),
            },
            last_model_path,
        )
        print(
            f"lr={current_lr:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )
        if should_stop:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    if not best_model_path.is_file():
        raise RuntimeError("No best checkpoint was created")
    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    segment_predictions = collect_prediction_rows(
        model,
        dataloaders["test"],
        config.device,
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
    save_metrics(segment_metrics, metrics_dir / "deepship_macnna_metrics.json")
    save_metrics(recording_metrics, metrics_dir / "recording_metrics.json")
    if vessel_metrics is not None:
        save_metrics(vessel_metrics, metrics_dir / "vessel_metrics.json")
    predictions_dir = output_root / "predictions"
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
    protocol_title = split_stats.get("protocol", "paper_segment_level")
    plot_confusion_matrix(
        segment_metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "segment_confusion_matrix.png",
        title=f"DeepShip {protocol_title} Segment Confusion Matrix",
    )
    plot_confusion_matrix(
        recording_metrics["confusion_matrix"],
        CLASS_NAMES,
        figures_dir / "recording_confusion_matrix.png",
        title=f"DeepShip {protocol_title} Recording Confusion Matrix",
    )
    if vessel_metrics is not None:
        plot_confusion_matrix(
            vessel_metrics["confusion_matrix"],
            CLASS_NAMES,
            figures_dir / "vessel_confusion_matrix.png",
            title=f"DeepShip {protocol_title} Vessel-group Confusion Matrix",
        )
    plot_training_curves(
        history,
        figures_dir / "deepship_macnna_training_curves.png",
        title=f"DeepShip {protocol_title} Training Curves",
    )

    run_config = asdict(config)
    run_config["num_parameters"] = model.num_parameters
    run_config["best_val_acc"] = best_val_acc
    run_config["best_epoch"] = best_epoch
    run_config["protocol"] = protocol_title
    run_config["split_manifest_sha256"] = split_stats.get("manifest_sha256")
    run_config["experiment_config_mismatches"] = config_mismatches
    run_config["fixed_architecture"] = {
        "sample_rate": TARGET_SAMPLE_RATE,
        "kernel_sizes": list(THREE_BRANCH_KERNEL_SIZES),
    }
    (reports_dir / "deepship_macnna_run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": protocol_title,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "split_manifest_sha256": split_stats.get("manifest_sha256"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return segment_metrics
