from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.belgian_ais import BelgianMelDataset, BelgianRecord, canonical_sha256  # noqa: E402
from src.data.deepship import CLASS_NAMES  # noqa: E402
from src.evaluation.classification import compute_metrics  # noqa: E402
from src.models.ma_cnn_a import build_macnna_model  # noqa: E402
from src.pipelines.mel_ml.train_belgian_macnna_global import load_manifest  # noqa: E402
from src.pipelines.mel_ml.train_deepship_macnna import (  # noqa: E402
    atomic_torch_save,
    get_default_device,
    runtime_environment,
    set_seed,
)
from src.pipelines.mel_ml.train_deepship_macnna_global import (  # noqa: E402
    validate_trainable_gradients,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that G0 can memorize a small balanced Belgian training subset."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser


def select_balanced_records(
    records: list[BelgianRecord],
    *,
    samples_per_class: int,
    seed: int,
) -> list[BelgianRecord]:
    selected: list[BelgianRecord] = []
    for class_name in CLASS_NAMES:
        candidates = [record for record in records if record.class_name == class_name]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Overfit subset needs {samples_per_class} {class_name} records, "
                f"found {len(candidates)}"
            )
        ranked = sorted(
            candidates,
            key=lambda record: hashlib.sha256(
                f"{seed}:{record.relative_path}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:samples_per_class])
    return selected


def _evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    with torch.no_grad():
        for features, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.numel()
            targets_all.extend(int(value) for value in targets.cpu().tolist())
            predictions_all.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
    metrics = compute_metrics(targets_all, predictions_all, CLASS_NAMES)
    return {
        "loss": total_loss / len(dataset),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_f1": {
            name: float(metrics["classification_report"][name]["f1-score"])
            for name in CLASS_NAMES
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse Belgian overfit output: {output_root}")
    experiment = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    if experiment.get("experiment_id") != "belgian_training_sanity_v1":
        raise ValueError("Overfit check requires belgian_training_sanity_v1")
    gate = experiment["overfit_gate"]
    manifest, train_records, _ = load_manifest(args.split_manifest, require_strict_audio=True)
    normalization = json.loads(Path(args.normalization_stats).read_text(encoding="utf-8"))
    if normalization.get("split_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Overfit normalization uses another Belgian manifest")
    if normalization.get("split") != "train" or normalization.get("test_evaluated") is not False:
        raise ValueError("Overfit normalization must be train-only and test-free")
    selected = select_balanced_records(
        train_records,
        samples_per_class=int(gate["samples_per_class"]),
        seed=args.seed,
    )
    dataset = BelgianMelDataset(
        selected,
        data_root=args.data_root,
        sample_rate=16_000,
        clip_duration=10.0,
        source_sample_rate=48_000,
        channel_policy="fixed_channel_0",
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=64,
        normalization_mean=float(normalization["mean"]),
        normalization_std=float(normalization["std"]),
    )
    print(f"materializing {len(dataset)} train-only log-Mel examples", flush=True)
    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for features, targets in DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    ):
        feature_parts.append(features)
        target_parts.append(targets)
    cached = TensorDataset(torch.cat(feature_parts), torch.cat(target_parts))
    selected_rows = [
        {
            "relative_path": record.relative_path,
            "class_name": record.class_name,
            "calendar_date": record.calendar_date,
            "station": record.station,
        }
        for record in selected
    ]
    device = args.device or get_default_device()
    set_seed(args.seed)
    model = build_macnna_model(4, model_variant="g0").to(device)
    if model.num_parameters != int(experiment["variants"]["g0"]["expected_num_parameters"]):
        raise ValueError("G0 parameter count changed before the overfit check")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    maximum_steps = int(gate["maximum_optimizer_steps"])
    history: list[dict[str, object]] = []
    consecutive_passes = 0
    gradients_validated = False
    step = 0
    started = time.perf_counter()
    while step < maximum_steps and consecutive_passes < 2:
        model.train()
        order = torch.randperm(len(cached), generator=generator)
        for start in range(0, len(cached), args.batch_size):
            indexes = order[start : start + args.batch_size]
            features = cached.tensors[0][indexes].to(device)
            targets = cached.tensors[1][indexes].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite overfit loss at step {step + 1}")
            loss.backward()
            if not gradients_validated:
                validate_trainable_gradients(model)
                gradients_validated = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step % args.eval_interval == 0 or step == maximum_steps:
                metrics = _evaluate(
                    model,
                    cached,
                    batch_size=args.batch_size,
                    device=device,
                )
                passed = (
                    metrics["accuracy"] >= float(gate["minimum_train_accuracy"])
                    and metrics["macro_f1"] >= float(gate["minimum_train_macro_f1"])
                    and metrics["loss"] <= float(gate["maximum_train_loss"])
                )
                consecutive_passes = consecutive_passes + 1 if passed else 0
                history.append({"step": step, **metrics, "passed": passed})
                print(
                    f"step={step}/{maximum_steps} loss={metrics['loss']:.4f} "
                    f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                    f"gate_passes={consecutive_passes}/2",
                    flush=True,
                )
            if step >= maximum_steps or consecutive_passes >= 2:
                break
    final_metrics = _evaluate(model, cached, batch_size=args.batch_size, device=device)
    passed = consecutive_passes >= 2
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "belgian_training_sanity_v1",
        "check": "balanced_small_subset_overfit",
        "status": "passed" if passed else "failed",
        "test_evaluated": False,
        "model_variant": "g0",
        "model_seed": args.seed,
        "num_parameters": model.num_parameters,
        "split_manifest_sha256": manifest["manifest_sha256"],
        "normalization_report_sha256": normalization.get("report_sha256"),
        "selected_records": selected_rows,
        "selected_records_sha256": canonical_sha256({"records": selected_rows}),
        "selected_by_class": dict(Counter(record.class_name for record in selected)),
        "selected_dates_by_class": {
            name: len({record.calendar_date for record in selected if record.class_name == name})
            for name in CLASS_NAMES
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "batch_size": args.batch_size,
            "steps": step,
        },
        "gate": gate,
        "consecutive_passing_evaluations": consecutive_passes,
        "final_metrics": final_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
    }
    if any(
        not math.isfinite(float(report["final_metrics"][name]))
        for name in ("loss", "accuracy", "macro_f1")
    ):
        raise FloatingPointError("Overfit report contains a non-finite metric")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "overfit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "report": report,
            "test_evaluated": False,
        },
        output_root / "overfit_last.pt",
    )
    if not passed:
        raise RuntimeError("Belgian G0 failed the pre-registered small-subset overfit gate")


if __name__ == "__main__":
    main()
