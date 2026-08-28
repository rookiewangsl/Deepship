from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipMelSegmentDataset,
    segment_record_from_dict,
)
from src.data.deepship_protocol_validation import load_split_manifest
from src.evaluation.embedding_diagnostics import diagnose_embeddings
from src.models.ma_cnn_a import MACNNAClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen MA-CNN-A pooled embeddings and diagnose whether they cluster "
            "by vessel name rather than ship class."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-metric-samples", type=int, default=3000)
    return parser


def _load_checkpoint(model: MACNNAClassifier, path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
            and isinstance(value, (str, int, float, bool, type(None)))
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    model.load_state_dict(state_dict, strict=True)
    return metadata


def _extract_split(
    model: MACNNAClassifier,
    dataset: DeepShipMelSegmentDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    captured: list[torch.Tensor] = []

    def capture_pool(_module: torch.nn.Module, _inputs: object, output: torch.Tensor) -> None:
        captured.append(output.detach().flatten(1).cpu())

    handle = model.pool.register_forward_hook(capture_pool)
    all_embeddings: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_indexes: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.inference_mode():
            for inputs, targets, indexes in loader:
                captured.clear()
                model(inputs.to(device, non_blocking=device.type == "cuda"))
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one pooled tensor, captured {len(captured)}")
                all_embeddings.append(captured[0])
                all_targets.append(targets.cpu())
                all_indexes.append(indexes.cpu())
    finally:
        handle.remove()
    return (
        torch.cat(all_embeddings).numpy(),
        torch.cat(all_targets).numpy(),
        torch.cat(all_indexes).numpy(),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_metric_samples <= 0:
        raise ValueError(
            "batch-size and max-metric-samples must be positive; "
            "num-workers must be non-negative"
        )
    allowed_splits = {"train", "val"}
    requested_splits = list(dict.fromkeys(args.splits))
    invalid_splits = set(requested_splits) - allowed_splits
    if invalid_splits:
        raise ValueError(
            "Only train/val are allowed for route-selection diagnostics; "
            f"got {sorted(invalid_splits)}"
        )

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(manifest_path)
    if str(manifest.get("protocol")) != "vessel_name_disjoint":
        raise ValueError("Embedding diagnosis requires the frozen vessel_name_disjoint manifest")

    split_segments = {split: [] for split in requested_splits}
    for row in manifest["segments"]:
        split = str(row["split"])
        if split in split_segments:
            split_segments[split].append(segment_record_from_dict(row))
    if any(not segments for segments in split_segments.values()):
        raise ValueError("One or more requested splits contain no segments")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = MACNNAClassifier(num_classes=len(CLASS_NAMES))
    checkpoint_metadata = _load_checkpoint(model, checkpoint_path)
    model.to(device)

    report: dict[str, object] = {
        "schema_version": 1,
        "diagnostic": "frozen_macnna_vessel_embedding",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest.get("manifest_sha256"),
        "protocol": manifest.get("protocol"),
        "device": str(device),
        "test_split_used": False,
        "splits": {},
    }
    for split in requested_splits:
        segments = split_segments[split]
        dataset = DeepShipMelSegmentDataset(
            segments,
            data_root=data_root,
            sample_rate=16000,
            clip_duration=3.0,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            n_mels=64,
            highpass_freq=None,
            return_index=True,
        )
        embeddings, targets, indexes = _extract_split(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        ordered_segments = [segments[int(index)] for index in indexes]
        class_names = [segment.class_name for segment in ordered_segments]
        vessel_keys = [segment.vessel_key for segment in ordered_segments]
        recording_paths = [segment.relative_path for segment in ordered_segments]
        expected_targets = np.asarray(
            [segment.label_index for segment in ordered_segments]
        )
        if not np.array_equal(targets, expected_targets):
            raise RuntimeError(f"Target/index mismatch while extracting {split}")

        np.savez_compressed(
            output_root / f"{split}_embeddings.npz",
            embeddings=embeddings,
            targets=targets,
            indexes=indexes,
            class_names=np.asarray(class_names),
            vessel_keys=np.asarray(vessel_keys),
            recording_paths=np.asarray(recording_paths),
        )
        split_report = diagnose_embeddings(
            embeddings,
            class_names=class_names,
            vessel_keys=vessel_keys,
            recording_paths=recording_paths,
            seed=args.seed,
            max_metric_samples=args.max_metric_samples,
        )
        report["splits"][split] = split_report
        (output_root / "diagnostic_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}: segments={split_report['segments']} "
            f"recordings={split_report['recordings']} vessels={split_report['vessels']}"
        )

    print(f"Saved diagnostic report to {output_root / 'diagnostic_report.json'}")


if __name__ == "__main__":
    main()
