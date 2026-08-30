from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.belgian_ais import BelgianMelDataset, canonical_sha256  # noqa: E402
from src.pipelines.mel_ml.train_belgian_macnna_global import load_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute scalar log-Mel normalization from one Belgian training fold only."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser


def _configure_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Belgian normalization: {output_path}")
    manifest, train_records, _ = load_manifest(args.split_manifest, require_strict_audio=True)
    dataset = BelgianMelDataset(
        train_records,
        data_root=args.data_root,
        sample_rate=16_000,
        clip_duration=10.0,
        source_sample_rate=48_000,
        channel_policy="fixed_channel_0",
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=64,
    )
    worker_options: dict[str, object] = {}
    if args.num_workers > 0:
        worker_options = {
            "persistent_workers": True,
            "prefetch_factor": 2,
            "worker_init_fn": _configure_worker,
        }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **worker_options,
    )
    value_sum = 0.0
    square_sum = 0.0
    elements = 0
    for batch_index, (features, _targets) in enumerate(loader, start=1):
        values = features.to(dtype=torch.float64)
        value_sum += float(values.sum().item())
        square_sum += float(values.square().sum().item())
        elements += values.numel()
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"normalization batch={batch_index}/{len(loader)} "
                f"records={min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    if elements <= 1:
        raise RuntimeError("Belgian normalization saw too few log-Mel elements")
    mean = value_sum / elements
    variance = max(0.0, square_sum / elements - mean * mean)
    std = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise FloatingPointError(f"Invalid Belgian normalization: mean={mean}, std={std}")
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "belgian_training_sanity_v1",
        "split_manifest_sha256": manifest["manifest_sha256"],
        "split": "train",
        "records": len(train_records),
        "elements": elements,
        "sample_rate": 16_000,
        "clip_duration_seconds": 10.0,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 512,
        "n_mels": 64,
        "source_channel_policy": "fixed_channel_0",
        "mean": mean,
        "std": std,
        "test_evaluated": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
