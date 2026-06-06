from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepship import (
    CLASS_NAMES,
    DeepShipSTFTDataset,
    build_segment_records,
    save_split_manifest,
    scan_deepship,
    stratified_split,
    summarize_records,
    summarize_segments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute DeepShip log-STFT features.")
    parser.add_argument("--data-root", default="DeepShip")
    parser.add_argument("--output-root", default="outputs/precomputed/deepship_stft")
    parser.add_argument("--sample-rate", type=int, default=3000)
    parser.add_argument("--clip-duration", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--highpass-freq", type=float, default=100.0)
    parser.add_argument("--freq-min", type=float, default=100.0)
    parser.add_argument("--freq-max", type=float, default=1500.0)
    parser.add_argument("--img-h", type=int, default=128)
    parser.add_argument("--img-w", type=int, default=128)
    return parser


def build_bundle(records, dataset: DeepShipSTFTDataset, split_name: str) -> dict[str, object]:
    features = []
    labels = []
    metadata = []
    progress = tqdm(
        range(len(dataset)),
        desc=f"Precompute {split_name}",
        dynamic_ncols=True,
    )
    for idx in progress:
        spec, label = dataset[idx]
        features.append(spec.to(torch.float16))
        labels.append(label)
        record = records[idx]
        metadata.append(
            {
                "path": record.path,
                "class_name": record.class_name,
                "label_index": record.label_index,
                "start_frame": record.start_frame,
                "num_frames": record.num_frames,
                "sample_rate": record.sample_rate,
                "segment_index": record.segment_index,
                "total_segments": record.total_segments,
            }
        )
    return {
        "features": torch.stack(features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "class_names": CLASS_NAMES,
        "metadata": metadata,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Scanning DeepShip and building recording-level stratified train/val/test split...")
    records = scan_deepship(args.data_root)
    train_records, val_records, test_records = stratified_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_segments = build_segment_records(train_records, clip_duration=args.clip_duration)
    val_segments = build_segment_records(val_records, clip_duration=args.clip_duration)
    test_segments = build_segment_records(test_records, clip_duration=args.clip_duration)

    split_stats = {
        "full_recordings": summarize_records(records),
        "train_recordings": summarize_records(train_records),
        "val_recordings": summarize_records(val_records),
        "test_recordings": summarize_records(test_records),
        "train_segments": summarize_segments(train_segments),
        "val_segments": summarize_segments(val_segments),
        "test_segments": summarize_segments(test_segments),
    }
    save_split_manifest(output_root, train_records, val_records, test_records)
    (output_root / "deepship_stft_split_stats.json").write_text(
        json.dumps(split_stats, indent=2),
        encoding="utf-8",
    )

    dataset_kwargs = dict(
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        highpass_freq=args.highpass_freq,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        img_h=args.img_h,
        img_w=args.img_w,
        augment=False,
    )
    bundles = {
        "train": build_bundle(
            train_segments,
            DeepShipSTFTDataset(train_segments, **dataset_kwargs),
            "train",
        ),
        "val": build_bundle(
            val_segments,
            DeepShipSTFTDataset(val_segments, **dataset_kwargs),
            "val",
        ),
        "test": build_bundle(
            test_segments,
            DeepShipSTFTDataset(test_segments, **dataset_kwargs),
            "test",
        ),
    }

    for split_name, bundle in bundles.items():
        bundle_path = output_root / f"{split_name}.pt"
        torch.save(bundle, bundle_path)
        print(f"Saved {split_name} bundle to {bundle_path}")

    config_summary = {
        "sample_rate": args.sample_rate,
        "clip_duration": args.clip_duration,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "highpass_freq": args.highpass_freq,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "storage_dtype": "float16",
    }
    (output_root / "precompute_config.json").write_text(
        json.dumps(config_summary, indent=2),
        encoding="utf-8",
    )
    print("Precompute complete.")


if __name__ == "__main__":
    main()
