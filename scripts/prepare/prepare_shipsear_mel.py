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

from src.data.shipsear import (
    CLASS_NAMES,
    ShipsearMelDataset,
    save_split_manifest,
    scan_shipsear,
    stratified_split,
    summarize_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute ShipsEar log-Mel features.")
    parser.add_argument("--data-root", default="ShipsEar")
    parser.add_argument("--output-root", default="outputs/precomputed/shipsear_mel")
    parser.add_argument("--sample-rate", type=int, default=4000)
    parser.add_argument("--clip-duration", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--win-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=2000.0)
    return parser


def build_bundle(records, dataset: ShipsearMelDataset, split_name: str) -> dict[str, object]:
    features = []
    labels = []
    metadata = []
    progress = tqdm(
        range(len(dataset)),
        desc=f"Precompute {split_name}",
        dynamic_ncols=True,
    )
    for idx in progress:
        mel_spec, label = dataset[idx]
        features.append(mel_spec)
        labels.append(label)
        record = records[idx]
        metadata.append(
            {
                "path": record.path,
                "subfolder": record.subfolder,
                "label_name": record.label_name,
                "label_index": record.label_index,
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

    print("Scanning ShipsEar and building stratified train/val/test split...")
    records = scan_shipsear(args.data_root)
    train_records, val_records, test_records = stratified_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_stats = {
        "full": summarize_records(records),
        "train": summarize_records(train_records),
        "val": summarize_records(val_records),
        "test": summarize_records(test_records),
    }
    save_split_manifest(output_root, train_records, val_records, test_records)
    (output_root / "shipsear_split_stats.json").write_text(
        json.dumps(split_stats, indent=2),
        encoding="utf-8",
    )

    dataset_kwargs = dict(
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        augment=False,
        cache_features=False,
    )
    bundles = {
        "train": build_bundle(
            train_records,
            ShipsearMelDataset(train_records, **dataset_kwargs),
            "train",
        ),
        "val": build_bundle(
            val_records,
            ShipsearMelDataset(val_records, **dataset_kwargs),
            "val",
        ),
        "test": build_bundle(
            test_records,
            ShipsearMelDataset(test_records, **dataset_kwargs),
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
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "n_mels": args.n_mels,
        "f_min": args.f_min,
        "f_max": args.f_max,
    }
    (output_root / "precompute_config.json").write_text(
        json.dumps(config_summary, indent=2),
        encoding="utf-8",
    )
    print("Precompute complete.")


if __name__ == "__main__":
    main()
