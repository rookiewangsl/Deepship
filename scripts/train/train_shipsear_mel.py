from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.mel_ml.train_shipsear_cnn import TrainConfig, get_default_device, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mel+CNN baseline on ShipsEar.")
    parser.add_argument("--data-root", default="ShipsEar")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precomputed-root", default="outputs/precomputed/shipsear_mel")
    parser.add_argument("--device", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--time-shift-frames", type=int, default=8)
    parser.add_argument("--time-mask-param", type=int, default=12)
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--use-augmentation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        precomputed_root=args.precomputed_root,
        device=args.device or get_default_device(),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        time_shift_frames=args.time_shift_frames,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        use_augmentation=args.use_augmentation,
    )
    metrics = train(config)
    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
