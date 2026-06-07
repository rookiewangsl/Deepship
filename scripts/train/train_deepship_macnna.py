from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.mel_ml.train_deepship_macnna import (  # noqa: E402
    TrainConfig,
    get_default_device,
    train,
)
from src.utils.pathing import default_deepship_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the stable three-branch MA-CNN-A DeepShip model.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-root", default="outputs/deepship_macnna_paper")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--train-per-class", type=int, default=3500)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--highpass-freq", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--branch-channels", type=int, default=88)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        clip_duration=args.clip_duration,
        samples_per_class=args.samples_per_class,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        highpass_freq=args.highpass_freq,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        branch_channels=args.branch_channels,
        device=args.device or get_default_device(),
    )
    metrics = train(config)
    print(
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
