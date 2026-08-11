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
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Frozen split_manifest.json. Required for the three isolation comparison runs.",
    )
    parser.add_argument(
        "--experiment-config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument("--protocol-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-experiment-overrides",
        action="store_true",
        help="Allow frozen hyperparameter overrides for smoke/debug runs only.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        cache_root=args.cache_root,
        split_manifest=args.split_manifest,
        experiment_config=args.experiment_config if args.split_manifest is not None else None,
        protocol_name=args.protocol_name,
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
        momentum=args.momentum,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        resume=args.resume,
        allow_experiment_overrides=args.allow_experiment_overrides,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
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
