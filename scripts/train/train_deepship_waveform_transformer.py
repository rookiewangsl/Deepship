from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.waveform_transformer.train_deepship_transformer import (
    TrainConfig,
    get_default_device,
    train,
)
from src.utils.pathing import default_deepship_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train STFT Transformer baseline on DeepShip.")
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precomputed-root", default="outputs/precomputed/deepship_stft")
    parser.add_argument(
        "--mae-pretrained-path",
        default="outputs/deepship_stft_mae_pretrain/models/deepship_stft_mae_best.pt",
    )
    parser.add_argument(
        "--use-weighted-sampler", action="store_true", default=True,
        help="Use WeightedRandomSampler for the training split (default: True).",
    )
    parser.add_argument(
        "--no-weighted-sampler", dest="use_weighted_sampler", action="store_false"
    )
    parser.add_argument(
        "--use-class-weights", action="store_true", default=True,
        help="Use class-weighted CrossEntropyLoss (default: True).",
    )
    parser.add_argument(
        "--no-class-weights", dest="use_class_weights", action="store_false"
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument(
        "--use-random-crop", action="store_true", default=True,
        help="Use random cropping from raw recordings for training (default: True).",
    )
    parser.add_argument("--no-random-crop", dest="use_random_crop", action="store_false")
    parser.add_argument("--max-segments-per-recording", type=int, default=12)
    parser.add_argument("--random-time-shift", type=int, default=400)
    parser.add_argument("--gain-min", type=float, default=0.85)
    parser.add_argument("--gain-max", type=float, default=1.15)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--highpass-freq", type=float, default=50.0)
    parser.add_argument("--freq-min", type=float, default=50.0)
    parser.add_argument("--freq-max", type=float, default=1500.0)
    parser.add_argument("--img-h", type=int, default=128)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--time-mask-param", type=int, default=30)
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--patch-size-freq", type=int, default=8)
    parser.add_argument("--patch-size-time", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr=args.min_lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device or get_default_device(),
        precomputed_root=args.precomputed_root,
        mae_pretrained_path=args.mae_pretrained_path,
        use_weighted_sampler=args.use_weighted_sampler,
        use_class_weights=args.use_class_weights,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_augmentation=args.use_augmentation,
        use_random_crop=args.use_random_crop,
        max_segments_per_recording=args.max_segments_per_recording,
        random_time_shift=args.random_time_shift,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        noise_std=args.noise_std,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        highpass_freq=args.highpass_freq,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        img_h=args.img_h,
        img_w=args.img_w,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        patch_size_freq=args.patch_size_freq,
        patch_size_time=args.patch_size_time,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
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
