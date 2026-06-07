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


def _parse_kernel_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise argparse.ArgumentTypeError("kernel sizes must contain at least two integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a MA-CNN-A style multi-scale asymmetric CNN on DeepShip.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-root", default="outputs/deepship_macnna")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=4000)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=50.0)
    parser.add_argument("--f-max", type=float, default=2000.0)
    parser.add_argument("--highpass-freq", type=float, default=50.0)
    parser.add_argument("--lowpass-freq", type=float, default=2000.0)
    parser.add_argument("--time-shift-frames", type=int, default=8)
    parser.add_argument("--time-mask-param", type=int, default=16)
    parser.add_argument("--freq-mask-param", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        default=True,
        help="Enable waveform and spectrogram augmentation (default: True).",
    )
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false")
    parser.add_argument(
        "--use-random-crop",
        action="store_true",
        default=True,
        help="Use random crop sampling for training (default: True).",
    )
    parser.add_argument("--no-random-crop", dest="use_random_crop", action="store_false")
    parser.add_argument("--max-segments-per-recording", type=int, default=12)
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        default=True,
        help="Use WeightedRandomSampler for class balance (default: True).",
    )
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Use class-weighted cross entropy (default: True).",
    )
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--kernel-sizes", type=_parse_kernel_sizes, default=(8, 16, 32))
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--branch-channels", type=int, default=24)
    parser.add_argument("--fused-channels", type=int, default=96)
    parser.add_argument("--classifier-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        seed=args.seed,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        highpass_freq=args.highpass_freq,
        lowpass_freq=args.lowpass_freq,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        use_augmentation=args.use_augmentation,
        use_weighted_sampler=args.use_weighted_sampler,
        use_class_weights=args.use_class_weights,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        time_shift_frames=args.time_shift_frames,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        use_random_crop=args.use_random_crop,
        max_segments_per_recording=args.max_segments_per_recording,
        kernel_sizes=args.kernel_sizes,
        stem_channels=args.stem_channels,
        branch_channels=args.branch_channels,
        fused_channels=args.fused_channels,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
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
