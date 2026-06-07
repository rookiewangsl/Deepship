from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.waveform_transformer.pretrain_deepship_mae import (
    PretrainConfig,
    pretrain,
)
from src.pipelines.mel_ml.train_shipsear_cnn import get_default_device
from src.utils.pathing import default_deepship_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain STFT-MAE on DeepShip wav-derived STFT features.")
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--sample-rate", type=int, default=3000)
    parser.add_argument("--clip-duration", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--highpass-freq", type=float, default=50.0)
    parser.add_argument("--freq-min", type=float, default=50.0)
    parser.add_argument("--freq-max", type=float, default=1500.0)
    parser.add_argument("--img-h", type=int, default=128)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--time-mask-param", type=int, default=12)
    parser.add_argument("--freq-mask-param", type=int, default=12)
    parser.add_argument("--noise-std-min", type=float, default=0.02)
    parser.add_argument("--noise-std-max", type=float, default=0.10)
    parser.add_argument("--color-noise-std-min", type=float, default=0.02)
    parser.add_argument("--color-noise-std-max", type=float, default=0.08)
    parser.add_argument("--stripe-prob", type=float, default=0.3)
    parser.add_argument("--patch-size-freq", type=int, default=8)
    parser.add_argument("--patch-size-time", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--decoder-embed-dim", type=int, default=64)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-heads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PretrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        mask_ratio=args.mask_ratio,
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
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        color_noise_std_min=args.color_noise_std_min,
        color_noise_std_max=args.color_noise_std_max,
        stripe_prob=args.stripe_prob,
        patch_size_freq=args.patch_size_freq,
        patch_size_time=args.patch_size_time,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        device=args.device or get_default_device(),
    )
    pretrain(config)


if __name__ == "__main__":
    main()
