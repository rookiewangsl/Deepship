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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain STFT-MAE on DeepShip precomputed STFT features.")
    parser.add_argument("--precomputed-root", default="outputs/precomputed/deepship_stft")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--scheduler-tmax", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--time-mask-param", type=int, default=30)
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--noise-std-min", type=float, default=0.02)
    parser.add_argument("--noise-std-max", type=float, default=0.10)
    parser.add_argument("--color-noise-std-min", type=float, default=0.02)
    parser.add_argument("--color-noise-std-max", type=float, default=0.08)
    parser.add_argument("--stripe-prob", type=float, default=0.3)
    parser.add_argument("--random-gain-db-min", type=float, default=-1.0)
    parser.add_argument("--random-gain-db-max", type=float, default=0.5)
    parser.add_argument("--patch-size-freq", type=int, default=32)
    parser.add_argument("--patch-size-time", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-embed-dim", type=int, default=64)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PretrainConfig(
        precomputed_root=args.precomputed_root,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_tmax=args.scheduler_tmax,
        num_workers=args.num_workers,
        mask_ratio=args.mask_ratio,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        color_noise_std_min=args.color_noise_std_min,
        color_noise_std_max=args.color_noise_std_max,
        stripe_prob=args.stripe_prob,
        random_gain_db_min=args.random_gain_db_min,
        random_gain_db_max=args.random_gain_db_max,
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
