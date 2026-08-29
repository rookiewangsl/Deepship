from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ma_cnn_a import MACNNA_MODEL_VARIANTS  # noqa: E402
from src.pipelines.mel_ml.train_deepship_macnna_global import (  # noqa: E402
    GlobalAttentionTrainConfig,
    get_default_device,
    train,
)
from src.utils.pathing import default_deepship_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train validation-selected DeepShip MA-CNN-A G0/G0-C/G1 variants.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--experiment-config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument(
        "--g-series-config",
        default=str(ROOT / "configs" / "experiments" / "macnna_global_v1.json"),
    )
    parser.add_argument("--protocol-name", default="vessel_name_disjoint")
    parser.add_argument("--model-variant", choices=MACNNA_MODEL_VARIANTS, required=True)
    parser.add_argument("--attention-d-model", type=int, default=128)
    parser.add_argument("--attention-num-heads", type=int, default=4)
    parser.add_argument("--attention-ffn-expansion", type=int, default=2)
    parser.add_argument("--attention-position-kernel-size", type=int, default=9)
    parser.add_argument("--attention-temporal-kernel-size", type=int, default=15)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--attention-gate-init", type=float, default=-2.0)
    parser.add_argument(
        "--training-sampling",
        choices=("fixed_anchor", "vessel_balanced_dynamic"),
        default="fixed_anchor",
    )
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-experiment-overrides",
        action="store_true",
        help="Allow frozen hyperparameter overrides only for smoke/debug runs.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = GlobalAttentionTrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        split_manifest=args.split_manifest,
        experiment_config=args.experiment_config,
        g_series_config=args.g_series_config,
        protocol_name=args.protocol_name,
        model_variant=args.model_variant,
        attention_d_model=args.attention_d_model,
        attention_num_heads=args.attention_num_heads,
        attention_ffn_expansion=args.attention_ffn_expansion,
        attention_position_kernel_size=args.attention_position_kernel_size,
        attention_temporal_kernel_size=args.attention_temporal_kernel_size,
        attention_dropout=args.attention_dropout,
        attention_gate_init=args.attention_gate_init,
        training_sampling=args.training_sampling,
        train_samples_per_epoch=args.train_samples_per_epoch,
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
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        precision=args.precision,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        log_interval=args.log_interval,
        resume=args.resume,
        allow_experiment_overrides=args.allow_experiment_overrides,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        device=args.device or get_default_device(),
    )
    result = train(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
