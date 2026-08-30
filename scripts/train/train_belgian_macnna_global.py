from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.mel_ml.train_belgian_macnna_global import (  # noqa: E402
    BelgianTrainConfig,
    get_default_device,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Belgian G0/G1 validation-only models.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--experiment-config",
        default=str(ROOT / "configs" / "experiments" / "belgian_attention_v1.json"),
    )
    parser.add_argument("--model-variant", choices=("g0", "g1"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--sampling-strategy",
        choices=(
            "class_date_balanced_dynamic",
            "full_epoch_shuffle",
            "strict_class_balanced_batch",
        ),
        default="class_date_balanced_dynamic",
    )
    parser.add_argument("--samples-per-class-per-epoch", type=int, default=None)
    parser.add_argument(
        "--loss-strategy",
        choices=("cross_entropy", "effective_number"),
        default="cross_entropy",
    )
    parser.add_argument("--effective-number-beta", type=float, default=0.999)
    parser.add_argument("--normalization-stats-path", default=None)
    parser.add_argument("--specaugment-frequency-mask-param", type=int, default=0)
    parser.add_argument("--specaugment-time-mask-param", type=int, default=0)
    parser.add_argument("--specaugment-frequency-masks", type=int, default=0)
    parser.add_argument("--specaugment-time-masks", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-experiment-overrides", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BelgianTrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        split_manifest=args.split_manifest,
        experiment_config=args.experiment_config,
        model_variant=args.model_variant,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sampling_strategy=args.sampling_strategy,
        samples_per_class_per_epoch=args.samples_per_class_per_epoch,
        loss_strategy=args.loss_strategy,
        effective_number_beta=args.effective_number_beta,
        normalization_stats_path=args.normalization_stats_path,
        specaugment_frequency_mask_param=args.specaugment_frequency_mask_param,
        specaugment_time_mask_param=args.specaugment_time_mask_param,
        specaugment_frequency_masks=args.specaugment_frequency_masks,
        specaugment_time_masks=args.specaugment_time_masks,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_start_epoch=args.early_stopping_start_epoch,
        precision=args.precision,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        log_interval=args.log_interval,
        device=args.device or get_default_device(),
        resume=args.resume,
        allow_experiment_overrides=args.allow_experiment_overrides,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print(json.dumps(train(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
