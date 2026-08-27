from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.wav2vec2_conformer import DEFAULT_PRETRAINED_MODEL  # noqa: E402
from src.pipelines.waveform_conformer.train_deepship_conformer import (  # noqa: E402
    ConformerTrainConfig,
    get_default_device,
    train,
)
from src.utils.pathing import default_deepship_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the raw-waveform Wav2Vec2-Conformer DeepShip baseline.",
    )
    parser.add_argument("--data-root", default=default_deepship_root())
    parser.add_argument(
        "--output-root",
        required=True,
        help="Use an external-disk or server path; the command refuses a non-empty run directory.",
    )
    parser.add_argument(
        "--split-manifest",
        required=True,
        help="Frozen DeepShip split_manifest.json; no split is generated during training.",
    )
    parser.add_argument(
        "--isolation-experiment-config",
        default=str(ROOT / "configs" / "experiments" / "isolation_comparison_v1.json"),
    )
    parser.add_argument("--protocol-name", default=None)
    parser.add_argument("--pretrained-model", default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument(
        "--pretrained-revision",
        default=None,
        help=(
            "Optional Hugging Face commit/tag; the resolved commit is written "
            "to model_report.json."
        ),
    )
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--clip-duration", type=float, default=20.0)
    parser.add_argument(
        "--finetuning-mode",
        choices=["frozen", "feature_encoder", "last_n", "full"],
        default="last_n",
    )
    parser.add_argument("--train-last-n-layers", type=int, default=4)
    parser.add_argument(
        "--apply-spec-augment",
        action="store_true",
        help="Enable the backbone's latent-space SpecAugment (off for the clean baseline).",
    )
    parser.add_argument(
        "--layerdrop",
        type=float,
        default=0.0,
        help="Encoder LayerDrop probability (zero for the clean partial-finetuning baseline).",
    )
    checkpointing_group = parser.add_mutually_exclusive_group()
    checkpointing_group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable non-reentrant gradient checkpointing. Disabled by default for the "
            "verified 12 GB RTX 4070 baseline."
        ),
    )
    checkpointing_group.add_argument(
        "--disable-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(gradient_checkpointing=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Report cumulative training/validation progress every N batches.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ConformerTrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        split_manifest=args.split_manifest,
        isolation_experiment_config=args.isolation_experiment_config,
        protocol_name=args.protocol_name,
        pretrained_model_name_or_path=args.pretrained_model,
        pretrained_revision=args.pretrained_revision,
        load_pretrained=not args.random_init,
        clip_duration=args.clip_duration,
        finetuning_mode=args.finetuning_mode,
        train_last_n_layers=args.train_last_n_layers,
        apply_spec_augment=args.apply_spec_augment,
        layerdrop=args.layerdrop,
        gradient_checkpointing=args.gradient_checkpointing,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        precision=args.precision,
        seed=args.seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        resume=args.resume,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        device=args.device or get_default_device(),
    )
    result = train(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
