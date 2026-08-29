from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ma_cnn_a import MACNNA_MODEL_VARIANTS, build_macnna_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit G0/G0-C/G1 size and inference cost.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=94)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser


def synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def audit_variant(
    variant: str,
    inputs: torch.Tensor,
    *,
    warmup_runs: int,
    timed_runs: int,
    device: str,
) -> dict[str, object]:
    model = build_macnna_model(4, model_variant=variant).to(device).eval()
    base = build_macnna_model(4, model_variant="g0")
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        output = model(inputs)
    flops = int(counter.get_total_flops())
    for _ in range(warmup_runs):
        with torch.no_grad():
            model(inputs)
    synchronize(device)
    timings = []
    for _ in range(timed_runs):
        started_at = time.perf_counter()
        with torch.no_grad():
            model(inputs)
        synchronize(device)
        timings.append((time.perf_counter() - started_at) * 1000.0)
    peak_memory = None
    if str(device).startswith("cuda"):
        peak_memory = int(torch.cuda.max_memory_allocated(torch.device(device)))
    return {
        "model_variant": variant,
        "num_parameters": model.num_parameters,
        "added_parameters_over_g0": model.num_parameters - base.num_parameters,
        "forward_flops": flops,
        "approximate_macs": flops / 2.0,
        "output_shape": list(output.shape),
        "latency_ms_median": statistics.median(timings),
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "peak_memory_bytes": peak_memory,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.n_mels <= 0 or args.time_frames <= 0:
        raise ValueError("Input dimensions must be positive")
    if args.warmup_runs < 0 or args.timed_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be positive")
    inputs = torch.zeros(
        args.batch_size,
        1,
        args.n_mels,
        args.time_frames,
        device=args.device,
    )
    rows = [
        audit_variant(
            variant,
            inputs,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            device=args.device,
        )
        for variant in MACNNA_MODEL_VARIANTS
    ]
    by_variant = {str(row["model_variant"]): row for row in rows}
    control = by_variant["g0_c"]
    attention = by_variant["g1"]
    report = {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "device": args.device,
        "input_shape": list(inputs.shape),
        "flop_counter": "torch.utils.flop_counter; approximate_macs=forward_flops/2",
        "variants": rows,
        "matching": {
            "added_parameter_relative_difference": abs(
                float(control["added_parameters_over_g0"])
                - float(attention["added_parameters_over_g0"])
            )
            / float(attention["added_parameters_over_g0"]),
            "flop_relative_difference": abs(
                float(control["forward_flops"]) - float(attention["forward_flops"])
            )
            / float(attention["forward_flops"]),
            "parameter_target_met": abs(
                float(control["added_parameters_over_g0"])
                - float(attention["added_parameters_over_g0"])
            )
            / float(attention["added_parameters_over_g0"])
            <= 0.10,
            "flop_target_met": abs(
                float(control["forward_flops"]) - float(attention["forward_flops"])
            )
            / float(attention["forward_flops"])
            <= 0.15,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
