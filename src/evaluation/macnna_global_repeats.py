"""Summarize the fully crossed DeepShip L20 global-attention repeat matrix."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Mapping


LEVELS = ("segment", "recording", "vessel")
VARIANTS = ("g0", "g0_c", "g1")


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required repeat result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _run_root(
    existing_root: Path,
    repeat_root: Path,
    *,
    variant: str,
    split_seed: int,
    model_seed: int,
) -> Path:
    if split_seed == 42 and model_seed == 42:
        return existing_root / f"formal_{variant}_l20_seed42"
    return (
        repeat_root
        / f"split{split_seed}"
        / f"formal_{variant}_l20_split{split_seed}_seed{model_seed}"
    )


def _read_run(
    run_root: Path,
    *,
    variant: str,
    split_seed: int,
    model_seed: int,
    expected_parameters: int,
    expected_existing_commit: str,
) -> dict[str, object]:
    complete = _load_json(run_root / "reports" / "run_complete.json")
    train_config = _load_json(run_root / "reports" / "train_config.json")
    environment = _load_json(run_root / "reports" / "environment.json")
    if complete.get("status") != "validation_complete" or complete.get("test_evaluated") is not False:
        raise ValueError(f"Repeat run is not validation-only complete: {run_root}")
    expected_config = {
        "model_variant": variant,
        "seed": model_seed,
        "clip_duration": 20.0,
        "training_sampling": "vessel_balanced_dynamic",
        "train_samples_per_epoch": 14000,
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
        "batch_size": 4,
        "eval_batch_size": 4,
        "epochs": 50,
        "precision": "bf16",
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {train_config.get(field)!r}"
        for field, expected in expected_config.items()
        if train_config.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"Repeat run configuration mismatch in {run_root}: " + "; ".join(mismatches))
    if environment.get("git_worktree_dirty") is not False:
        raise ValueError(f"Repeat run used a dirty Git worktree: {run_root}")
    git_commit = environment.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit:
        raise ValueError(f"Repeat run has no Git commit: {run_root}")
    if split_seed == 42 and model_seed == 42 and not git_commit.startswith(expected_existing_commit):
        raise ValueError(f"Reused seed42 run has unexpected commit {git_commit!r}: {run_root}")
    model_report = _load_json(run_root / "reports" / "model_report.json")
    if model_report.get("num_parameters") != expected_parameters:
        raise ValueError(f"Repeat model parameter count mismatch: {run_root}")

    row: dict[str, object] = {
        "split_seed": split_seed,
        "model_seed": model_seed,
        "variant": variant,
        "run_root": str(run_root),
        "git_commit": git_commit,
        "best_epoch": complete.get("best_epoch"),
        "num_parameters": expected_parameters,
    }
    for level in LEVELS:
        metrics = _load_json(run_root / "metrics" / f"validation_best_{level}_metrics.json")
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {level} {metric} in {run_root}")
            row[f"{level}_{metric}"] = float(value)
    return row


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hierarchical_bootstrap(
    values_by_split: Mapping[int, Mapping[int, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Resample vessel partitions, then optimization seeds within each partition."""

    split_seeds = sorted(values_by_split)
    if len(split_seeds) < 2 or resamples <= 0:
        raise ValueError("Hierarchical bootstrap needs multiple splits and positive resamples")
    model_seeds = sorted(next(iter(values_by_split.values())))
    if any(sorted(values_by_split[split]) != model_seeds for split in split_seeds):
        raise ValueError("Every split must contain the same model seeds")
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        selected_splits = [rng.choice(split_seeds) for _ in split_seeds]
        sampled_values = []
        for split_seed in selected_splits:
            for _ in model_seeds:
                model_seed = rng.choice(model_seeds)
                sampled_values.append(float(values_by_split[split_seed][model_seed]))
        samples.append(mean(sampled_values))
    return {
        "method": "resample split seeds, then model seeds within each selected split",
        "resamples": resamples,
        "seed": seed,
        "point": mean(
            float(values_by_split[split][model])
            for split in split_seeds
            for model in model_seeds
        ),
        "bootstrap_mean": mean(samples),
        "bootstrap_std": stdev(samples),
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
        "probability_greater_than_zero": sum(value > 0.0 for value in samples) / len(samples),
    }


def summarize_global_repeats(
    existing_root: str | Path,
    repeat_root: str | Path,
    experiment_config: str | Path,
) -> dict[str, object]:
    config = _load_json(Path(experiment_config).expanduser().resolve())
    training = config["training"]
    base_protocol = config["base_protocol"]
    variants_config = config["variants"]
    repeat_matrix = config["repeat_matrix"]
    decision_rule = config["decision_rule"]
    assert isinstance(training, Mapping)
    assert isinstance(base_protocol, Mapping)
    assert isinstance(variants_config, Mapping)
    assert isinstance(repeat_matrix, Mapping)
    assert isinstance(decision_rule, Mapping)
    split_seeds = [int(value) for value in base_protocol["split_seeds"]]
    model_seeds = [int(value) for value in training["model_seeds"]]
    reused = repeat_matrix["reuse_existing"]
    assert isinstance(reused, Mapping)

    existing_path = Path(existing_root).expanduser().resolve()
    repeat_path = Path(repeat_root).expanduser().resolve()
    rows = []
    for split_seed in split_seeds:
        for model_seed in model_seeds:
            for variant in VARIANTS:
                variant_config = variants_config[variant]
                assert isinstance(variant_config, Mapping)
                root = _run_root(
                    existing_path,
                    repeat_path,
                    variant=variant,
                    split_seed=split_seed,
                    model_seed=model_seed,
                )
                rows.append(
                    _read_run(
                        root,
                        variant=variant,
                        split_seed=split_seed,
                        model_seed=model_seed,
                        expected_parameters=int(variant_config["expected_num_parameters"]),
                        expected_existing_commit=str(reused["git_commit"]),
                    )
                )

    summaries = []
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        for level in LEVELS:
            values = [float(row[f"{level}_macro_f1"]) for row in variant_rows]
            summaries.append(
                {
                    "variant": variant,
                    "level": level,
                    "n": len(values),
                    "mean_macro_f1": mean(values),
                    "std_macro_f1": stdev(values),
                    "min_macro_f1": min(values),
                    "max_macro_f1": max(values),
                }
            )

    lookup = {
        (int(row["split_seed"]), int(row["model_seed"]), str(row["variant"])): row
        for row in rows
    }
    paired_rows = []
    for split_seed in split_seeds:
        for model_seed in model_seeds:
            for reference in ("g0", "g0_c"):
                g1 = lookup[(split_seed, model_seed, "g1")]
                base = lookup[(split_seed, model_seed, reference)]
                paired_rows.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "comparison": "g1",
                        "reference": reference,
                        "vessel_macro_f1_delta": float(g1["vessel_macro_f1"])
                        - float(base["vessel_macro_f1"]),
                        "recording_macro_f1_delta": float(g1["recording_macro_f1"])
                        - float(base["recording_macro_f1"]),
                    }
                )

    comparisons = {}
    for reference in ("g0", "g0_c"):
        comparison_rows = [row for row in paired_rows if row["reference"] == reference]
        item: dict[str, object] = {}
        for level in ("vessel", "recording"):
            values_by_split = {
                split_seed: {
                    model_seed: float(
                        next(
                            row[f"{level}_macro_f1_delta"]
                            for row in comparison_rows
                            if row["split_seed"] == split_seed
                            and row["model_seed"] == model_seed
                        )
                    )
                    for model_seed in model_seeds
                }
                for split_seed in split_seeds
            }
            split_means = {
                str(split): mean(values_by_split[split].values()) for split in split_seeds
            }
            item[level] = {
                "mean_delta": mean(
                    value for split_values in values_by_split.values() for value in split_values.values()
                ),
                "positive_cells": sum(
                    value > 0.0
                    for split_values in values_by_split.values()
                    for value in split_values.values()
                ),
                "positive_split_means": sum(value > 0.0 for value in split_means.values()),
                "split_mean_deltas": split_means,
                "hierarchical_bootstrap": hierarchical_bootstrap(
                    values_by_split,
                    resamples=int(decision_rule["bootstrap_resamples"]),
                    seed=int(decision_rule["bootstrap_seed"]),
                ),
            }
        comparisons[f"g1_minus_{reference}"] = item

    g1_g0_vessel = comparisons["g1_minus_g0"]["vessel"]  # type: ignore[index]
    g1_g0c_vessel = comparisons["g1_minus_g0_c"]["vessel"]  # type: ignore[index]
    g1_g0_recording = comparisons["g1_minus_g0"]["recording"]  # type: ignore[index]
    gates = {
        "mean_vessel_gain_vs_g0": g1_g0_vessel["mean_delta"]
        >= float(decision_rule["minimum_mean_g1_minus_g0_vessel_macro_f1"]),
        "mean_vessel_gain_vs_g0_c": g1_g0c_vessel["mean_delta"] > 0.0,
        "recording_not_degraded_vs_g0": g1_g0_recording["mean_delta"]
        >= -float(decision_rule["maximum_mean_recording_macro_f1_drop_vs_g0"]),
        "positive_vessel_split_means_vs_g0": g1_g0_vessel["positive_split_means"]
        >= int(decision_rule["minimum_positive_split_means"]),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "run_count": len(rows),
        "runs": rows,
        "summary": summaries,
        "paired_deltas": paired_rows,
        "comparisons": comparisons,
        "decision_gates": gates,
        "attention_supported_by_point_gates": all(gates.values()),
        "test_evaluated": False,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_global_repeat_summary(summary: Mapping[str, object], output_dir: str | Path) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "global_attention_repeat_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "individual_runs.csv", list(summary["runs"]))  # type: ignore[arg-type]
    _write_csv(output / "paired_deltas.csv", list(summary["paired_deltas"]))  # type: ignore[arg-type]
    _write_csv(output / "variant_summary.csv", list(summary["summary"]))  # type: ignore[arg-type]

    lookup = {
        (row["variant"], row["level"]): row for row in summary["summary"]  # type: ignore[union-attr]
    }
    lines = [
        "# DeepShip L20 global-attention repeats",
        "",
        "Validation-only results over 3 vessel splits × 3 optimization seeds.",
        "",
        "| Variant | Vessel macro-F1 | Recording macro-F1 |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        vessel = lookup[(variant, "vessel")]
        recording = lookup[(variant, "recording")]
        lines.append(
            f"| {variant} | {vessel['mean_macro_f1']:.4f} ± {vessel['std_macro_f1']:.4f} | "
            f"{recording['mean_macro_f1']:.4f} ± {recording['std_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Point-estimate decision gates passed: **{summary['attention_supported_by_point_gates']}**.",
            "",
            "DeepShip test was not evaluated.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
