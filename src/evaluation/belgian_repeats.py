"""Aggregate the fully crossed Belgian G0/G1 development matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from src.data.deepship import CLASS_NAMES
from src.evaluation.belgian_attention import paired_date_cluster_deltas
from src.utils.pathing import resolve_path


FOLDS = (1, 2, 3)
MODEL_SEEDS = (42, 43, 44)
VARIANTS = ("g0", "g1")


def load_prediction_csv(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with resolve_path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "distance_km": float(row["distance_km"]),
                    "true_label": int(row["true_label"]),
                    "predicted_label": int(row["predicted_label"]),
                }
            )
    if not rows:
        raise ValueError(f"Belgian prediction file is empty: {path}")
    if any(str(row.get("official_split")) == "test" for row in rows):
        raise ValueError(f"Sealed Belgian test rows appeared in development predictions: {path}")
    return rows


def _cell_root(run_root: Path, fold: int, variant: str, seed: int) -> Path:
    return run_root / f"formal_fold{fold}_{variant}_seed{seed}"


def _validate_cell(
    root: Path,
    *,
    fold: int,
    variant: str,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    completion_path = root / "reports" / "run_complete.json"
    prediction_path = root / "predictions" / "validation_best_predictions.csv"
    best_path = root / "models" / "belgian_best.pt"
    last_path = root / "models" / "belgian_last.pt"
    required = (completion_path, prediction_path, best_path, last_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Belgian cell is incomplete: {missing}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "status": "validation_complete",
        "test_evaluated": False,
        "fold": fold,
        "model_seed": seed,
        "model_variant": variant,
    }
    mismatches = {
        key: {"expected": value, "actual": completion.get(key)}
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Belgian cell metadata mismatch at {root}: {mismatches}")
    return completion, load_prediction_csv(prediction_path)


def _hierarchical_draws(
    cell_draws: dict[tuple[int, int], np.ndarray],
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        values: list[float] = []
        for fold in generator.choice(FOLDS, size=len(FOLDS), replace=True):
            for model_seed in generator.choice(
                MODEL_SEEDS, size=len(MODEL_SEEDS), replace=True
            ):
                candidates = cell_draws[(int(fold), int(model_seed))]
                values.append(float(candidates[generator.integers(0, len(candidates))]))
        draws[index] = float(np.mean(values))
    return draws


def summarize_belgian_matrix(
    run_root: str | Path,
    *,
    resamples: int = 50_000,
    seed: int = 42,
) -> dict[str, object]:
    root = resolve_path(run_root)
    completions: dict[tuple[int, int, str], dict[str, object]] = {}
    predictions: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for fold in FOLDS:
        for model_seed in MODEL_SEEDS:
            for variant in VARIANTS:
                completion, rows = _validate_cell(
                    _cell_root(root, fold, variant, model_seed),
                    fold=fold,
                    variant=variant,
                    seed=model_seed,
                )
                completions[(fold, model_seed, variant)] = completion
                predictions[(fold, model_seed, variant)] = rows

    cells = []
    date_draws: dict[tuple[int, int], np.ndarray] = {}
    class_deltas: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    for fold in FOLDS:
        for model_seed in MODEL_SEEDS:
            reference = predictions[(fold, model_seed, "g0")]
            comparison = predictions[(fold, model_seed, "g1")]
            g0_value, g1_value, deltas = paired_date_cluster_deltas(
                reference,
                comparison,
                CLASS_NAMES,
                resamples=resamples,
                seed=seed + 10_000 * fold + model_seed,
            )
            date_draws[(fold, model_seed)] = deltas
            g0_report = completions[(fold, model_seed, "g0")]["best_validation_metrics"]
            g1_report = completions[(fold, model_seed, "g1")]["best_validation_metrics"]
            per_class = {}
            for class_name in CLASS_NAMES:
                g0_f1 = float(
                    g0_report["date_balanced"]["classification_report"][class_name]["f1-score"]
                )
                g1_f1 = float(
                    g1_report["date_balanced"]["classification_report"][class_name]["f1-score"]
                )
                per_class[class_name] = g1_f1 - g0_f1
                class_deltas[class_name].append(g1_f1 - g0_f1)
            cells.append(
                {
                    "fold": fold,
                    "model_seed": model_seed,
                    "g0_date_macro_f1": g0_value,
                    "g1_date_macro_f1": g1_value,
                    "delta_g1_minus_g0": g1_value - g0_value,
                    "date_bootstrap_ci95": [
                        float(np.quantile(deltas, 0.025)),
                        float(np.quantile(deltas, 0.975)),
                    ],
                    "per_class_f1_delta": per_class,
                }
            )
    hierarchical = _hierarchical_draws(date_draws, resamples=resamples, seed=seed)
    mean_g0 = float(np.mean([cell["g0_date_macro_f1"] for cell in cells]))
    mean_g1 = float(np.mean([cell["g1_date_macro_f1"] for cell in cells]))
    fold_means = {
        str(fold): float(
            np.mean(
                [cell["delta_g1_minus_g0"] for cell in cells if cell["fold"] == fold]
            )
        )
        for fold in FOLDS
    }
    mean_class_deltas = {
        name: float(np.mean(values)) for name, values in class_deltas.items()
    }
    interval = [
        float(np.quantile(hierarchical, 0.025)),
        float(np.quantile(hierarchical, 0.975)),
    ]
    decision_checks = {
        "mean_delta_at_least_one_pp": mean_g1 - mean_g0 >= 0.01,
        "hierarchical_ci_lower_above_zero": interval[0] > 0.0,
        "at_least_two_positive_fold_means": sum(value > 0.0 for value in fold_means.values())
        >= 2,
        "passenger_drop_within_three_pp": mean_class_deltas["Passenger"] >= -0.03,
        "tug_drop_within_three_pp": mean_class_deltas["Tug"] >= -0.03,
    }
    return {
        "schema_version": 1,
        "experiment_id": "belgian_attention_v1",
        "run_count": len(completions),
        "paired_cell_count": len(cells),
        "test_evaluated": False,
        "selection_metric": "validation date-balanced macro-F1",
        "g0_mean": mean_g0,
        "g1_mean": mean_g1,
        "mean_delta_g1_minus_g0": mean_g1 - mean_g0,
        "hierarchical_bootstrap_ci95": interval,
        "probability_delta_gt_zero": float(np.mean(hierarchical > 0.0)),
        "paired_cell_wins": sum(cell["delta_g1_minus_g0"] > 0.0 for cell in cells),
        "fold_mean_deltas": fold_means,
        "mean_per_class_f1_deltas": mean_class_deltas,
        "decision_checks": decision_checks,
        "stable_positive_gain": all(decision_checks.values()),
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "levels": ["UTC calendar_date", "model_seed within fold", "fold"],
        },
        "cells": cells,
    }
