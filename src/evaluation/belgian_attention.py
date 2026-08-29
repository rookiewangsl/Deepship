"""Date-balanced Belgian validation metrics and paired cluster bootstrap."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def date_balanced_metrics(
    rows: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    if not rows:
        raise ValueError("At least one Belgian prediction is required")
    date_counts = Counter(str(row["calendar_date"]) for row in rows)
    weights = np.asarray(
        [1.0 / date_counts[str(row["calendar_date"])] for row in rows],
        dtype=np.float64,
    )
    y_true = [int(row["true_label"]) for row in rows]
    y_pred = [int(row["predicted_label"]) for row in rows]
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        sample_weight=weights,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=weights)
    return {
        "dates": len(date_counts),
        "samples": len(rows),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "weighting": "each UTC calendar date has equal total weight",
    }


def stratified_metrics(
    rows: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    by_station: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_distance: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_station[str(row["station"])].append(row)
        distance = float(row["distance_km"])
        if distance <= 1.0:
            distance_bin = "0-1"
        elif distance <= 2.0:
            distance_bin = "1-2"
        elif distance <= 3.0:
            distance_bin = "2-3"
        else:
            distance_bin = "3-5"
        by_distance[distance_bin].append(row)
    return {
        "station": {
            key: date_balanced_metrics(value, class_names)
            for key, value in sorted(by_station.items())
        },
        "distance_km": {
            key: date_balanced_metrics(value, class_names)
            for key, value in sorted(by_distance.items())
        },
    }


def paired_date_cluster_bootstrap(
    reference_rows: Sequence[dict[str, object]],
    comparison_rows: Sequence[dict[str, object]],
    class_names: Sequence[str],
    *,
    resamples: int = 50_000,
    seed: int = 42,
) -> dict[str, object]:
    reference_point, comparison_point, deltas = paired_date_cluster_deltas(
        reference_rows,
        comparison_rows,
        class_names,
        resamples=resamples,
        seed=seed,
    )
    return {
        "reference_macro_f1": reference_point,
        "comparison_macro_f1": comparison_point,
        "delta_macro_f1": comparison_point - reference_point,
        "confidence_interval_95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "resamples": resamples,
        "seed": seed,
        "cluster": "UTC calendar_date",
    }


def _macro_f1_from_confusion(matrices: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    predicted = matrices.sum(axis=-2)
    actual = matrices.sum(axis=-1)
    denominator = actual + predicted
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0.0,
    )
    return class_f1.mean(axis=-1)


def paired_date_cluster_deltas(
    reference_rows: Sequence[dict[str, object]],
    comparison_rows: Sequence[dict[str, object]],
    class_names: Sequence[str],
    *,
    resamples: int = 50_000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Return point estimates and efficient paired UTC-date bootstrap draws."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    key = lambda row: str(row["relative_path"])
    reference = {key(row): row for row in reference_rows}
    comparison = {key(row): row for row in comparison_rows}
    if reference.keys() != comparison.keys():
        raise ValueError("Paired Belgian predictions must contain identical files")
    dates = sorted({str(row["calendar_date"]) for row in reference.values()})
    if not dates:
        raise ValueError("Paired Belgian predictions do not contain calendar dates")
    class_count = len(class_names)
    date_index = {date: index for index, date in enumerate(dates)}
    reference_matrices = np.zeros((len(dates), class_count, class_count), dtype=np.float64)
    comparison_matrices = np.zeros_like(reference_matrices)
    date_sizes = np.zeros(len(dates), dtype=np.float64)
    for relative_path, reference_row in reference.items():
        comparison_row = comparison[relative_path]
        if int(reference_row["true_label"]) != int(comparison_row["true_label"]):
            raise ValueError(f"Paired Belgian labels disagree: {relative_path}")
        if str(reference_row["calendar_date"]) != str(comparison_row["calendar_date"]):
            raise ValueError(f"Paired Belgian dates disagree: {relative_path}")
        date = date_index[str(reference_row["calendar_date"])]
        truth = int(reference_row["true_label"])
        reference_prediction = int(reference_row["predicted_label"])
        comparison_prediction = int(comparison_row["predicted_label"])
        reference_matrices[date, truth, reference_prediction] += 1.0
        comparison_matrices[date, truth, comparison_prediction] += 1.0
        date_sizes[date] += 1.0
    reference_matrices /= date_sizes[:, None, None]
    comparison_matrices /= date_sizes[:, None, None]
    reference_point = float(_macro_f1_from_confusion(reference_matrices.mean(axis=0)))
    comparison_point = float(_macro_f1_from_confusion(comparison_matrices.mean(axis=0)))
    generator = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    batch_size = 1_000
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        sampled = generator.integers(0, len(dates), size=(stop - start, len(dates)))
        reference_draws = reference_matrices[sampled].mean(axis=1)
        comparison_draws = comparison_matrices[sampled].mean(axis=1)
        deltas[start:stop] = _macro_f1_from_confusion(
            comparison_draws
        ) - _macro_f1_from_confusion(reference_draws)
    return reference_point, comparison_point, deltas
