from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GroupPrediction:
    group_key: str
    true_label: int
    predicted_label: int
    probabilities: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_group_predictions(
    path: str | Path,
    *,
    group_key_field: str = "vessel_key",
) -> tuple[list[GroupPrediction], list[str], str]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prediction CSV has no header: {source}")
        probability_fields = [
            field for field in reader.fieldnames if field.startswith("probability_")
        ]
        if len(probability_fields) < 2:
            raise ValueError(f"Prediction CSV has fewer than two probability columns: {source}")
        required = {group_key_field, "true_label", "predicted_label"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Prediction CSV is missing columns {missing}: {source}")

        rows: list[GroupPrediction] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            group_key = str(row[group_key_field]).strip()
            if not group_key:
                raise ValueError(f"Empty {group_key_field} at row {row_number}: {source}")
            if group_key in seen:
                raise ValueError(f"Duplicate group key {group_key!r}: {source}")
            seen.add(group_key)
            probabilities = np.asarray(
                [float(row[field]) for field in probability_fields], dtype=np.float64
            )
            if not bool(np.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
                raise ValueError(f"Invalid probabilities for {group_key!r}: {source}")
            if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.02, abs_tol=0.02):
                raise ValueError(f"Probabilities do not sum to one for {group_key!r}: {source}")
            true_label = int(row["true_label"])
            predicted_label = int(row["predicted_label"])
            if true_label not in range(len(probability_fields)):
                raise ValueError(f"Invalid true label for {group_key!r}: {true_label}")
            if predicted_label != int(probabilities.argmax()):
                raise ValueError(f"Predicted label disagrees with probabilities for {group_key!r}")
            rows.append(
                GroupPrediction(
                    group_key=group_key,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    probabilities=tuple(float(value) for value in probabilities),
                )
            )
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {source}")
    class_names = [field.removeprefix("probability_") for field in probability_fields]
    return rows, class_names, _sha256(source)


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    per_class = []
    f1_values = []
    for label in range(num_classes):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum() - true_positive)
        false_negative = int(confusion[label, :].sum() - true_positive)
        support = int(confusion[label, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _stratified_bootstrap_indices(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = []
    for label in sorted(int(value) for value in np.unique(labels)):
        indexes = np.flatnonzero(labels == label)
        sampled.append(rng.choice(indexes, size=len(indexes), replace=True))
    return np.concatenate(sampled)


def _interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
    }


def analyze_group_predictions(
    rows: list[GroupPrediction],
    class_names: list[str],
    *,
    bootstrap_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    comparison_rows: list[GroupPrediction] | None = None,
) -> dict[str, object]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    true_labels = np.asarray([row.true_label for row in rows], dtype=np.int64)
    predicted_labels = np.asarray([row.predicted_label for row in rows], dtype=np.int64)
    point = classification_metrics(
        true_labels, predicted_labels, num_classes=len(class_names)
    )
    for item in point["per_class"]:  # type: ignore[union-attr]
        item["class_name"] = class_names[int(item["label"])]

    ordered_comparison: list[GroupPrediction] | None = None
    comparison_point: dict[str, object] | None = None
    if comparison_rows is not None:
        comparison_by_key = {row.group_key: row for row in comparison_rows}
        expected_keys = {row.group_key for row in rows}
        if set(comparison_by_key) != expected_keys:
            raise ValueError("Comparison predictions do not contain identical group keys")
        ordered_comparison = [comparison_by_key[row.group_key] for row in rows]
        if any(
            left.true_label != right.true_label
            for left, right in zip(rows, ordered_comparison, strict=True)
        ):
            raise ValueError("Comparison predictions contain conflicting true labels")
        comparison_predicted = np.asarray(
            [row.predicted_label for row in ordered_comparison], dtype=np.int64
        )
        comparison_point = classification_metrics(
            true_labels, comparison_predicted, num_classes=len(class_names)
        )

    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(bootstrap_resamples, dtype=np.float64)
    macro_f1_samples = np.empty(bootstrap_resamples, dtype=np.float64)
    comparison_accuracy_deltas = np.empty(bootstrap_resamples, dtype=np.float64)
    comparison_f1_deltas = np.empty(bootstrap_resamples, dtype=np.float64)
    comparison_predicted = (
        np.asarray([row.predicted_label for row in ordered_comparison], dtype=np.int64)
        if ordered_comparison is not None
        else None
    )
    for index in range(bootstrap_resamples):
        sampled = _stratified_bootstrap_indices(true_labels, rng=rng)
        metrics = classification_metrics(
            true_labels[sampled], predicted_labels[sampled], num_classes=len(class_names)
        )
        accuracy_samples[index] = float(metrics["accuracy"])
        macro_f1_samples[index] = float(metrics["macro_f1"])
        if comparison_predicted is not None:
            compared = classification_metrics(
                true_labels[sampled],
                comparison_predicted[sampled],
                num_classes=len(class_names),
            )
            comparison_accuracy_deltas[index] = float(compared["accuracy"]) - float(
                metrics["accuracy"]
            )
            comparison_f1_deltas[index] = float(compared["macro_f1"]) - float(
                metrics["macro_f1"]
            )

    analysis: dict[str, object] = {
        "schema_version": 1,
        "group_count": len(rows),
        "class_names": class_names,
        "class_support": {
            class_names[label]: int((true_labels == label).sum())
            for label in range(len(class_names))
        },
        "point_estimate": point,
        "bootstrap": {
            "method": "class-stratified group resampling with replacement",
            "resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
            "accuracy": _interval(accuracy_samples, confidence),
            "macro_f1": _interval(macro_f1_samples, confidence),
        },
    }
    if comparison_point is not None:
        assert comparison_predicted is not None
        analysis["paired_comparison"] = {
            "direction": "comparison_minus_reference",
            "point_estimate": comparison_point,
            "accuracy_delta": {
                "point": float(comparison_point["accuracy"]) - float(point["accuracy"]),
                **_interval(comparison_accuracy_deltas, confidence),
                "probability_greater_than_zero": float(
                    np.mean(comparison_accuracy_deltas > 0)
                ),
            },
            "macro_f1_delta": {
                "point": float(comparison_point["macro_f1"]) - float(point["macro_f1"]),
                **_interval(comparison_f1_deltas, confidence),
                "probability_greater_than_zero": float(np.mean(comparison_f1_deltas > 0)),
            },
        }
    return analysis


def error_rows(
    rows: Iterable[GroupPrediction], class_names: list[str]
) -> list[dict[str, object]]:
    errors = []
    for row in rows:
        if row.true_label == row.predicted_label:
            continue
        probabilities = np.asarray(row.probabilities, dtype=np.float64)
        order = np.argsort(probabilities)[::-1]
        clipped = np.clip(probabilities, 1e-12, 1.0)
        errors.append(
            {
                "group_key": row.group_key,
                "true_class": class_names[row.true_label],
                "predicted_class": class_names[row.predicted_label],
                "predicted_confidence": float(probabilities[row.predicted_label]),
                "true_class_probability": float(probabilities[row.true_label]),
                "top_two_margin": float(probabilities[order[0]] - probabilities[order[1]]),
                "entropy_nats": float(-(clipped * np.log(clipped)).sum()),
            }
        )
    return sorted(
        errors,
        key=lambda row: (-float(row["predicted_confidence"]), str(row["group_key"])),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_group_prediction_analysis(
    analysis: dict[str, object],
    rows: list[GroupPrediction],
    class_names: list[str],
    output_dir: str | Path,
    *,
    source_path: str | Path,
    source_sha256: str,
    comparison_path: str | Path | None = None,
    comparison_sha256: str | None = None,
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        **analysis,
        "source": {"path": str(Path(source_path).expanduser().resolve()), "sha256": source_sha256},
    }
    if comparison_path is not None:
        payload["comparison_source"] = {
            "path": str(Path(comparison_path).expanduser().resolve()),
            "sha256": comparison_sha256,
        }
    (output / "group_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    point = analysis["point_estimate"]
    assert isinstance(point, dict)
    _write_csv(output / "per_class_metrics.csv", list(point["per_class"]))
    _write_csv(output / "misclassified_groups.csv", error_rows(rows, class_names))
    bootstrap = analysis["bootstrap"]
    assert isinstance(bootstrap, dict)
    accuracy = bootstrap["accuracy"]
    macro_f1 = bootstrap["macro_f1"]
    assert isinstance(accuracy, dict) and isinstance(macro_f1, dict)
    lines = [
        "# Group prediction analysis",
        "",
        f"Groups: {analysis['group_count']}",
        f"Accuracy: {float(point['accuracy']):.4f} "
        f"({float(accuracy['lower']):.4f}–{float(accuracy['upper']):.4f} bootstrap CI)",
        f"Macro-F1: {float(point['macro_f1']):.4f} "
        f"({float(macro_f1['lower']):.4f}–{float(macro_f1['upper']):.4f} bootstrap CI)",
        "",
        "The interval resamples groups within each true class. It quantifies validation-set "
        "sampling uncertainty, not training-seed uncertainty.",
    ]
    paired = analysis.get("paired_comparison")
    if isinstance(paired, dict):
        delta = paired["macro_f1_delta"]
        assert isinstance(delta, dict)
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                f"Macro-F1 delta (comparison − reference): {float(delta['point']):+.4f} "
                f"({float(delta['lower']):+.4f}–{float(delta['upper']):+.4f}).",
            ]
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
