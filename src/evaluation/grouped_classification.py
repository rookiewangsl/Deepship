from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def aggregate_recording_predictions(
    segment_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in segment_rows:
        grouped[str(row["relative_path"])].append(row)

    results = []
    for relative_path, rows in sorted(grouped.items()):
        labels = {int(row["true_label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Recording has conflicting labels: {relative_path}")
        vessel_keys = {str(row.get("vessel_key", "")) for row in rows}
        if len(vessel_keys) != 1:
            raise ValueError(f"Recording has conflicting vessel keys: {relative_path}")
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64).mean(axis=0)
        results.append(
            {
                "relative_path": relative_path,
                "vessel_key": next(iter(vessel_keys)),
                "true_label": next(iter(labels)),
                "predicted_label": int(probabilities.argmax()),
                "probabilities": probabilities.tolist(),
                "segments": len(rows),
            }
        )
    return results


def aggregate_vessel_predictions(
    recording_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in recording_rows:
        vessel_key = str(row.get("vessel_key", ""))
        if vessel_key:
            grouped[vessel_key].append(row)

    results = []
    for vessel_key, rows in sorted(grouped.items()):
        labels = {int(row["true_label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Vessel group has conflicting labels: {vessel_key}")
        # Recordings are averaged with equal weight so long WAV files do not dominate.
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64).mean(axis=0)
        results.append(
            {
                "vessel_key": vessel_key,
                "true_label": next(iter(labels)),
                "predicted_label": int(probabilities.argmax()),
                "probabilities": probabilities.tolist(),
                "recordings": len(rows),
            }
        )
    return results


def compute_grouped_metrics(
    segment_rows: list[dict[str, object]],
    class_names: list[str],
) -> tuple[
    dict[str, dict[str, object] | None],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not segment_rows:
        raise ValueError("At least one segment prediction is required")

    # Lazy import keeps probability aggregation usable without plotting/metrics
    # dependencies, including lightweight manifest-audit environments.
    from src.evaluation.classification import compute_metrics

    recording_rows = aggregate_recording_predictions(segment_rows)
    vessel_rows = aggregate_vessel_predictions(recording_rows)

    def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
        return compute_metrics(
            [int(row["true_label"]) for row in rows],
            [int(row["predicted_label"]) for row in rows],
            class_names,
        )

    grouped_metrics: dict[str, dict[str, object] | None] = {
        "segment": metrics(segment_rows),
        "recording": metrics(recording_rows),
        "vessel": metrics(vessel_rows) if vessel_rows else None,
    }
    return grouped_metrics, recording_rows, vessel_rows


def save_prediction_rows(
    rows: list[dict[str, object]],
    output_path: str | Path,
    class_names: list[str],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = sorted({key for row in rows for key in row if key != "probabilities"})
    probability_fields = [f"probability_{class_name}" for class_name in class_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=base_fields + probability_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            output = {key: value for key, value in row.items() if key != "probabilities"}
            probabilities = row["probabilities"]
            output.update(
                {
                    field: float(probabilities[index])
                    for index, field in enumerate(probability_fields)
                }
            )
            writer.writerow(output)
