from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_2d_float_array(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    return values


def _validate_metadata_length(num_rows: int, **columns: Sequence[object]) -> None:
    mismatches = {
        name: len(values)
        for name, values in columns.items()
        if len(values) != num_rows
    }
    if mismatches:
        raise ValueError(f"metadata length mismatch for {mismatches}; expected {num_rows}")


def aggregate_by_recording(
    embeddings: np.ndarray,
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    recording_paths: Sequence[str],
) -> tuple[np.ndarray, dict[str, list[str] | list[int]]]:
    values = _as_2d_float_array(embeddings)
    _validate_metadata_length(
        values.shape[0],
        class_names=class_names,
        vessel_keys=vessel_keys,
        recording_paths=recording_paths,
    )

    indexes_by_recording: dict[str, list[int]] = defaultdict(list)
    for index, recording_path in enumerate(recording_paths):
        indexes_by_recording[str(recording_path)].append(index)

    recording_embeddings: list[np.ndarray] = []
    output_classes: list[str] = []
    output_vessels: list[str] = []
    output_paths: list[str] = []
    output_segments: list[int] = []
    for recording_path in sorted(indexes_by_recording):
        indexes = indexes_by_recording[recording_path]
        recording_classes = {str(class_names[index]) for index in indexes}
        recording_vessels = {str(vessel_keys[index]) for index in indexes}
        if len(recording_classes) != 1 or len(recording_vessels) != 1:
            raise ValueError(f"inconsistent metadata within recording {recording_path!r}")
        recording_embeddings.append(values[indexes].mean(axis=0))
        output_classes.append(recording_classes.pop())
        output_vessels.append(recording_vessels.pop())
        output_paths.append(recording_path)
        output_segments.append(len(indexes))

    return np.stack(recording_embeddings), {
        "class_names": output_classes,
        "vessel_keys": output_vessels,
        "recording_paths": output_paths,
        "segment_counts": output_segments,
    }


def safe_silhouette(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    max_samples: int,
    seed: int,
) -> float | None:
    values = _as_2d_float_array(embeddings)
    _validate_metadata_length(values.shape[0], labels=labels)
    label_counts = Counter(str(label) for label in labels)
    keep = np.asarray([label_counts[str(label)] >= 2 for label in labels], dtype=bool)
    values = values[keep]
    kept_labels = np.asarray(labels, dtype=str)[keep]
    if (
        values.shape[0] < 3
        or len(set(kept_labels)) < 2
        or len(set(kept_labels)) >= values.shape[0]
    ):
        return None
    sample_size = min(int(max_samples), values.shape[0])
    if sample_size < 3:
        return None
    return float(
        silhouette_score(
            values,
            kept_labels,
            metric="cosine",
            sample_size=sample_size if sample_size < values.shape[0] else None,
            random_state=seed,
        )
    )


def within_class_vessel_silhouette(
    embeddings: np.ndarray,
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    max_samples_per_class: int,
    seed: int,
) -> dict[str, object]:
    values = _as_2d_float_array(embeddings)
    _validate_metadata_length(
        values.shape[0], class_names=class_names, vessel_keys=vessel_keys
    )
    classes = np.asarray(class_names, dtype=str)
    vessels = np.asarray(vessel_keys, dtype=str)
    per_class: dict[str, float | None] = {}
    weighted_values: list[tuple[float, int]] = []
    for class_name in sorted(set(classes)):
        class_mask = classes == class_name
        score = safe_silhouette(
            values[class_mask],
            vessels[class_mask],
            max_samples=max_samples_per_class,
            seed=seed,
        )
        per_class[class_name] = score
        if score is not None:
            weighted_values.append((score, int(class_mask.sum())))
    weighted_mean = None
    if weighted_values:
        weighted_mean = float(
            sum(score * count for score, count in weighted_values)
            / sum(count for _, count in weighted_values)
        )
    return {"weighted_mean": weighted_mean, "per_class": per_class}


def nearest_neighbor_identity_rates(
    embeddings: np.ndarray,
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    max_samples: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    values = _as_2d_float_array(embeddings)
    _validate_metadata_length(
        values.shape[0], class_names=class_names, vessel_keys=vessel_keys
    )
    rng = np.random.default_rng(seed)
    sample_size = min(int(max_samples), values.shape[0])
    if sample_size < 2:
        return {
            "samples": int(sample_size),
            "status": "unavailable",
        }
    indexes = np.arange(values.shape[0])
    if sample_size < values.shape[0]:
        indexes = np.sort(rng.choice(indexes, size=sample_size, replace=False))
    sampled_values = values[indexes]
    sampled_classes = np.asarray(class_names, dtype=str)[indexes]
    sampled_vessels = np.asarray(vessel_keys, dtype=str)[indexes]

    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    neighbors.fit(sampled_values)
    _, neighbor_indexes = neighbors.kneighbors(sampled_values)
    nearest = neighbor_indexes[:, 1]
    same_vessel = sampled_vessels == sampled_vessels[nearest]
    same_class = sampled_classes == sampled_classes[nearest]

    random_same_vessel_within_class = []
    for class_name, vessel_key in zip(sampled_classes, sampled_vessels, strict=True):
        class_mask = sampled_classes == class_name
        denominator = int(class_mask.sum()) - 1
        if denominator <= 0:
            continue
        numerator = int(((sampled_vessels == vessel_key) & class_mask).sum()) - 1
        random_same_vessel_within_class.append(numerator / denominator)

    return {
        "samples": int(sample_size),
        "same_vessel_rate": float(same_vessel.mean()),
        "same_class_rate": float(same_class.mean()),
        "same_class_different_vessel_rate": float((same_class & ~same_vessel).mean()),
        "different_class_rate": float((~same_class).mean()),
        "random_same_vessel_within_class_rate": (
            float(np.mean(random_same_vessel_within_class))
            if random_same_vessel_within_class
            else None
        ),
    }


def build_recording_disjoint_probe_split(
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    recording_paths: Sequence[str],
    seed: int,
) -> dict[str, object]:
    num_rows = len(class_names)
    _validate_metadata_length(
        num_rows, vessel_keys=vessel_keys, recording_paths=recording_paths
    )
    recordings_by_vessel: dict[tuple[str, str], set[str]] = defaultdict(set)
    for class_name, vessel_key, recording_path in zip(
        class_names, vessel_keys, recording_paths, strict=True
    ):
        recordings_by_vessel[(str(class_name), str(vessel_key))].add(str(recording_path))

    eligible = {
        key: sorted(recordings)
        for key, recordings in recordings_by_vessel.items()
        if len(recordings) >= 2
    }
    rng = np.random.default_rng(seed)
    held_out_recordings = {
        key: recordings[int(rng.integers(0, len(recordings)))]
        for key, recordings in sorted(eligible.items())
    }

    train_indexes: list[int] = []
    test_indexes: list[int] = []
    label_by_vessel = {
        key: f"{key[0]}::{key[1]}" for key in sorted(held_out_recordings)
    }
    for index, (class_name, vessel_key, recording_path) in enumerate(
        zip(class_names, vessel_keys, recording_paths, strict=True)
    ):
        key = (str(class_name), str(vessel_key))
        if key not in held_out_recordings:
            continue
        if str(recording_path) == held_out_recordings[key]:
            test_indexes.append(index)
        else:
            train_indexes.append(index)

    return {
        "train_indexes": train_indexes,
        "test_indexes": test_indexes,
        "eligible_vessels": len(eligible),
        "held_out_recordings": {
            label_by_vessel[key]: value for key, value in held_out_recordings.items()
        },
    }


def vessel_linear_probe(
    embeddings: np.ndarray,
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    recording_paths: Sequence[str],
    seed: int,
) -> dict[str, object]:
    values = _as_2d_float_array(embeddings)
    split = build_recording_disjoint_probe_split(
        class_names=class_names,
        vessel_keys=vessel_keys,
        recording_paths=recording_paths,
        seed=seed,
    )
    train_indexes = np.asarray(split["train_indexes"], dtype=int)
    test_indexes = np.asarray(split["test_indexes"], dtype=int)
    compact_split = {
        "eligible_vessels": int(split["eligible_vessels"]),
        "held_out_recordings": len(split["held_out_recordings"]),
    }
    if len(train_indexes) == 0 or len(test_indexes) == 0:
        return {**compact_split, "status": "unavailable"}

    labels = np.asarray(
        [
            f"{class_name}::{vessel_key}"
            for class_name, vessel_key in zip(
                class_names,
                vessel_keys,
                strict=True,
            )
        ],
        dtype=str,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(values[train_indexes], labels[train_indexes])
    predictions = model.predict(values[test_indexes])
    test_labels = labels[test_indexes]
    return {
        **compact_split,
        "status": "completed",
        "train_segments": int(len(train_indexes)),
        "test_segments": int(len(test_indexes)),
        "accuracy": float((predictions == test_labels).mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "uniform_chance_accuracy": 1.0 / int(split["eligible_vessels"]),
    }


def diagnose_embeddings(
    embeddings: np.ndarray,
    *,
    class_names: Sequence[str],
    vessel_keys: Sequence[str],
    recording_paths: Sequence[str],
    seed: int,
    max_metric_samples: int = 3000,
) -> dict[str, object]:
    values = _as_2d_float_array(embeddings)
    _validate_metadata_length(
        values.shape[0],
        class_names=class_names,
        vessel_keys=vessel_keys,
        recording_paths=recording_paths,
    )
    recording_values, recording_metadata = aggregate_by_recording(
        values,
        class_names=class_names,
        vessel_keys=vessel_keys,
        recording_paths=recording_paths,
    )
    return {
        "segments": int(values.shape[0]),
        "recordings": int(recording_values.shape[0]),
        "vessels": len(set(str(value) for value in vessel_keys)),
        "embedding_dimension": int(values.shape[1]),
        "segment_class_silhouette": safe_silhouette(
            values,
            class_names,
            max_samples=max_metric_samples,
            seed=seed,
        ),
        "segment_within_class_vessel_silhouette": within_class_vessel_silhouette(
            values,
            class_names=class_names,
            vessel_keys=vessel_keys,
            max_samples_per_class=max_metric_samples,
            seed=seed,
        ),
        "recording_class_silhouette": safe_silhouette(
            recording_values,
            recording_metadata["class_names"],
            max_samples=max_metric_samples,
            seed=seed,
        ),
        "recording_within_class_vessel_silhouette": within_class_vessel_silhouette(
            recording_values,
            class_names=recording_metadata["class_names"],
            vessel_keys=recording_metadata["vessel_keys"],
            max_samples_per_class=max_metric_samples,
            seed=seed,
        ),
        "recording_nearest_neighbor": nearest_neighbor_identity_rates(
            recording_values,
            class_names=recording_metadata["class_names"],
            vessel_keys=recording_metadata["vessel_keys"],
            max_samples=max_metric_samples,
            seed=seed,
        ),
        "recording_disjoint_vessel_linear_probe": vessel_linear_probe(
            values,
            class_names=class_names,
            vessel_keys=vessel_keys,
            recording_paths=recording_paths,
            seed=seed,
        ),
    }
