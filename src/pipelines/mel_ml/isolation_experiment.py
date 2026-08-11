from __future__ import annotations

from typing import Mapping


def find_training_config_mismatches(
    train_config: Mapping[str, object],
    experiment_config: Mapping[str, object],
) -> list[str]:
    features = experiment_config["features"]
    training = experiment_config["training"]
    split = experiment_config["split"]
    assert isinstance(features, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(split, Mapping)
    targets = split["target_segments_per_class"]
    assert isinstance(targets, Mapping)

    expected = {
        "clip_duration": features["clip_duration_seconds"],
        "n_fft": features["n_fft"],
        "hop_length": features["hop_length"],
        "win_length": features["win_length"],
        "n_mels": features["n_mels"],
        "highpass_freq": features["highpass_freq"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "momentum": training["momentum"],
        "min_learning_rate": training["min_learning_rate"],
        "warmup_epochs": training["warmup_epochs"],
        "early_stopping_patience": training["early_stopping_patience"],
        "train_per_class": targets["train"],
        "val_per_class": targets["val"],
        "test_per_class": targets["test"],
    }
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = train_config.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{field}: expected {expected_value!r}, got {actual_value!r}")

    allowed_seeds = training["model_seeds"]
    if train_config.get("seed") not in allowed_seeds:
        mismatches.append(
            f"seed: expected one of {list(allowed_seeds)!r}, got {train_config.get('seed')!r}"
        )
    for field in ("max_train_batches", "max_eval_batches"):
        if train_config.get(field) is not None:
            mismatches.append(f"{field}: expected None, got {train_config.get(field)!r}")
    return mismatches


def enforce_training_config(
    train_config: Mapping[str, object],
    experiment_config: Mapping[str, object],
    *,
    allow_overrides: bool,
) -> list[str]:
    mismatches = find_training_config_mismatches(train_config, experiment_config)
    if mismatches and not allow_overrides:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(
            "Training configuration differs from the frozen isolation experiment. "
            "Use --allow-experiment-overrides only for smoke/debug runs:\n" + details
        )
    return mismatches
