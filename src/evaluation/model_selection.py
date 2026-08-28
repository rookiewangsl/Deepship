from __future__ import annotations

import math


SELECTION_SCHEMA_VERSION = 2


def validation_selection_rule(protocol: str) -> dict[str, object]:
    rules: dict[str, dict[str, object]] = {
        "vessel_name_disjoint": {
            "name": "vessel_macro_f1",
            "ordered_metrics": [
                {"level": "vessel", "metric": "macro_f1", "direction": "max"},
                {"level": "vessel", "metric": "accuracy", "direction": "max"},
                {"level": "recording", "metric": "macro_f1", "direction": "max"},
                {"level": "validation", "metric": "loss", "direction": "min"},
            ],
        },
        "recording_disjoint": {
            "name": "recording_macro_f1",
            "ordered_metrics": [
                {"level": "recording", "metric": "macro_f1", "direction": "max"},
                {"level": "recording", "metric": "accuracy", "direction": "max"},
                {"level": "segment", "metric": "macro_f1", "direction": "max"},
                {"level": "validation", "metric": "loss", "direction": "min"},
            ],
        },
        "segment_level": {
            "name": "segment_macro_f1",
            "ordered_metrics": [
                {"level": "segment", "metric": "macro_f1", "direction": "max"},
                {"level": "segment", "metric": "accuracy", "direction": "max"},
                {"level": "validation", "metric": "loss", "direction": "min"},
            ],
        },
    }
    if protocol not in rules:
        raise ValueError(f"Unsupported validation selection protocol: {protocol}")
    return rules[protocol]


def build_validation_selection(
    protocol: str,
    grouped_metrics: dict[str, dict[str, object] | None],
    val_loss: float,
) -> dict[str, object]:
    rule = validation_selection_rule(protocol)
    ordered_metrics = rule["ordered_metrics"]
    if not isinstance(ordered_metrics, list):
        raise TypeError("Validation selection rule has invalid ordered_metrics")

    values: dict[str, float] = {}
    score = []
    for item in ordered_metrics:
        if not isinstance(item, dict):
            raise TypeError("Validation selection rule contains an invalid metric")
        level = str(item["level"])
        metric = str(item["metric"])
        direction = str(item["direction"])
        if level == "validation" and metric == "loss":
            value = float(val_loss)
        else:
            level_metrics = grouped_metrics.get(level)
            if level_metrics is None:
                raise ValueError(
                    f"Selection rule requires unavailable validation level: {level}"
                )
            value = float(level_metrics[metric])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Non-finite validation selection value: {level}.{metric}={value}"
            )
        values[f"{level}.{metric}"] = value
        score.append(value if direction == "max" else -value)
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "rule": rule,
        "values": values,
        "score": score,
        "primary_value": score[0],
    }


def selection_is_better(
    candidate: dict[str, object],
    incumbent: dict[str, object] | None,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if incumbent is None:
        return True
    if candidate.get("rule") != incumbent.get("rule"):
        raise ValueError("Cannot compare validation selections with different rules")
    candidate_score = [float(value) for value in candidate["score"]]
    incumbent_score = [float(value) for value in incumbent["score"]]
    if len(candidate_score) != len(incumbent_score):
        raise ValueError("Cannot compare validation selections with different rules")
    for candidate_value, incumbent_value in zip(
        candidate_score, incumbent_score, strict=True
    ):
        if candidate_value > incumbent_value + tolerance:
            return True
        if candidate_value < incumbent_value - tolerance:
            return False
    return False


def should_stop_early(
    *,
    improved: bool,
    epoch: int,
    best_epoch: int,
    patience: int,
) -> bool:
    return not improved and epoch - best_epoch >= patience


def validate_resume_selection_state(
    checkpoint: dict[str, object],
    *,
    manifest_sha256: str,
    protocol: str,
) -> None:
    if checkpoint.get("split_manifest_sha256") != manifest_sha256:
        raise ValueError("Resume checkpoint uses a different split manifest")
    if checkpoint.get("selection_schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(
            "Resume checkpoint predates group-aware validation selection. "
            "Finish it with the original code or start a new output directory."
        )
    expected_rule = validation_selection_rule(protocol)
    if checkpoint.get("selection_rule") != expected_rule:
        raise ValueError("Resume checkpoint uses a different validation selection rule")
    best_selection = checkpoint.get("best_selection")
    if not isinstance(best_selection, dict) or best_selection.get("rule") != expected_rule:
        raise ValueError("Resume checkpoint has invalid best-selection state")
