from __future__ import annotations

import unittest

from src.evaluation.model_selection import (
    build_validation_selection,
    primary_metric_improves,
    selection_is_better,
    should_stop_early,
    validate_resume_selection_state,
    validation_selection_rule,
)


def _metrics(
    *,
    segment_macro_f1: float = 0.5,
    segment_accuracy: float = 0.5,
    recording_macro_f1: float = 0.5,
    recording_accuracy: float = 0.5,
    vessel_macro_f1: float | None = 0.5,
    vessel_accuracy: float | None = 0.5,
) -> dict[str, dict[str, object] | None]:
    return {
        "segment": {"macro_f1": segment_macro_f1, "accuracy": segment_accuracy},
        "recording": {
            "macro_f1": recording_macro_f1,
            "accuracy": recording_accuracy,
        },
        "vessel": (
            {"macro_f1": vessel_macro_f1, "accuracy": vessel_accuracy}
            if vessel_macro_f1 is not None and vessel_accuracy is not None
            else None
        ),
    }


class ConformerModelSelectionTests(unittest.TestCase):
    def test_vessel_protocol_prioritizes_vessel_macro_f1_over_segment_accuracy(self) -> None:
        incumbent = build_validation_selection(
            "vessel_name_disjoint",
            _metrics(vessel_macro_f1=0.60, segment_accuracy=0.95),
            0.5,
        )
        candidate = build_validation_selection(
            "vessel_name_disjoint",
            _metrics(vessel_macro_f1=0.61, segment_accuracy=0.20),
            1.0,
        )

        self.assertTrue(selection_is_better(candidate, incumbent))

    def test_vessel_protocol_uses_documented_tie_break_order(self) -> None:
        incumbent = build_validation_selection(
            "vessel_name_disjoint",
            _metrics(
                vessel_macro_f1=0.60,
                vessel_accuracy=0.70,
                recording_macro_f1=0.80,
            ),
            0.2,
        )
        candidate = build_validation_selection(
            "vessel_name_disjoint",
            _metrics(
                vessel_macro_f1=0.60,
                vessel_accuracy=0.71,
                recording_macro_f1=0.10,
            ),
            2.0,
        )

        self.assertTrue(selection_is_better(candidate, incumbent))

    def test_validation_loss_is_minimized_after_metric_ties(self) -> None:
        incumbent = build_validation_selection(
            "recording_disjoint", _metrics(vessel_macro_f1=None), 0.5
        )
        candidate = build_validation_selection(
            "recording_disjoint", _metrics(vessel_macro_f1=None), 0.4
        )

        self.assertTrue(selection_is_better(candidate, incumbent))

    def test_rejects_nonfinite_selection_values(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            build_validation_selection(
                "recording_disjoint",
                _metrics(recording_macro_f1=float("nan"), vessel_macro_f1=None),
                0.4,
            )

    def test_rejects_comparison_across_protocol_rules(self) -> None:
        vessel = build_validation_selection(
            "vessel_name_disjoint", _metrics(), 0.4
        )
        recording = build_validation_selection(
            "recording_disjoint", _metrics(vessel_macro_f1=None), 0.4
        )
        with self.assertRaisesRegex(ValueError, "different rules"):
            selection_is_better(vessel, recording)

    def test_protocol_rules_match_declared_primary_levels(self) -> None:
        self.assertEqual(
            validation_selection_rule("vessel_name_disjoint")["name"],
            "vessel_macro_f1",
        )
        self.assertEqual(
            validation_selection_rule("recording_disjoint")["name"],
            "recording_macro_f1",
        )
        self.assertEqual(
            validation_selection_rule("segment_level")["name"],
            "segment_macro_f1",
        )

    def test_early_stopping_counts_epochs_since_group_metric_improvement(self) -> None:
        self.assertFalse(
            should_stop_early(improved=False, epoch=8, best_epoch=1, patience=8)
        )
        self.assertTrue(
            should_stop_early(improved=False, epoch=9, best_epoch=1, patience=8)
        )
        self.assertFalse(
            should_stop_early(improved=True, epoch=9, best_epoch=9, patience=8)
        )

    def test_early_stopping_min_delta_uses_only_the_primary_group_metric(self) -> None:
        candidate = build_validation_selection(
            "vessel_name_disjoint",
            _metrics(vessel_macro_f1=0.504, vessel_accuracy=0.9),
            0.1,
        )

        self.assertFalse(
            primary_metric_improves(candidate, 0.5, min_delta=0.005)
        )
        self.assertTrue(primary_metric_improves(candidate, 0.49, min_delta=0.005))
        self.assertTrue(primary_metric_improves(candidate, None, min_delta=0.005))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            primary_metric_improves(candidate, 0.5, min_delta=-0.001)

    def test_resume_requires_matching_selection_schema_and_rule(self) -> None:
        rule = validation_selection_rule("vessel_name_disjoint")
        selection = build_validation_selection(
            "vessel_name_disjoint", _metrics(), 0.4
        )
        checkpoint = {
            "split_manifest_sha256": "manifest-a",
            "selection_schema_version": 2,
            "selection_rule": rule,
            "best_selection": selection,
        }

        validate_resume_selection_state(
            checkpoint,
            manifest_sha256="manifest-a",
            protocol="vessel_name_disjoint",
        )
        checkpoint["selection_schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "predates group-aware"):
            validate_resume_selection_state(
                checkpoint,
                manifest_sha256="manifest-a",
                protocol="vessel_name_disjoint",
            )

if __name__ == "__main__":
    unittest.main()
