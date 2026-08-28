from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.evaluation.group_prediction_analysis import (
    GroupPrediction,
    analyze_group_predictions,
    error_rows,
    load_group_predictions,
)


class GroupPredictionAnalysisTests(unittest.TestCase):
    def _rows(self) -> list[GroupPrediction]:
        return [
            GroupPrediction("a", 0, 0, (0.9, 0.1)),
            GroupPrediction("b", 0, 1, (0.4, 0.6)),
            GroupPrediction("c", 1, 1, (0.2, 0.8)),
            GroupPrediction("d", 1, 1, (0.1, 0.9)),
        ]

    def test_stratified_bootstrap_is_reproducible(self) -> None:
        first = analyze_group_predictions(
            self._rows(), ["A", "B"], bootstrap_resamples=200, seed=7
        )
        second = analyze_group_predictions(
            self._rows(), ["A", "B"], bootstrap_resamples=200, seed=7
        )

        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertAlmostEqual(first["point_estimate"]["accuracy"], 0.75)  # type: ignore[index]

    def test_paired_comparison_uses_identical_group_keys(self) -> None:
        comparison = [
            GroupPrediction(row.group_key, row.true_label, row.true_label, (0.9, 0.1))
            for row in self._rows()
        ]
        comparison[2] = GroupPrediction("c", 1, 1, (0.1, 0.9))
        comparison[3] = GroupPrediction("d", 1, 1, (0.1, 0.9))

        result = analyze_group_predictions(
            self._rows(),
            ["A", "B"],
            bootstrap_resamples=100,
            comparison_rows=comparison,
        )

        paired = result["paired_comparison"]
        self.assertGreater(paired["macro_f1_delta"]["point"], 0)  # type: ignore[index]

    def test_loads_probability_columns_and_lists_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "predictions.csv"
            path.write_text(
                "vessel_key,true_label,predicted_label,probability_A,probability_B\n"
                "a,0,0,0.8,0.2\n"
                "b,1,0,0.7,0.3\n",
                encoding="utf-8",
            )
            rows, class_names, digest = load_group_predictions(path)

        self.assertEqual(class_names, ["A", "B"])
        self.assertEqual(len(digest), 64)
        self.assertEqual(error_rows(rows, class_names)[0]["group_key"], "b")


if __name__ == "__main__":
    unittest.main()
