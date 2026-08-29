from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.data.deepship import CLASS_NAMES
from src.evaluation.belgian_attention import date_balanced_metrics
from src.evaluation.belgian_repeats import summarize_belgian_matrix


class BelgianRepeatSummaryTests(unittest.TestCase):
    def test_complete_matrix_is_paired_without_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for fold in (1, 2, 3):
                for seed in (42, 43, 44):
                    for variant in ("g0", "g1"):
                        cell = run_root / f"formal_fold{fold}_{variant}_seed{seed}"
                        (cell / "reports").mkdir(parents=True)
                        (cell / "predictions").mkdir()
                        (cell / "models").mkdir()
                        rows = self._prediction_rows(variant)
                        metrics = date_balanced_metrics(rows, CLASS_NAMES)
                        completion = {
                            "status": "validation_complete",
                            "test_evaluated": False,
                            "fold": fold,
                            "model_seed": seed,
                            "model_variant": variant,
                            "best_validation_metrics": {"date_balanced": metrics},
                        }
                        (cell / "reports" / "run_complete.json").write_text(
                            json.dumps(completion), encoding="utf-8"
                        )
                        self._write_predictions(
                            cell / "predictions" / "validation_best_predictions.csv",
                            rows,
                        )
                        (cell / "models" / "belgian_best.pt").touch()
                        (cell / "models" / "belgian_last.pt").touch()

            report = summarize_belgian_matrix(run_root, resamples=100, seed=42)

            self.assertEqual(report["run_count"], 18)
            self.assertEqual(report["paired_cell_count"], 9)
            self.assertFalse(report["test_evaluated"])
            self.assertGreater(report["mean_delta_g1_minus_g0"], 0.0)
            self.assertEqual(report["paired_cell_wins"], 9)

    @staticmethod
    def _prediction_rows(variant: str) -> list[dict[str, object]]:
        rows = []
        for date_index, date in enumerate(("2022-01-01", "2022-01-02")):
            for label, class_name in enumerate(CLASS_NAMES):
                predicted = label
                if variant == "g0" and label == 3 and date_index == 0:
                    predicted = 0
                rows.append(
                    {
                        "relative_path": f"{date}-{class_name}.flac",
                        "official_split": "train",
                        "calendar_date": date,
                        "station": "Grafton",
                        "distance_km": 1.0,
                        "true_label": label,
                        "predicted_label": predicted,
                    }
                )
        return rows

    @staticmethod
    def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
