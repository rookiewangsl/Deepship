from __future__ import annotations

import unittest

from src.evaluation.macnna_global_repeats import hierarchical_bootstrap


class MACNNAGlobalRepeatSummaryTests(unittest.TestCase):
    def test_hierarchical_bootstrap_preserves_positive_paired_effect(self) -> None:
        values = {
            42: {42: 0.02, 43: 0.03, 44: 0.01},
            43: {42: 0.04, 43: 0.02, 44: 0.03},
            44: {42: 0.01, 43: 0.02, 44: 0.01},
        }
        result = hierarchical_bootstrap(values, resamples=2000, seed=42)

        self.assertGreater(result["lower"], 0.0)
        self.assertEqual(result["probability_greater_than_zero"], 1.0)
        self.assertAlmostEqual(result["point"], sum(sum(row.values()) for row in values.values()) / 9)

    def test_hierarchical_bootstrap_rejects_incomplete_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "same model seeds"):
            hierarchical_bootstrap(
                {42: {42: 0.1, 43: 0.2}, 43: {42: 0.1}},
                resamples=100,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
