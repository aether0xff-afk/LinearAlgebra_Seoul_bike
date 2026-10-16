"""Small deterministic checks of the regression, validation split, and metrics."""
import unittest

import numpy as np
import pandas as pd

from src.run_analysis import assign_split, design_matrix, fit_models, predict, score, stable_hash


class RegressionTests(unittest.TestCase):
    def test_design_matrix_columns_and_periodicity(self):
        h = np.arange(24)
        w = np.zeros(24)
        self.assertEqual(design_matrix(h, w, "harmonic_24h").shape, (24, 3))
        self.assertEqual(design_matrix(h, w, "harmonic_24h_12h").shape, (24, 5))
        self.assertEqual(design_matrix(h, w, "harmonic_weekend").shape, (24, 10))
        np.testing.assert_allclose(
            design_matrix(h, w, "harmonic_24h_12h"),
            design_matrix(h + 24, w, "harmonic_24h_12h"),
            atol=1e-13,
        )

    def test_exact_least_squares_recovery(self):
        h = np.tile(np.arange(24), 4)
        w = np.repeat([0, 0, 1, 1], 24)
        X = design_matrix(h, w, "harmonic_weekend")
        beta = np.array([650, -200, -80, -100, -50, -40, 5, 6, 70, 80])
        target = X @ beta
        train = pd.DataFrame({"hour": h, "weekend": w, "y": target})
        fitted = fit_models(train)
        np.testing.assert_allclose(fitted["harmonic_weekend"], beta, atol=1e-9)
        np.testing.assert_allclose(
            predict(train, fitted, "harmonic_weekend"), target, atol=1e-9
        )

    def test_split_keeps_days_together_and_repeatable(self):
        dates = pd.date_range("2024-01-01", periods=160, freq="D")
        frame = pd.DataFrame({
            "date_key": np.repeat(dates.strftime("%Y-%m-%d"), 24),
            "hour": np.tile(np.arange(24), 160),
            "weekend": np.repeat((dates.dayofweek >= 5).astype(int), 24),
            "season": np.repeat(["Winter"] * 40 + ["Spring"] * 40
                                + ["Summer"] * 40 + ["Autumn"] * 40, 24)
        })
        a = assign_split(frame)
        b = assign_split(frame)
        self.assertTrue((a["split"] == b["split"]).all())
        self.assertTrue((a.groupby("date_key")["split"].nunique() == 1).all())
        self.assertEqual(set(a["split"]), {"train", "validation", "test"})
        self.assertEqual(stable_hash("2024-01-01|42"),
                         stable_hash("2024-01-01|42"))

    def test_metrics(self):
        result = score(np.array([0, 2, 4]), np.array([1, 2, 3]))
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2 / 3))


if __name__ == "__main__":
    unittest.main()
