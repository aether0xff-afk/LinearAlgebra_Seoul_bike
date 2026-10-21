"""Regression tests for the more detailed M4 mathematical claims in the report."""
import unittest

import numpy as np
import pandas as pd

from src.improved_model import design_improved, fit
from src.run_analysis import assign_split, read_data, design_matrix, predict, fit_models, score


class ImprovedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, audit = read_data("data/SeoulBikeData.csv")
        cls.df = assign_split(df, seed=42)
        cls.train = cls.df[cls.df["split"] == "train"]
        cls.valid = cls.df[cls.df["split"] == "validation"]
        cls.test = cls.df[cls.df["split"] == "test"]
        cls.audit = audit

    def test_original_counts_and_no_date_overlap(self):
        self.assertEqual(self.audit["original_rows"], 8760)
        self.assertEqual(self.audit["non_functioning_rows"], 295)
        self.assertEqual(self.audit["functioning_rows"], 8465)
        self.assertEqual((len(self.train), len(self.valid), len(self.test)),
                         (5832, 1121, 1512))
        self.assertTrue(set(self.train["date_key"]).isdisjoint(
            self.valid["date_key"]))
        self.assertTrue(set(self.train["date_key"]).isdisjoint(
            self.test["date_key"]))
        self.assertTrue(set(self.valid["date_key"]).isdisjoint(
            self.test["date_key"]))

    def test_m4_full_rank_and_fitted_metrics(self):
        x_train = design_improved(self.train)
        self.assertEqual(x_train.shape, (5832, 16))
        self.assertEqual(np.linalg.matrix_rank(x_train), 16)
        y_train = self.train["y"].to_numpy()
        beta = fit(x_train, y_train, lam=0)
        x_test = design_improved(self.test)
        m = score(self.test["y"].to_numpy(), x_test @ beta)
        self.assertAlmostEqual(m["mae"], 280.575010, places=3)
        self.assertAlmostEqual(m["rmse"], 389.953939, places=3)
        self.assertAlmostEqual(m["r2"], 0.62130193, places=3)

    def test_ridge_choice_based_on_validation(self):
        x_train = design_improved(self.train)
        y_train = self.train["y"].to_numpy()
        x_valid = design_improved(self.valid)
        y_valid = self.valid["y"].to_numpy()
        choices = (0, 1, 10, 100, 1000)
        observed = {
            lam: score(y_valid,
                       x_valid @ fit(x_train, y_train, lam))["rmse"]
            for lam in choices
        }
        self.assertEqual(min(observed, key=observed.get), 0)
        self.assertAlmostEqual(observed[0], 424.328123, places=3)
        self.assertGreater(observed[1000], observed[0])

    def test_residual_bias_and_squared_error_identity(self):
        old = fit_models(self.train)
        winter = self.test[self.test["Seasons"] == "Winter"]
        e = winter["y"].to_numpy() - predict(
            winter, old, "harmonic_weekend")
        self.assertEqual(len(e), 384)
        self.assertAlmostEqual(e.mean(), -492.701144, places=2)
        rmse_sq = np.mean(e**2)
        self.assertAlmostEqual(rmse_sq, np.var(e) + e.mean()**2,
                               places=6)
        self.assertAlmostEqual(e.mean()**2 / rmse_sq, 0.732, places=2)

    def test_plotting_hour_means_are_from_source(self):
        means = self.df.groupby("hour")["y"].mean()
        self.assertAlmostEqual(means.loc[4], 137.5, delta=0.25)
        self.assertAlmostEqual(means.loc[8], 1050.2, delta=1)
        self.assertAlmostEqual(means.loc[18], 1554.0, delta=1)


if __name__ == "__main__":
    unittest.main()
