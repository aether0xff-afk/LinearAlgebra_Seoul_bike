"""Reproduce the M4 extension used in the school modeling report.

M4 refits all M3 Fourier/weekend coefficients and adds observed temperature,
rainfall, holiday and three season indicators. These are *conditional*
predictions using weather measured at the corresponding hour; they do not
establish next-year forecasting accuracy.

Usage: python -m src.improved_model
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from src.run_analysis import read_data, assign_split, design_matrix, score

MODEL_ORDER = [
    "a0", "a1_sin24", "b1_cos24", "a2_sin12", "b2_cos12",
    "c0_weekend", "c1_weekend_sin24", "d1_weekend_cos24",
    "c2_weekend_sin12", "d2_weekend_cos12",
    "theta_temperature_per10C", "theta_rainfall_per10mm",
    "theta_holiday", "delta_spring", "delta_summer", "delta_autumn",
]
LAMBDAS = (0, 1, 10, 100, 1000)


def design_improved(df: pd.DataFrame) -> np.ndarray:
    base = design_matrix(
        df["hour"].to_numpy(),
        df["weekend"].to_numpy(),
        "harmonic_weekend",
    )
    extras = np.column_stack([
        df["Temperature(C)"].to_numpy(dtype=float) / 10.0,
        df["Rainfall(mm)"].to_numpy(dtype=float) / 10.0,
        df["Holiday"].eq("Holiday").to_numpy(dtype=float),
        df["Seasons"].eq("Spring").to_numpy(dtype=float),
        df["Seasons"].eq("Summer").to_numpy(dtype=float),
        df["Seasons"].eq("Autumn").to_numpy(dtype=float),
    ])
    X = np.column_stack([base, extras])
    assert X.shape[1] == len(MODEL_ORDER)
    return X


def fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    # Penalize all terms EXCEPT the constant intercept (column zero).
    d = np.diag([0.0] + [np.sqrt(lam)] * (X.shape[1] - 1))
    X_aug = np.vstack([X, d])
    y_aug = np.concatenate([y, np.zeros(X.shape[1])])
    return np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]


def execute(data: Path, output: Path) -> None:
    df, audit = read_data(data)
    df = assign_split(df, seed=42)
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    X_tr = design_improved(train)
    X_val = design_improved(validation)
    X_te = design_improved(test)
    y_tr = train["y"].to_numpy(dtype=float)
    y_val = validation["y"].to_numpy(dtype=float)
    y_te = test["y"].to_numpy(dtype=float)

    candidates = []
    for lam in LAMBDAS:
        beta = fit(X_tr, y_tr, lam)
        m = score(y_val, X_val @ beta)
        candidates.append(dict(lam=lam, rmse=m["rmse"], mae=m["mae"], beta=beta))

    # Validation decides lambda. Test data is NOT used for lambda selection.
    best = min(candidates, key=lambda x: x["rmse"])
    beta = best["beta"]
    prediction = X_te @ beta
    metrics = score(y_te, prediction)

    assert audit["original_rows"] == 8760
    assert audit["functioning_rows"] == 8465
    assert (len(train), len(validation), len(test)) == (5832, 1121, 1512)
    assert best["lam"] == 0, f"Unexpected lambda: {best['lam']}"
    assert abs(metrics["mae"] - 280.575010) < 0.01
    assert abs(metrics["rmse"] - 389.953939) < 0.01

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "term": MODEL_ORDER,
        "coefficient": beta,
    }).to_csv(output / "improved_coefficients.csv",
              index=False, float_format="%.10f")
    pd.DataFrame([
        {"lambda": item["lam"], "validation_mae": item["mae"],
         "validation_rmse": item["rmse"]}
        for item in sorted(candidates, key=lambda x: x["lam"])
    ]).to_csv(output / "improved_validation.csv",
              index=False, float_format="%.6f")

    def subset_stats(mask: np.ndarray, group_name: str) -> dict:
        sub = test.loc[mask]
        pred = prediction[mask]
        e = sub["y"].to_numpy(dtype=float) - pred
        return {
            "group": group_name, "n": len(sub),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "mean_residual": float(np.mean(e)),
        }

    groups = []
    for season in ("Winter", "Spring", "Summer", "Autumn"):
        groups.append(subset_stats(test["Seasons"].eq(season).to_numpy(),
                                  season))
    rain = test["Rainfall(mm)"].to_numpy(dtype=float)
    groups.append(subset_stats(rain > 0, "rainfall_gt_0"))
    groups.append(subset_stats(rain == 0, "rainfall_eq_0"))
    pd.DataFrame(groups).to_csv(output / "improved_group_errors.csv",
                                index=False, float_format="%.6f")

    print("M4 validation lambda selection:", [(r["lam"], round(r["rmse"], 4))
                                               for r in candidates])
    print("Selected lambda:", best["lam"])
    print("Test MAE / RMSE / R2:",
          round(metrics["mae"], 6),
          round(metrics["rmse"], 6),
          round(metrics["r2"], 6))
    for term, value in zip(MODEL_ORDER, beta):
        print(f"{term:>28s}: {value: .9f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/SeoulBikeData.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    options = parser.parse_args()
    execute(options.data, options.out)
