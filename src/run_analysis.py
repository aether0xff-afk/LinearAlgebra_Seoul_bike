"""Reproducible Fourier regression of Seoul hourly bike rentals.

Usage: python -m src.run_analysis --data data/SeoulBikeData.csv --out results

The task is estimating demand for unseen DAYS in the same observed year,
not forecasting a subsequent year or real-time bike availability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
MODEL_NAMES = ("mean", "harmonic_24h", "harmonic_24h_12h", "harmonic_weekend")
COEFFICIENT_NAMES = {
    "harmonic_24h": ["a0", "a1", "b1"],
    "harmonic_24h_12h": ["a0", "a1", "b1", "a2", "b2"],
    "harmonic_weekend": ["a0", "a1", "b1", "a2", "b2",
                         "c0", "c1", "d1", "c2", "d2"],
}


def read_data(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Date", "Hour", "Rented Bike Count", "Functioning Day",
                "Seasons", "Holiday", "Rainfall(mm)"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Missing values in required columns")
    raw["date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y")
    raw["hour"] = pd.to_numeric(raw["Hour"], errors="raise").astype(int)
    raw["y"] = pd.to_numeric(raw["Rented Bike Count"], errors="raise")
    if raw["hour"].lt(0).any() or raw["hour"].gt(23).any():
        raise ValueError("Hour must be between 0 and 23")
    if raw["y"].lt(0).any():
        raise ValueError("Rental counts must be nonnegative")
    if raw.duplicated(["date", "hour"]).any():
        raise ValueError("Duplicated date/hour records")
    if not raw["Functioning Day"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected Functioning Day values")
    audit = {
        "original_rows": int(len(raw)),
        "non_functioning_rows": int((raw["Functioning Day"] == "No").sum()),
        "date_count": int(raw["date"].nunique()),
    }
    df = raw.loc[raw["Functioning Day"] == "Yes"].copy()
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["season"] = df["Seasons"].astype(str)
    df["date_key"] = df["date"].dt.strftime("%Y-%m-%d")
    audit["functioning_rows"] = int(len(df))
    if not len(df):
        raise ValueError("No usable records")
    return df, audit


def stable_hash(value: str) -> int:
    """32-bit FNV-1a for a reproducible date-level split across environments."""
    h = 2166136261
    for byte in value.encode("utf-8"):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def assign_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratify by season and weekend, but keep every date in exactly one split."""
    daily = df[["date_key", "season", "weekend"]].drop_duplicates()
    if daily["date_key"].duplicated().any():
        raise ValueError("A date has inconsistent season/weekend labels")
    lookup = {}
    for _, block in daily.groupby(["season", "weekend"]):
        keys = sorted(block["date_key"].tolist(),
                      key=lambda d: (stable_hash(f"{d}|{seed}"), d))
        n = len(keys)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        for i, date in enumerate(keys):
            lookup[date] = ("train" if i < n_train else
                            "validation" if i < n_train + n_val else "test")
    df = df.copy()
    df["split"] = df["date_key"].map(lookup)
    if df["split"].isna().any():
        raise ValueError("Split assignment failed")
    if set(df["split"]) != {"train", "validation", "test"}:
        raise ValueError("Need all three nonempty partitions")
    return df


def design_matrix(hours: np.ndarray, weekend: np.ndarray, name: str) -> np.ndarray:
    t = np.asarray(hours, dtype=float)
    w = np.asarray(weekend, dtype=float)
    if t.shape != w.shape:
        raise ValueError("hours and weekend must have the same shape")
    p = 2.0 * np.pi * t / 24.0
    ones = np.ones_like(t)
    first = np.column_stack((ones, np.sin(p), np.cos(p)))
    if name == "harmonic_24h":
        return first
    second = np.column_stack((first, np.sin(2 * p), np.cos(2 * p)))
    if name == "harmonic_24h_12h":
        return second
    if name == "harmonic_weekend":
        return np.column_stack((second, second * w[:, None]))
    raise ValueError(f"Unknown regression model: {name}")


def fit_models(train: pd.DataFrame) -> dict:
    result = {"mean": float(train["y"].mean())}
    for name in MODEL_NAMES[1:]:
        X = design_matrix(train["hour"].to_numpy(),
                          train["weekend"].to_numpy(), name)
        beta, _, rank, _ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
        if rank != X.shape[1]:
            raise ValueError(f"Rank deficiency for model {name}")
        result[name] = beta
    return result


def predict(df: pd.DataFrame, fitted: dict, name: str) -> np.ndarray:
    if name == "mean":
        return np.full(len(df), fitted["mean"], dtype=float)
    X = design_matrix(df["hour"].to_numpy(), df["weekend"].to_numpy(), name)
    return X @ fitted[name]


def score(y: np.ndarray, estimate: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    residual = y - np.asarray(estimate, dtype=float)
    mse = float(np.mean(residual**2))
    denom = float(np.sum((y - y.mean())**2))
    r2 = (1.0 - float(np.sum(residual**2)) / denom) if denom else float("nan")
    return {"mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(mse)), "r2": r2}


def make_plots(df: pd.DataFrame, metrics: pd.DataFrame, fitted: dict,
               output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    hours = np.arange(24)
    real = df.groupby("hour")["y"].mean().reindex(hours)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(hours, real, marker="o", linewidth=1.8, label="Actual hourly mean")
    ax.set(xlabel="Hour of day", ylabel="Rentals per hour",
           title="Seoul bike rentals: hourly mean (functioning hours)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "hourly_mean.png", dpi=180)
    plt.close(fig)

    test = df.loc[df["split"] == "test"].copy()
    for w, label in ((0, "weekday"), (1, "weekend")):
        subset = test[test["weekend"] == w]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        observed = subset.groupby("hour")["y"].mean().reindex(hours)
        ax.plot(hours, observed, "ko-", label="Test observed mean")
        grid = pd.DataFrame({"hour": hours, "weekend": w})
        for name in MODEL_NAMES[1:]:
            ax.plot(hours, predict(grid, fitted, name), label=name)
        ax.set(xlabel="Hour of day", ylabel="Rentals per hour",
               title=f"Test-day profile: {label}")
        ax.set_xticks(np.arange(0, 24, 2))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"model_fit_{label}.png", dpi=180)
        plt.close(fig)

    test_scores = metrics[metrics["split"] == "test"].set_index("model")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(MODEL_NAMES)), [test_scores.loc[n, "mae"]
                                    for n in MODEL_NAMES])
    ax.set(xlabel="Model", ylabel="Test MAE (rentals / hour)",
           title="Unseen-date generalization error")
    ax.set_xticks(range(len(MODEL_NAMES)), ["mean", "24h", "24h+12h", "+weekend"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mae_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in MODEL_NAMES[1:]:
        res = test.assign(residual=test["y"] - predict(test, fitted, name))
        hourly = res.groupby("hour")["residual"].mean().reindex(hours)
        ax.plot(hours, hourly, marker=".", label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Hour of day", ylabel="Mean residual (observed - predicted)",
           title="Test residuals by hour")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "residual_by_hour.png", dpi=180)
    plt.close(fig)


def run(data: Path, out: Path, seed: int = SEED) -> None:
    df, audit = read_data(data)
    df = assign_split(df, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    train = df[df["split"] == "train"]
    fitted = fit_models(train)

    metrics_rows = []
    predictions = df[["date_key", "hour", "weekend", "season",
                      "Holiday", "Rainfall(mm)", "y", "split"]].copy()
    for name in MODEL_NAMES:
        predictions[name] = predict(df, fitted, name)
        for partition in ("train", "validation", "test"):
            mask = df["split"] == partition
            result = score(df.loc[mask, "y"].to_numpy(),
                           predictions.loc[mask, name].to_numpy())
            metrics_rows.append({"model": name, "split": partition,
                                 "n": int(mask.sum()), **result})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.6f")

    coefficients = []
    coefficients.append({"model": "mean", "term": "train_mean",
                         "coefficient": fitted["mean"]})
    for name in MODEL_NAMES[1:]:
        for term, value in zip(COEFFICIENT_NAMES[name], fitted[name]):
            coefficients.append({"model": name, "term": term,
                                 "coefficient": float(value)})
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv",
                                      index=False, float_format="%.8f")
    df.groupby("hour").agg(
        n=("y", "size"), mean_rentals=("y", "mean")
    ).reset_index().to_csv(out / "hourly_means.csv", index=False,
                            float_format="%.4f")

    test = predictions[predictions["split"] == "test"]
    residual_rows = []
    for name in MODEL_NAMES:
        tmp = test.assign(residual=test["y"] - test[name])
        summary = tmp.groupby("hour")["residual"].agg(["mean", "count"])
        for hour, row in summary.iterrows():
            residual_rows.append({"model": name, "hour": hour,
                                  "mean_residual": row["mean"],
                                  "n": int(row["count"])})
    pd.DataFrame(residual_rows).to_csv(
        out / "test_residuals_by_hour.csv", index=False, float_format="%.6f")
    make_plots(df, metrics, fitted, out / "figures")

    test_m = metrics[metrics["split"] == "test"].set_index("model")
    before = test_m.loc["harmonic_24h", "mae"]
    after = test_m.loc["harmonic_24h_12h", "mae"]
    improvement = 100 * (before - after) / before
    lines = [
        "# Reproducible analysis summary",
        "",
        "Data: UCI Seoul Bike Sharing Demand (CC BY 4.0), 2017-12-01 to 2018-11-30.",
        f"Original rows: {audit['original_rows']}",
        f"Non-functioning rows removed: {audit['non_functioning_rows']}",
        f"Analyzed rows: {audit['functioning_rows']}",
        "",
        "Date-level, stratified, deterministic split; season and weekend strata.",
        f"Seed: {seed}; no single date appears in multiple partitions.",
        "",
        "| Model | Test MAE | Test RMSE | Test R² |",
        "|---|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        row = test_m.loc[name]
        lines.append(f"| {name} | {row['mae']:.3f} | "
                     f"{row['rmse']:.3f} | {row['r2']:.3f} |")
    lines.extend([
        "",
        f"MAE reduction from 24h-only to 24h+12h: {improvement:.2f}%.",
        "A negative percentage indicates the richer model performed worse.",
        "",
        "Interpretation: errors refer to held-out dates WITHIN the source year.",
        "No claim is made about forecasting subsequent years or causal effects.",
        "Linear least squares predictions are not forced to be nonnegative.",
        "",
        "See metrics.csv, coefficients.csv, predictions.csv and figures/.",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/SeoulBikeData.csv"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(args.data, args.out, args.seed)


if __name__ == "__main__":
    main()
