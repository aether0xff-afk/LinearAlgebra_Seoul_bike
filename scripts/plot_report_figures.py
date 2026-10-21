"""Generate submission-quality Matplotlib charts from actual UCI CSV and fitted models.

Run from repository root:
    python -m scripts.plot_report_figures

All points/metrics are recomputed from source data. No image-generation AI
or hand-drawn/smoothed demand values are used. Output PNG, SVG and vector PDF.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.run_analysis import read_data, assign_split, fit_models, predict, score
from src.improved_model import design_improved, fit, LAMBDAS

DATA = Path("data/SeoulBikeData.csv")
DEST = Path("results/figures")
COLORS = {
    "actual": "#173C60",
    "m1": "#B87538",
    "m2": "#2F75AB",
    "m3": "#16807A",
    "m4": "#6556A3",
}
NAMES = ["mean", "harmonic_24h", "harmonic_24h_12h", "harmonic_weekend"]
LABELS = ["평균", "M1 · 24시간", "M2 · 24+12시간", "M3 · 주중/주말", "M4 · 개선"]


def set_fonts() -> None:
    candidates = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic",
                  "AppleGothic", "Noto Sans CJK KR")
    available = {f.name for f in font_manager.fontManager.ttflist}
    selection = next((font for font in candidates if font in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": selection,
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 11.4,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",  # legible even without Korean fonts on viewer
    })
    if selection == "DejaVu Sans":
        print("WARNING: no Korean font found; install fonts-nanum for charts.")


def style(ax, ymax=1900) -> None:
    ax.set_xlim(-0.4, 23.4)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("시각 (시)", labelpad=3)
    ax.set_ylabel("대여 건수/시간", labelpad=4)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B97A4")
    ax.spines["bottom"].set_color("#8B97A4")


def save(fig, stem) -> None:
    DEST.mkdir(exist_ok=True, parents=True)
    for ext in ("pdf", "svg", "png"):
        options = {"dpi": 230} if ext == "png" else {}
        fig.savefig(DEST / f"{stem}.{ext}", **options)
    plt.close(fig)


def score_improved(train, validation, test):
    X_train = design_improved(train)
    y_train = train["y"].to_numpy(dtype=float)
    X_val = design_improved(validation)
    y_val = validation["y"].to_numpy(dtype=float)
    X_test = design_improved(test)
    y_test = test["y"].to_numpy(dtype=float)
    fits = [(lam, fit(X_train, y_train, lam)) for lam in LAMBDAS]
    chosen = min(fits, key=lambda item: score(y_val, X_val @ item[1])["rmse"])
    return chosen[0], score(y_test, X_test @ chosen[1]), chosen[1]


def generate() -> None:
    set_fonts()
    df, audit = read_data(DATA)
    df = assign_split(df, seed=42)
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    fitted = fit_models(train)

    # 1: actually observed average of all operating hourly observations.
    hours = np.arange(24)
    observed = df.groupby("hour")["y"].mean().reindex(hours)
    assert observed.notna().all()
    fig, ax = plt.subplots(figsize=(8.6, 2.65))
    ax.plot(hours, observed, color=COLORS["actual"], marker="o",
            linewidth=2.2, markersize=3.6)
    style(ax, ymax=1850)
    ax.set_title("정상 운영 기록의 시간대별 평균 대여량", weight="bold", pad=9)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.84, bottom=0.24)
    save(fig, "report_hourly_mean")

    # 2: held-out test set hourly mean vs learned Fourier functions. M3 is
    # evaluated per record with that record's actual weekend flag, then
    # averaged at each clock hour; do not assume a fixed weekend ratio.
    test = test.copy()
    for name in NAMES[1:]:
        test[name] = predict(test, fitted, name)
    grouped = test.groupby("hour")
    actual_test = grouped["y"].mean().reindex(hours)
    fig, ax = plt.subplots(figsize=(8.6, 2.85))
    ax.plot(hours, actual_test, label="테스트 실측 평균",
            color=COLORS["actual"], marker="o", markersize=3.3, linewidth=2)
    for name, label, color, dash in (
        ("harmonic_24h", "M1 · 24시간", COLORS["m1"], "--"),
        ("harmonic_24h_12h", "M2 · 24+12시간", COLORS["m2"], "-"),
        ("harmonic_weekend", "M3 · 주중/주말", COLORS["m3"], "-"),
    ):
        values = grouped[name].mean().reindex(hours)
        ax.plot(hours, values, label=label, linestyle=dash, color=color,
                linewidth=1.8)
    style(ax, ymax=1900)
    fig.suptitle("테스트 63일 · 시간대별 실측 평균과 예측",
                 weight="bold", y=0.985)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.52, 0.88), ncol=4,
               frameon=False, fontsize=8.9, handlelength=2.3)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.72, bottom=0.26)
    save(fig, "report_test_profile")

    # 3: held-out test error bars. M4 is fitted using train data, lambda
    # selected ONLY on validation data.
    metrics = [score(test["y"].to_numpy(), predict(test, fitted, name))
               for name in NAMES]
    chosen_lambda, m4_score, _ = score_improved(train, validation, test)
    metrics.append(m4_score)
    assert len(test) == 1512
    assert chosen_lambda == 0
    assert abs(metrics[-1]["mae"] - 280.575010) < 0.01
    x = np.arange(5)
    width = 0.31
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    a = ax.bar(x - width/2, [m["mae"] for m in metrics], width,
               color=COLORS["m2"], label="MAE")
    b = ax.bar(x + width/2, [m["rmse"] for m in metrics], width,
               color=COLORS["m3"], label="RMSE")
    for bars in (a, b):
        ax.bar_label(bars, fmt="%.1f", fontsize=8.0, padding=3)
    ax.set_xticks(x, ["평균", "M1\n24시간", "M2\n24+12시간",
                      "M3\n주중/주말", "M4\n개선"])
    ax.set_ylim(0, 725)
    ax.set_ylabel("대여 건수/시간")
    ax.set_title("미학습 테스트 1,512건 · 모델별 오차 (낮을수록 우수)",
                 weight="bold", pad=9)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.7)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.84, bottom=0.23)
    save(fig, "report_model_errors")

    # 4: the part of the squared error due to systematic bias by season.
    test["m3_residual"] = (test["y"].to_numpy() -
                            predict(test, fitted, "harmonic_weekend"))
    seasonal = test.groupby("Seasons")["m3_residual"].agg(["mean", "size"])
    seasonal = seasonal.reindex(["Winter", "Spring", "Summer", "Autumn"])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(seasonal.index, seasonal["mean"],
           color=[COLORS["m2"] if x >= 0 else COLORS["m1"]
                  for x in seasonal["mean"]])
    ax.axhline(0, color="#586B7B", lw=0.8)
    ax.set_ylabel("평균 잔차 (실제 − 예측)")
    ax.set_title("M3의 계절별 평균 잔차 · 테스트 데이터",
                 weight="bold", pad=8)
    ax.grid(axis="y", color="#DCE3E8", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.23)
    save(fig, "report_season_bias")

    pd.DataFrame([{"model": label, "test_mae": m["mae"],
                   "test_rmse": m["rmse"], "test_r2": m["r2"]}
                  for label, m in zip(LABELS, metrics)]).to_csv(
        DEST / "report_model_scores.csv", index=False,
        float_format="%.6f")
    print("Created four Matplotlib figures from", audit["functioning_rows"],
          "operating-hour observations. Test n =", len(test))


if __name__ == "__main__":
    generate()
