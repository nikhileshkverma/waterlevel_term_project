"""
Analyze water-level nowcasting results and generate figures.

Run from the project root:

    python src/analyze_results.py

Outputs:
    - Prints summary statistics and metrics to the console.
    - Saves PNG figures under: figures/ subfolders:
        - figures/eda/
        - figures/features/
        - figures/interpretability/
        - figures/models/
        - figures/errors/
        - figures/robustness/
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import PROCESSED_DIR, MODELS_DIR

# --------------------------------------------------------------------
# Global paths
# --------------------------------------------------------------------
FIG_DIR = Path("figures")
EDA_DIR = FIG_DIR / "eda"
ERROR_DIR = FIG_DIR / "errors"
FEATURE_DIR = FIG_DIR / "features"
INTERP_DIR = FIG_DIR / "interpretability"
MODELS_FIG_DIR = FIG_DIR / "models"
ROBUST_DIR = FIG_DIR / "robustness"

for d in [FIG_DIR, EDA_DIR, ERROR_DIR, FEATURE_DIR, INTERP_DIR, MODELS_FIG_DIR, ROBUST_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# Helper: metrics
# --------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------
def load_data():
    dataset_path = PROCESSED_DIR / "dataset.parquet"
    preds_path = PROCESSED_DIR / "test_predictions.csv"

    df_all = pd.read_parquet(dataset_path)
    df_preds = pd.read_csv(preds_path, parse_dates=["datetime"])

    return df_all, df_preds


# --------------------------------------------------------------------
# 1. Basic data description
# --------------------------------------------------------------------
def analyze_basic_stats(df_all: pd.DataFrame):
    print("\n=== BASIC DATA DESCRIPTION ===")
    dt_min = df_all["datetime"].min()
    dt_max = df_all["datetime"].max()
    print(f"Datetime range: {dt_min} -> {dt_max}")

    years = df_all["datetime"].dt.year.value_counts().sort_index()
    print("\nSamples per year:")
    print(years)

    stations = ["packery", "bob_hall", "lexington", "port_aransas"]
    desc = df_all[stations].describe().T
    print("\nStation summary statistics:")
    print(desc)


# --------------------------------------------------------------------
# 2. EDA plots
# --------------------------------------------------------------------
def plot_station_timeseries(df_all: pd.DataFrame):
    """
    Plot a 7-day window of all 4 stations.
    """
    print("\nGenerating station time-series plot (EDA)...")

    # Data ends in 2022-04-14, so use early March 2022
    start_date = "2022-03-01"
    end_date = "2022-03-08"

    mask = (df_all["datetime"] >= start_date) & (df_all["datetime"] < end_date)
    sub = df_all.loc[mask, ["datetime", "packery", "bob_hall", "lexington", "port_aransas"]]

    plt.figure(figsize=(10, 5))
    for col in ["packery", "bob_hall", "lexington", "port_aransas"]:
        plt.plot(sub["datetime"], sub[col], label=col)

    plt.xlabel("Time")
    plt.ylabel("Water level (m)")
    plt.title(f"Hourly water levels at 4 stations ({start_date} to {end_date})")
    plt.legend()
    plt.tight_layout()
    out_path = EDA_DIR / "eda_stations_timeseries_2022-03-01_to_2022-03-08.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_packery_histogram(df_all: pd.DataFrame):
    print("\nGenerating Packery histogram (EDA)...")

    series = df_all["packery"].dropna()

    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=50)
    plt.xlabel("Packery water level (m)")
    plt.ylabel("Count")
    plt.title("Distribution of Packery water level (all years)")
    plt.tight_layout()
    out_path = EDA_DIR / "eda_packery_histogram.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_packery_boxplot_by_month(df_all: pd.DataFrame):
    print("\nGenerating Packery boxplot by month (EDA)...")

    df = df_all.copy()
    df["month"] = df["datetime"].dt.month
    data = [df.loc[df["month"] == m, "packery"].dropna() for m in range(1, 13)]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, positions=range(1, 13), showfliers=False)
    plt.xlabel("Month")
    plt.ylabel("Packery water level (m)")
    plt.title("Monthly distribution of Packery water level")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    out_path = EDA_DIR / "eda_packery_boxplot_by_month.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_packery_diurnal_cycle(df_all: pd.DataFrame):
    print("\nGenerating diurnal cycle plot for Packery (EDA)...")

    df = df_all.copy()
    df["hour"] = df["datetime"].dt.hour

    hourly_mean = df.groupby("hour")["packery"].mean().reset_index()

    plt.figure(figsize=(8, 4))
    plt.plot(hourly_mean["hour"], hourly_mean["packery"], marker="o")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean Packery water level (m)")
    plt.title("Average diurnal cycle of Packery water level")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    out_path = EDA_DIR / "eda_packery_diurnal_cycle.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_packery_annual_cycle(df_all: pd.DataFrame):
    print("\nGenerating annual cycle plot for Packery (EDA)...")

    df = df_all.copy()
    df["dayofyear"] = df["datetime"].dt.dayofyear

    daily_mean = df.groupby("dayofyear")["packery"].mean().reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(daily_mean["dayofyear"], daily_mean["packery"])
    plt.xlabel("Day of year")
    plt.ylabel("Mean Packery water level (m)")
    plt.title("Average annual cycle of Packery water level")
    plt.tight_layout()
    out_path = EDA_DIR / "eda_packery_annual_cycle.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_station_correlation(df_all: pd.DataFrame):
    print("\nGenerating station correlation heatmap (EDA)...")
    stations = ["packery", "bob_hall", "lexington", "port_aransas"]
    corr = df_all[stations].corr()

    print("\nCorrelation matrix:")
    print(corr)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(stations)), stations, rotation=45, ha="right")
    plt.yticks(range(len(stations)), stations)
    plt.title("Correlation between station water levels")
    plt.tight_layout()
    out_path = EDA_DIR / "eda_stations_correlation_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 3. Feature analysis (lag correlation)
# --------------------------------------------------------------------
def plot_lag_correlation(df_all: pd.DataFrame):
    print("\nGenerating Packery lag correlation plot (features)...")

    # Use existing lag columns if they exist
    lag_cols = [c for c in df_all.columns if c.startswith("packery_lag")]
    if lag_cols:
        def lag_num(c):
            try:
                return int(c.replace("packery_lag", ""))
            except ValueError:
                return 9999

        lag_cols = sorted(lag_cols, key=lag_num)
        lags = [lag_num(c) for c in lag_cols]
        corrs = [df_all["packery"].corr(df_all[c]) for c in lag_cols]
    else:
        # fallback: compute directly using shift
        max_lag = 24
        series = df_all["packery"]
        lags = list(range(1, max_lag + 1))
        corrs = []
        for lag in lags:
            corrs.append(series.corr(series.shift(lag)))

    # Remove NaNs to be safe
    valid = [(lag, corr) for lag, corr in zip(lags, corrs) if pd.notna(corr)]
    if not valid:
        print("No valid correlations found for lag plot.")
        return

    lags_clean, corrs_clean = zip(*valid)

    plt.figure(figsize=(8, 4))
    # no use_line_collection (not supported in newer Matplotlib)
    plt.stem(lags_clean, corrs_clean)
    plt.xlabel("Lag (hours)")
    plt.ylabel("Correlation with Packery(t)")
    plt.title("Correlation between Packery(t) and Packery(t - lag)")
    plt.tight_layout()
    out_path = FEATURE_DIR / "feat_packery_lag_correlation.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 4. Global model metrics + comparison bar chart
# --------------------------------------------------------------------
def analyze_model_metrics(df_preds: pd.DataFrame):
    print("\n=== MODEL METRICS ON TEST SET ===")

    needed_cols = ["y_true", "y_pred_baseline", "y_pred_svr", "y_pred_rf"]
    df = df_preds.dropna(subset=needed_cols).copy()

    if df.empty:
        print("No valid rows after dropping NaNs in predictions. Check test_predictions.csv.")
        return

    y_true = df["y_true"].values

    # Baseline
    rmse_b, mae_b, r2_b = regression_metrics(y_true, df["y_pred_baseline"].values)
    print(f"[Baseline]      RMSE={rmse_b:.4f}, MAE={mae_b:.4f}, R^2={r2_b:.4f}")

    # SVR
    rmse_svr, mae_svr, r2_svr = regression_metrics(y_true, df["y_pred_svr"].values)
    print(f"[SVR (RBF)]     RMSE={rmse_svr:.4f}, MAE={mae_svr:.4f}, R^2={r2_svr:.4f}")

    # Random Forest
    rmse_rf, mae_rf, r2_rf = regression_metrics(y_true, df["y_pred_rf"].values)
    print(f"[Random Forest] RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}, R^2={r2_rf:.4f}")

    models = ["Baseline", "SVR", "RF"]
    rmse_vals = [rmse_b, rmse_svr, rmse_rf]
    mae_vals = [mae_b, mae_svr, mae_rf]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, rmse_vals, width, label="RMSE")
    plt.bar(x + width / 2, mae_vals, width, label="MAE")
    plt.xticks(x, models)
    plt.ylabel("Error (m)")
    plt.title("Test-set RMSE and MAE by model")
    plt.legend()
    plt.tight_layout()
    out_path = MODELS_FIG_DIR / "models_error_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 4b. RF true vs predicted scatter
# --------------------------------------------------------------------
def plot_rf_true_vs_pred_scatter(df_preds: pd.DataFrame):
    print("\nGenerating RF true vs predicted scatter plot (models)...")

    df = df_preds.dropna(subset=["y_true", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for RF scatter plot (after dropping NaNs).")
        return

    y_true = df["y_true"].values
    y_pred = df["y_pred_rf"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=2, alpha=0.3)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("True Packery water level (m)")
    plt.ylabel("Predicted Packery water level (m)")
    plt.title("Random Forest: true vs predicted (test set)")
    plt.tight_layout()
    out_path = MODELS_FIG_DIR / "models_rf_true_vs_pred_scatter.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 5. True vs Predicted (RF vs Baseline) for a short window
# --------------------------------------------------------------------
def plot_true_vs_pred(df_preds: pd.DataFrame):
    print("\nGenerating true vs predicted plot (RF vs baseline, window)...")

    df = df_preds.dropna(subset=["y_true", "y_pred_baseline", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for true vs pred plot (after dropping NaNs).")
        return

    start_date = df["datetime"].min() + pd.Timedelta(days=10)
    end_date = start_date + pd.Timedelta(days=4)

    mask = (df["datetime"] >= start_date) & (df["datetime"] < end_date)
    sub = df.loc[mask].copy()

    if sub.empty:
        print("Chosen window for true vs pred plot is empty. Adjust the dates if needed.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(sub["datetime"], sub["y_true"], label="True", linewidth=2)
    plt.plot(sub["datetime"], sub["y_pred_baseline"], label="Baseline", alpha=0.7)
    plt.plot(sub["datetime"], sub["y_pred_rf"], label="Random Forest", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Water level (m)")
    plt.title(f"True vs predicted Packery water level ({start_date.date()} to {end_date.date()})")
    plt.legend()
    plt.tight_layout()
    out_path = MODELS_FIG_DIR / "models_true_vs_pred_rf_baseline_window.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 6. Residual analysis for Random Forest
# --------------------------------------------------------------------
def plot_residuals(df_preds: pd.DataFrame):
    print("\nGenerating residual plots for Random Forest (errors)...")

    df = df_preds.dropna(subset=["y_true", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for residual plots (after dropping NaNs).")
        return

    df["residual_rf"] = df["y_true"] - df["y_pred_rf"]

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df["residual_rf"], bins=50)
    plt.xlabel("Residual (m)")
    plt.ylabel("Count")
    plt.title("Distribution of RF residuals on test set")
    plt.tight_layout()
    out_path = ERROR_DIR / "errors_rf_residual_histogram.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    # Over time
    plt.figure(figsize=(10, 4))
    plt.plot(df["datetime"], df["residual_rf"], linewidth=0.5)
    plt.xlabel("Time")
    plt.ylabel("Residual (m)")
    plt.title("Random Forest residuals over test period")
    plt.tight_layout()
    out_path = ERROR_DIR / "errors_rf_residuals_over_time.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 6b. Error vs true water level (binned)
# --------------------------------------------------------------------
def plot_error_vs_level_binned(df_preds: pd.DataFrame, n_bins: int = 10):
    print("\nGenerating RF error vs true level (binned) plot (errors)...")

    df = df_preds.dropna(subset=["y_true", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for error-vs-level plot (after dropping NaNs).")
        return

    df["residual_rf"] = df["y_true"] - df["y_pred_rf"]

    df["level_bin"] = pd.qcut(df["y_true"], q=n_bins, duplicates="drop")

    stats = (
        df.groupby("level_bin", observed=False)["residual_rf"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    rmse_per_bin = []
    centers = []
    for _, row in stats.iterrows():
        bin_mask = df["level_bin"] == row["level_bin"]
        residuals = df.loc[bin_mask, "residual_rf"].values
        rmse = math.sqrt(np.mean(residuals ** 2))
        rmse_per_bin.append(rmse)
        centers.append(df.loc[bin_mask, "y_true"].mean())

    plt.figure(figsize=(8, 4))
    plt.bar(centers, rmse_per_bin, width=0.02)
    plt.xlabel("True water level (m) (bin centers)")
    plt.ylabel("RMSE (m)")
    plt.title("Random Forest RMSE vs water level (binned)")
    plt.tight_layout()
    out_path = ERROR_DIR / "errors_rf_rmse_vs_true_level_binned.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 7. Per-year performance for RF (robustness)
# --------------------------------------------------------------------
def analyze_rf_per_year(df_preds: pd.DataFrame):
    print("\n=== RF PERFORMANCE PER YEAR (TEST SET) ===")

    df = df_preds.dropna(subset=["y_true", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for per-year RF metrics (after dropping NaNs).")
        return

    df["year"] = df["datetime"].dt.year

    years = []
    rmses = []

    for year, group in df.groupby("year"):
        y_true = group["y_true"].values
        y_pred = group["y_pred_rf"].values
        rmse, mae, r2 = regression_metrics(y_true, y_pred)
        print(f"{year}: RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")
        years.append(year)
        rmses.append(rmse)

    plt.figure(figsize=(8, 4))
    plt.bar(years, rmses)
    plt.xlabel("Year")
    plt.ylabel("RMSE (m)")
    plt.title("Random Forest RMSE per year (test set)")
    plt.tight_layout()
    out_path = ROBUST_DIR / "robust_rf_rmse_per_year.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# 7b. Threshold accuracy: within 5 / 10 / 15 cm
# --------------------------------------------------------------------
def summarize_threshold_accuracy(df_preds: pd.DataFrame):
    """
    Report how often RF predictions fall within certain absolute error
    thresholds (in meters), e.g., 0.15 m = 15 cm.
    """
    print("\n=== THRESHOLD ACCURACY (RF) ===")

    df = df_preds.dropna(subset=["y_true", "y_pred_rf"]).copy()
    if df.empty:
        print("No valid rows for threshold accuracy (after dropping NaNs).")
        return

    df["abs_err_rf"] = (df["y_true"] - df["y_pred_rf"]).abs()

    # thresholds in meters: 5 cm, 10 cm, 15 cm
    thresholds = [0.05, 0.10, 0.15]

    for thr in thresholds:
        frac = (df["abs_err_rf"] <= thr).mean()
        print(
            f"Fraction of RF predictions within ±{thr:.2f} m "
            f"({thr * 100:.0f} cm): {frac * 100:.2f}%"
        )


# --------------------------------------------------------------------
# 8. Random Forest feature importance (interpretability)
# --------------------------------------------------------------------
def plot_rf_feature_importance(df_all: pd.DataFrame):
    print("\nGenerating Random Forest feature importance plot (interpretability)...")

    target_col = "packery"
    feature_cols = [c for c in df_all.columns if c not in ["datetime", target_col]]

    rf_path = MODELS_DIR / "rf_model.joblib"
    rf = joblib.load(rf_path)

    importances = rf.feature_importances_
    if len(importances) != len(feature_cols):
        print("Warning: feature_importances_ length does not match feature_cols length.")
        return

    idx = np.argsort(importances)[::-1]
    top_n = 20
    top_idx = idx[:top_n]
    top_features = [feature_cols[i] for i in top_idx]
    top_importances = importances[top_idx]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances)
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Random Forest top 20 feature importances")
    plt.tight_layout()
    out_path = INTERP_DIR / "interp_rf_feature_importance_top20.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    df_all, df_preds = load_data()

    # 1. Basic stats
    analyze_basic_stats(df_all)

    # 2. EDA
    plot_station_timeseries(df_all)
    plot_packery_histogram(df_all)
    plot_packery_boxplot_by_month(df_all)
    plot_packery_diurnal_cycle(df_all)
    plot_packery_annual_cycle(df_all)
    plot_station_correlation(df_all)

    # 3. Feature analysis
    plot_lag_correlation(df_all)

    # 4. Global model metrics
    analyze_model_metrics(df_preds)

    # 4b. RF true vs predicted scatter
    plot_rf_true_vs_pred_scatter(df_preds)

    # 5. True vs predicted (window)
    plot_true_vs_pred(df_preds)

    # 6. Residual analysis
    plot_residuals(df_preds)

    # 6b. Error vs level (binned)
    plot_error_vs_level_binned(df_preds)

    # 7. RF performance per year
    analyze_rf_per_year(df_preds)

    # 7b. Threshold accuracy (within 5/10/15 cm)
    summarize_threshold_accuracy(df_preds)

    # 8. RF feature importance
    plot_rf_feature_importance(df_all)

    print("\nAll analyses complete. Check the 'figures/' folder (and subfolders) for PNG files.\n")


if __name__ == "__main__":
    main()
