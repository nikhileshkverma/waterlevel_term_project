from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import BASE_DIR

# -------------------------------------------------------------------
# metrics from train_multihorizon.py run
#
# Horizons are in HOURS: 1h, 6h, 12h
# -------------------------------------------------------------------
METRICS = {
    1: {  # 1-hour ahead
        "Baseline":     {"rmse": 0.0031, "mae": 0.0020, "r2": 0.9993},
        "Linear SVR":   {"rmse": 0.0031, "mae": 0.0021, "r2": 0.9994},
        "Random Forest": {"rmse": 0.0028, "mae": 0.0018, "r2": 0.9995},
    },
    6: {  # 6-hour ahead
        "Baseline":     {"rmse": 0.0086, "mae": 0.0066, "r2": 0.9949},
        "Linear SVR":   {"rmse": 0.0117, "mae": 0.0092, "r2": 0.9905},
        "Random Forest": {"rmse": 0.0060, "mae": 0.0040, "r2": 0.9975},
    },
    12: {  # 12-hour ahead
        "Baseline":     {"rmse": 0.0152, "mae": 0.0120, "r2": 0.9842},
        "Linear SVR":   {"rmse": 0.0108, "mae": 0.0081, "r2": 0.9920},
        "Random Forest": {"rmse": 0.0092, "mae": 0.0063, "r2": 0.9942},
    },
}

FIG_MULTI_DIR = BASE_DIR / "figures" / "multihorizon"
FIG_MULTI_DIR.mkdir(parents=True, exist_ok=True)


def build_metrics_df() -> pd.DataFrame:
    """Convert METRICS dict -> tidy DataFrame."""
    rows = []
    for horizon_h, models in METRICS.items():
        for model_name, m in models.items():
            rows.append(
                {
                    "horizon_h": horizon_h,
                    "model": model_name,
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["horizon_h", "model"])
    return df


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str):
    """
    Generic plot: metric vs horizon for each model.
    - metric: "rmse", "mae", or "r2"
    """
    pivot = df.pivot(index="horizon_h", columns="model", values=metric)
    horizons = pivot.index.values

    plt.figure(figsize=(7, 4))
    for model in pivot.columns:
        plt.plot(
            horizons,
            pivot[model].values,
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel(ylabel)
    plt.xticks(horizons, [str(h) for h in horizons])
    plt.title(f"{metric.upper()} vs forecast horizon")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_MULTI_DIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    df = build_metrics_df()
    print("\nMulti-horizon metrics table:")
    print(df)

    # 1) RMSE vs horizon
    plot_metric(df, metric="rmse", ylabel="RMSE (m)",
                filename="multihorizon_rmse.png")

    # 2) MAE vs horizon
    plot_metric(df, metric="mae", ylabel="MAE (m)",
                filename="multihorizon_mae.png")

    # 3) R^2 vs horizon
    plot_metric(df, metric="r2", ylabel="R²",
                filename="multihorizon_r2.png")

    print("\nAll multi-horizon plots saved in 'figures/multihorizon/'.\n")


if __name__ == "__main__":
    main()
