"""
Plot multi-horizon (1h, 6h, 12h) performance for Baseline, Linear SVR,
and Random Forest.

Run from project root:

    source .venv/bin/activate
    python src/plot_multihorizon_results.py

Assumptions:
- You have already run:

      python src/train_multihorizon.py

  which created:
      data/processed/multihorizon_metrics.csv

- config.py defines BASE_DIR and PROCESSED_DIR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import BASE_DIR, PROCESSED_DIR

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
METRICS_PATH = PROCESSED_DIR / "multihorizon_metrics.csv"
FIG_DIR = BASE_DIR / "figures" / "multihorizon"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Multi-horizon metrics file not found at:\n  {METRICS_PATH}\n"
            "Run `python src/train_multihorizon.py` first."
        )

    df = pd.read_csv(METRICS_PATH)

    # Expect columns: horizon_h, model, rmse, mae, r2
    required_cols = {"horizon_h", "model", "rmse", "mae", "r2"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Metrics CSV is missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Sort for nicer plots
    df = df.sort_values(["horizon_h", "model"]).reset_index(drop=True)

    print("Multi-horizon metrics table:")
    print(df)

    # Common style: horizon on x-axis, one line per model
    horizons = sorted(df["horizon_h"].unique())
    models = df["model"].unique()

    # 1) RMSE vs horizon
    plt.figure(figsize=(6, 4))
    for m in models:
        sub = df[df["model"] == m]
        plt.plot(sub["horizon_h"], sub["rmse"], marker="o", label=m)
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("RMSE (m)")
    plt.title("Multi-horizon RMSE (1h, 6h, 12h)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_rmse = FIG_DIR / "multihorizon_rmse.png"
    plt.savefig(out_rmse, dpi=200)
    plt.close()
    print(f"Saved {out_rmse}")

    # 2) MAE vs horizon
    plt.figure(figsize=(6, 4))
    for m in models:
        sub = df[df["model"] == m]
        plt.plot(sub["horizon_h"], sub["mae"], marker="o", label=m)
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("MAE (m)")
    plt.title("Multi-horizon MAE (1h, 6h, 12h)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_mae = FIG_DIR / "multihorizon_mae.png"
    plt.savefig(out_mae, dpi=200)
    plt.close()
    print(f"Saved {out_mae}")

    # 3) R^2 vs horizon
    plt.figure(figsize=(6, 4))
    for m in models:
        sub = df[df["model"] == m]
        plt.plot(sub["horizon_h"], sub["r2"], marker="o", label=m)
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("$R^2$")
    plt.title("Multi-horizon $R^2$ (1h, 6h, 12h)")
    plt.ylim(0.97, 1.001)  # zoom near 1 since all are very high
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_r2 = FIG_DIR / "multihorizon_r2.png"
    plt.savefig(out_r2, dpi=200)
    plt.close()
    print(f"Saved {out_r2}")

    print("\nAll multi-horizon plots saved in 'figures/multihorizon/'.\n")


if __name__ == "__main__":
    main()
