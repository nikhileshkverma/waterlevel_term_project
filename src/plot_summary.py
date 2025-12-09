import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR
import numpy as np
import os

# ----------------------------------------------------------
# LOAD METRICS
# ----------------------------------------------------------
csv_path = PROCESSED_DIR / "multihorizon_metrics.csv"
df = pd.read_csv(csv_path)
print("\nLoaded multi-horizon metrics:")
print(df)

models = ["Baseline", "Linear SVR", "Random Forest"]
horizons = sorted(df["horizon_h"].unique())

# Pivot tables
rmse_table = df.pivot(index="model", columns="horizon_h", values="rmse").loc[models]
mae_table  = df.pivot(index="model", columns="horizon_h", values="mae").loc[models]
r2_table   = df.pivot(index="model", columns="horizon_h", values="r2").loc[models]

# ----------------------------------------------------------
# PROFESSIONAL FIGURE LAYOUT
# ----------------------------------------------------------
fig = plt.figure(figsize=(22, 16))
plt.subplots_adjust(hspace=0.55, wspace=0.30)

# -----------------------------
#  Row 1 – Three metric plots
# -----------------------------
ax_rmse = plt.subplot2grid((3, 3), (0, 0))
ax_mae  = plt.subplot2grid((3, 3), (0, 1))
ax_r2   = plt.subplot2grid((3, 3), (0, 2))

plot_info = [
    (ax_rmse, "RMSE vs Forecast Horizon", "RMSE (m)", "rmse"),
    (ax_mae,  "MAE vs Forecast Horizon",  "MAE (m)",  "mae"),
    (ax_r2,   "$R^2$ vs Forecast Horizon", "$R^2$", "r2")
]

for ax, title, ylabel, metric in plot_info:
    for model in models:
        ax.plot(
            horizons,
            df[df.model == model][metric],
            marker="o",
            linewidth=2,
            label=model
        )

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Horizon (hours)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)

ax_rmse.legend(fontsize=11)

# ----------------------------------------------------------
#  Row 2+3 – TABLES with HEADINGS
# ----------------------------------------------------------

tables = [
    (rmse_table, "RMSE Table (m)"),
    (mae_table,  "MAE Table (m)"),
    (r2_table,   "$R^2$ Table")
]

for idx, (tbl_data, title) in enumerate(tables):
    ax = plt.subplot2grid((3, 3), (1 + idx//3, idx % 3))
    ax.axis("off")

    # Title ABOVE table
    ax.text(
        0.5, 1.15, title,
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        transform=ax.transAxes
    )

    table = ax.table(
        cellText=np.round(tbl_data.values, 4),
        rowLabels=tbl_data.index,
        colLabels=tbl_data.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.3, 1.7)

# ----------------------------------------------------------
# SAVE FIGURE
# ----------------------------------------------------------
OUTPUT_DIR = PROCESSED_DIR.parent / "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = OUTPUT_DIR / "summary_labeled_tables.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("\nSaved FINAL labeled + headed summary figure to:")
print(" ", output_path)

plt.close()
