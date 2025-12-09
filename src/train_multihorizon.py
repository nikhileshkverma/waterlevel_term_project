from __future__ import annotations

import math
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from config import PROCESSED_DIR, TARGET_STATION_KEY, RANDOM_SEED

# Horizons to train
HORIZONS = [1, 6, 12]


# ---------------------- METRICS ----------------------
def regression_metrics(y_true, y_pred):
    rmse = math.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.abs(y_true - y_pred).mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def print_threshold_stats(y_true, y_pred, label: str):
    abs_err = np.abs(y_true - y_pred)
    print(f"    Threshold accuracy for {label}:")
    for thr in [0.05, 0.10, 0.15]:
        frac = (abs_err <= thr).mean() * 100.0
        print(f"        ±{int(thr * 100)} cm : {frac:6.2f}%")


# ---------------------- BUILD DATASET ----------------------
def build_horizon_dataset(df: pd.DataFrame, horizon: int):
    df_h = df.copy()

    target_col = f"{TARGET_STATION_KEY}_lead{horizon}"
    df_h[target_col] = df_h[TARGET_STATION_KEY].shift(-horizon)

    df_h = df_h.dropna(subset=[target_col]).reset_index(drop=True)

    n_total = len(df_h)
    split_idx = int(n_total * 0.8)
    split_date = df_h.loc[split_idx, "datetime"]

    train_df = df_h.iloc[:split_idx].copy()
    test_df = df_h.iloc[split_idx:].copy()

    print(f"\n=== Horizon {horizon} h ===")
    print("Time-based 80/20 split:")
    print(f"  Train: 0 -> {split_idx - 1}  ({len(train_df)} samples)")
    print(f"  Test : {split_idx} -> {n_total - 1}  ({len(test_df)} samples)")
    print(f"  Split date: {split_date}")

    feature_cols = [
        c for c in df_h.columns if c not in ["datetime", target_col]
    ]

    print(f"Number of features: {len(feature_cols)}")
    print(f"Target column     : {target_col}")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    return feature_cols, train_df, test_df, X_train, y_train, X_test, y_test, target_col


# ================================================================
#                           MAIN
# ================================================================
def main():
    dataset_path = PROCESSED_DIR / "dataset.parquet"
    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset from {dataset_path}")
    print(f"Datetime coverage: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Total samples: {len(df)}")

    results = []

    for H in HORIZONS:
        (
            feature_cols,
            train_df,
            test_df,
            X_train,
            y_train,
            X_test,
            y_test,
            target_col,
        ) = build_horizon_dataset(df, H)

        baseline_pred = test_df[TARGET_STATION_KEY].values
        rmse_b, mae_b, r2_b = regression_metrics(y_test, baseline_pred)
        print(f"\n[Baseline {H}h]")
        print(f"    RMSE={rmse_b:.4f}, MAE={mae_b:.4f}, R^2={r2_b:.4f}")
        print_threshold_stats(y_test, baseline_pred, f"Baseline {H}h")

        scaler = StandardScaler()
        print("\n    [1] Fitting StandardScaler...")
        t0 = time()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"        Done in {time() - t0:.1f} s")

        print("    [2] Training LinearSVR...")
        svr = LinearSVR(
            epsilon=0.001, C=50.0,
            random_state=RANDOM_SEED, max_iter=2000,
        )
        t0 = time()
        svr.fit(X_train_scaled, y_train)
        print(f"        SVR fit done in {time() - t0:.1f} s")

        y_svr = svr.predict(X_test_scaled)
        rmse_svr, mae_svr, r2_svr = regression_metrics(y_test, y_svr)
        print(f"        [SVR] RMSE={rmse_svr:.4f}, MAE={mae_svr:.4f}, R^2={r2_svr:.4f}")
        print_threshold_stats(y_test, y_svr, f"SVR {H}h")

        print("    [3] Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=20,
            max_features=0.4,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        t0 = time()
        rf.fit(X_train_scaled, y_train)
        print(f"        RF fit done in {time() - t0:.1f} s")

        y_rf = rf.predict(X_test_scaled)
        rmse_rf, mae_rf, r2_rf = regression_metrics(y_test, y_rf)
        print(f"        [RF] RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}, R^2={r2_rf:.4f}")
        print_threshold_stats(y_test, y_rf, f"RF {H}h")

        print(f"\n    >>> SUMMARY (horizon = {H} h) <<<")
        print(f"      Baseline   RMSE={rmse_b:.4f}")
        print(f"      Linear SVR RMSE={rmse_svr:.4f}")
        print(f"      RF         RMSE={rmse_rf:.4f}")
        print("-" * 60)

        results.extend([
            {"horizon_h": H, "model": "Baseline", "rmse": rmse_b, "mae": mae_b, "r2": r2_b},
            {"horizon_h": H, "model": "Linear SVR", "rmse": rmse_svr, "mae": mae_svr, "r2": r2_svr},
            {"horizon_h": H, "model": "Random Forest", "rmse": rmse_rf, "mae": mae_rf, "r2": r2_rf},
        ])

    df_results = pd.DataFrame(results)
    csv_path = PROCESSED_DIR / "multihorizon_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved multi-horizon metrics to {csv_path}\n")


if __name__ == "__main__":
    main()
