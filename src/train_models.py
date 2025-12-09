from __future__ import annotations

import math
from time import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    TARGET_STATION_KEY,
    RANDOM_SEED,
    START_DATE,
    END_DATE,
)


# -------------------------------------------------------
# Helper: metrics
# -------------------------------------------------------
def regression_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def print_threshold_stats(y_true, y_pred, label: str):
    """Print fraction of predictions within 5 / 10 / 15 cm."""
    abs_err = np.abs(y_true - y_pred)
    print(f"    Threshold accuracy for {label}:")
    for thr in [0.05, 0.10, 0.15]:
        frac = (abs_err <= thr).mean() * 100.0
        print(f"        ±{int(thr * 100)} cm : {frac:6.2f}%")


# -------------------------------------------------------
# Main training script
# -------------------------------------------------------
def main():
    dataset_path = PROCESSED_DIR / "dataset.parquet"
    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset from {dataset_path}")
    print(f"Original datetime coverage: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Original samples: {len(df)}")

    # Filter by START_DATE/END_DATE from config.py
    df = df[(df["datetime"] >= START_DATE) & (df["datetime"] <= END_DATE)].copy()
    print(f"\nDataset datetime coverage (after filtering to {START_DATE}–{END_DATE}):")
    print(f"  {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Total samples after filter: {len(df)}")

    if df.empty:
        raise RuntimeError("Filtered dataset is empty. Check START_DATE/END_DATE or raw NOAA downloads.")

    # Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    # ---------- Time-based 80/20 split ----------
    n_total = len(df)
    split_idx = int(n_total * 0.80)
    split_date = df.loc[split_idx, "datetime"]

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("\nTime-based 80/20 split:")
    print(f"  Train: 0 -> {split_idx - 1}  ({len(train_df)} samples)")
    print(f"  Test : {split_idx} -> {n_total - 1}  ({len(test_df)} samples)")
    print(f"  Split date (first sample in test set): {split_date}")

    if test_df.empty:
        raise RuntimeError("Test/validation split ended up empty. Something is wrong with the split logic.")

    # ---------- Features and target ----------
    target_col = TARGET_STATION_KEY   # "packery"
    feature_cols = [c for c in df.columns if c not in ["datetime", target_col]]

    print(f"\nNumber of features: {len(feature_cols)}")
    print("Target column     :", target_col)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # ---------- 1) Scale features ----------
    scaler = StandardScaler()
    print("\n[1/4] Fitting StandardScaler on training features...")
    t0 = time()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Done in {time() - t0:.1f} s")

    # ---------- 2) Baseline persistence ----------
    print("\n[2/4] Evaluating baseline persistence model...")
    lag1_col = f"{target_col}_lag1"
    if lag1_col not in test_df.columns:
        raise KeyError(
            f"Expected column '{lag1_col}' for baseline persistence, "
            "but it is not present in the dataset."
        )

    baseline_pred = test_df[lag1_col].values
    rmse_b, mae_b, r2_b = regression_metrics(y_test, baseline_pred)
    print(f"    [Baseline]      RMSE={rmse_b:.4f}, MAE={mae_b:.4f}, R^2={r2_b:.4f}")
    print_threshold_stats(y_test, baseline_pred, "Baseline")

    # Flags to control which models to train
    TRAIN_SVR = True   # set False if you want it faster
    TRAIN_RF = True

    results = {"Baseline": (rmse_b, mae_b, r2_b)}

    # ---------- 3) LinearSVR (fast linear kernel, recent window) ----------
    if TRAIN_SVR:
        print("\n[3/4] Training LinearSVR (linear kernel, recent window)...")

        # Use only data from 2020 onward for SVR training
        recent_mask = train_df["datetime"] >= pd.Timestamp("2020-01-01")
        X_recent = X_train_scaled[recent_mask.values]
        y_recent = y_train[recent_mask.values]
        print(f"    Using {len(y_recent)} recent samples for SVR.")

        # Stronger subsampling so it definitely finishes quickly
        SUBSAMPLE_STEP = 8
        X_svr = X_recent[::SUBSAMPLE_STEP]
        y_svr = y_recent[::SUBSAMPLE_STEP]
        print(f"    After subsampling: {len(y_svr)} samples.")

        svr = LinearSVR(
            C=10.0,
            epsilon=0.001,
            random_state=RANDOM_SEED,
            max_iter=5000,
        )
        t0 = time()
        svr.fit(X_svr, y_svr)
        print(f"    LinearSVR fit done in {time() - t0:.1f} s")

        t0 = time()
        y_pred_svr = svr.predict(X_test_scaled)
        print(f"    LinearSVR prediction on test set done in {time() - t0:.1f} s")
        rmse_svr, mae_svr, r2_svr = regression_metrics(y_test, y_pred_svr)
        print(f"    [SVR (linear)]   RMSE={rmse_svr:.4f}, MAE={mae_svr:.4f}, R^2={r2_svr:.4f}")
        print_threshold_stats(y_test, y_pred_svr, "SVR")
        results["SVR"] = (rmse_svr, mae_svr, r2_svr)
    else:
        svr = None
        y_pred_svr = np.full_like(y_test, np.nan, dtype=float)

    # ---------- 4) Random Forest (main model, with progress output) ----------
    if TRAIN_RF:
        print("\n[4/4] Training Random Forest (more conservative to improve robustness)...")
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=20,
            max_features=0.4,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=1,
        )
        t0 = time()
        rf.fit(X_train_scaled, y_train)
        print(f"\n    RF fit done in {time() - t0:.1f} s")

        # Train-set performance
        y_train_rf = rf.predict(X_train_scaled)
        rmse_tr, mae_tr, r2_tr = regression_metrics(y_train, y_train_rf)
        print("    RF performance on TRAIN set:")
        print(f"        RMSE={rmse_tr:.4f}, MAE={mae_tr:.4f}, R^2={r2_tr:.4f}")

        # Test-set performance
        t0 = time()
        y_pred_rf = rf.predict(X_test_scaled)
        print(f"    RF prediction on test set done in {time() - t0:.1f} s")
        rmse_rf, mae_rf, r2_rf = regression_metrics(y_test, y_pred_rf)
        print(f"    [Random Forest] RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}, R^2={r2_rf:.4f}")
        print_threshold_stats(y_test, y_pred_rf, "RF")
        results["RF"] = (rmse_rf, mae_rf, r2_rf)
    else:
        rf = None
        y_pred_rf = np.full_like(y_test, np.nan, dtype=float)

    # ---------- Summary ----------
    print("\n>>> SUMMARY OF MODELS (test set) <<<")
    for name, (rmse, mae, r2) in results.items():
        print(f"  {name:10s} RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")

    best_name = min(results.items(), key=lambda kv: kv[1][0])[0]
    best_rmse, best_mae, best_r2 = results[best_name]
    print(f"\n>>> BEST MODEL ON TEST SET (by RMSE): {best_name} <<<")
    print(f"    RMSE={best_rmse:.4f}, MAE={best_mae:.4f}, R^2={best_r2:.4f}")

    # ---------- Save models + scaler + predictions ----------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    if TRAIN_SVR and svr is not None:
        joblib.dump(svr, MODELS_DIR / "svr_model.joblib")
    if TRAIN_RF and rf is not None:
        joblib.dump(rf, MODELS_DIR / "rf_model.joblib")

    print(f"\nSaved scaler and models to {MODELS_DIR}")

    preds_path = PROCESSED_DIR / "test_predictions.csv"
    df_preds = pd.DataFrame(
        {
            "datetime": test_df["datetime"].values,
            "y_true": y_test,
            "y_pred_baseline": baseline_pred,
            "y_pred_svr": y_pred_svr,
            "y_pred_rf": y_pred_rf,
        }
    )
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved test predictions to {preds_path}")


if __name__ == "__main__":
    main()
