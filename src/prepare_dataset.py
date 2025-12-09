"""Prepare merged, lagged dataset from raw NOAA CSV files.

Run from the project root as:

    python src/prepare_dataset.py

This will create data/processed/dataset.parquet
"""

from __future__ import annotations

import glob
from datetime import datetime

import numpy as np
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, STATIONS, TARGET_STATION_KEY, N_LAGS


def detect_columns(df: pd.DataFrame):
    """Heuristically detect datetime and water-level columns from a NOAA CSV."""
    # Many NOAA CSVs have first column as datetime and second as water level.
    if df.shape[1] >= 2:
        time_col = df.columns[0]
        wl_col = df.columns[1]
        return time_col, wl_col
    raise ValueError("Could not detect datetime/water-level columns; please inspect CSV header.")


def load_station_series(station_key: str) -> pd.DataFrame:
    pattern = str(RAW_DIR / f"{station_key}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw CSVs found for station key '{station_key}' with pattern {pattern}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        time_col, wl_col = detect_columns(df)
        df = df[[time_col, wl_col]].copy()
        df.rename(columns={time_col: "datetime", wl_col: station_key}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", station_key])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("datetime")
    # Drop duplicates just in case
    merged = merged.drop_duplicates(subset=["datetime"])
    merged.set_index("datetime", inplace=True)
    return merged


def build_merged_frame() -> pd.DataFrame:
    merged = None
    for key in STATIONS.keys():
        series = load_station_series(key)
        if merged is None:
            merged = series
        else:
            merged = merged.join(series, how="outer")
    merged = merged.sort_index()
    # Optional: forward-fill small gaps, then drop remaining NA
    merged = merged.interpolate(limit=3)
    merged = merged.dropna()
    return merged


def add_lag_features(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Add lag features for each numeric column (water level series)."""
    df_lagged = df.copy()
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            df_lagged[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)
    df_lagged = df_lagged.dropna()
    return df_lagged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day and seasonal features based on the datetime index:

    - hour (0–23)
    - dayofyear (1–365/366)
    - sin_hour, cos_hour  [daily cycle]
    - sin_doy,  cos_doy   [annual cycle]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex before adding time features.")

    out = df.copy()

    hour = out.index.hour
    dayofyear = out.index.dayofyear

    out["hour"] = hour
    out["dayofyear"] = dayofyear

    # Cyclical encodings
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

    out["sin_doy"] = np.sin(2 * np.pi * dayofyear / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * dayofyear / 365.25)

    return out


def main():
    print("Loading and merging station time series...")
    merged = build_merged_frame()
    print("Merged shape (no lags):", merged.shape)  # (rows, 4 stations)

    print(f"Adding lag features (N_LAGS={N_LAGS})...")
    lagged = add_lag_features(merged, N_LAGS)
    print("Shape after lags:", lagged.shape)

    print("Adding time-of-day and seasonal features...")
    lagged_with_time = add_time_features(lagged)
    print("Shape after time features:", lagged_with_time.shape)

    # Move index to column
    final_df = lagged_with_time.reset_index()  # datetime column

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset.parquet"
    final_df.to_parquet(out_path, index=False)
    print("Saved processed dataset to", out_path)


if __name__ == "__main__":
    main()
