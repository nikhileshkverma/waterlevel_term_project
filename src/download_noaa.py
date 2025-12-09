"""Download hourly water level data from NOAA CO-OPS for configured stations.

Run from the project root as:

    python src/download_noaa.py

This will save monthly CSV files into data/raw/.
"""

from datetime import datetime, timedelta
import os
import sys
import time
import io            # <-- add this line
import requests
import pandas as pd

from config import NOAA_BASE_URL, STATIONS, START_DATE, END_DATE, RAW_DIR


def month_range(start: datetime, end: datetime):
    """Yield (month_start, month_end) tuples covering [start, end]."""
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1)
        month_end = min(nxt - timedelta(days=1), end)
        yield cur, month_end
        cur = nxt


def fetch_noaa_chunk(station_id: str, begin: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "product": "water_level",
        "application": "COSC6380_term_project",
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "GMT",
        "units": "metric",
        "interval": "h",  # hourly
        "format": "csv",
    }
    resp = requests.get(NOAA_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    # NOAA sometimes returns an HTML error as CSV; let pandas parse and we can inspect
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(START_DATE)
    end_dt = datetime.fromisoformat(END_DATE)

    print(f"Downloading NOAA water level data from {START_DATE} to {END_DATE}\n")

    for key, station_id in STATIONS.items():
        print(f"=== Station {key} ({station_id}) ===")
        for month_start, month_end in month_range(start_dt, end_dt):
            print(f"  {month_start.date()} -> {month_end.date()}", end=" ... ", flush=True)
            try:
                df = fetch_noaa_chunk(station_id, month_start, month_end)
            except Exception as e:
                print("ERROR", e)
                continue

            # Save raw CSV
            out_name = f"{key}_{station_id}_{month_start.strftime('%Y%m%d')}_{month_end.strftime('%Y%m%d')}.csv"
            out_path = RAW_DIR / out_name
            df.to_csv(out_path, index=False)
            print("saved", out_path.name)
            time.sleep(0.2)  # be gentle to API

    print("\nDone. Check data/raw/ for downloaded CSV files.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
