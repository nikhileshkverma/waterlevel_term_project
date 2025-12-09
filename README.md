# Multi-Station Coastal Water Level Nowcasting – Term Project (COSC 6380)

This project downloads hourly water level data from NOAA CO-OPS for several Texas Gulf Coast
stations, builds a multi-station lagged-feature dataset, and trains machine learning models to
nowcast the water level at a target station (Packery Channel).

The code is organized as simple Python scripts that you can run in VS Code or any terminal.

# 1. Setup

1. Install **Python 3.9+** on your system.

2. Open this folder in **VS Code** (or any terminal).

3. (Recommended) Create and activate a virtual environment:

   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate

4. Install dependencies:

   pip install -r requirements.txt

# 2. Station configuration

The stations and date range are configured in `src/config.py`.

By default we use:

- Target station:
  - Packery Channel, TX – station ID **8775792**

- Neighbor (input) stations:
  - Bob Hall Pier, TX – station ID **8775870**
  - South Bird Island, TX – station ID **8776139**
  - Corpus Christi NAS, TX – station ID **8775421**
  - USS Lexington, Corpus Christi Bay, TX – station ID **8775296** (optional)

- Default period (can be changed in `config.py`):
  - `START_DATE = "2023-01-01"`
  - `END_DATE   = "2024-12-31"`

You can change `START_DATE`, `END_DATE`, or the `STATIONS` dict in `src/config.py` if needed.

# 3. Steps to run

## Step 1 – Download NOAA water level data

This script calls the NOAA CO-OPS API and saves CSV files into `data/raw/`
for each configured station and date range.

Run:

   python src/download_noaa.py

If everything works, you should see messages like:

   Downloading packery (8775792) 2023-01-01 -> 2023-01-31
   Saved data/raw/packery_8775792_20230101_20230131.csv
   ...

If you change stations or dates in `config.py`, simply rerun this script.

## Step 2 – Build merged, lagged dataset

This script will:

- Load all raw CSVs from `data/raw/`
- Extract datetime and water level for each station
- Align timestamps across stations
- Merge all stations into a single table
- Create lag features and time-of-day / day-of-year features
- Save the final dataset to `data/processed/dataset.parquet`

Run:

   python src/prepare_dataset.py

## Step 3 – Train and evaluate models

This script will:

- Load the processed dataset from `data/processed/dataset.parquet`
- Split data into train/validation/test using time-based splits
- Evaluate a **persistence baseline** (predict current = last value)
- Train multiple ML models, for example:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting / XGBoost (if installed)
- Compute metrics (MAE, RMSE, R²) on the test set
- Save trained models and scalers into `models/`
- Save metrics and plots into `results/`

Run:

   python src/train_models.py

You should see printed metrics for each model and any plots stored under `results/`.

## Step 4 – (Optional) Re-generate figures for the report/presentation

If you have a separate script/notebook for plots (e.g. `notebooks/05_plots.ipynb`
or `src/make_plots.py`), run it after training:

   # Example script version (if present)
   python src/make_plots.py

   # or open the notebook in VS Code / Jupyter and run all cells

# 4. Project structure (expected)

- `data/`
  - `raw/`        # Raw CSVs downloaded from NOAA
  - `processed/`  # Final merged and feature-engineered datasets
- `models/`       # Saved model weights, scalers, etc.
- `results/`      # Metrics, plots, and any exported tables
- `src/`
  - `config.py`           # Central configuration: stations, dates, paths, lags
  - `download_noaa.py`    # Download hourly water levels from NOAA CO-OPS API
  - `prepare_dataset.py`  # Clean, merge, and build lag/time features
  - `train_models.py`     # Train & evaluate baseline + ML models
  - (optional) `make_plots.py` or plotting helpers
- `notebooks/`    # (Optional) Jupyter notebooks for EDA and figures
- `requirements.txt`  # Python dependencies

# 5. Notes

- All paths are **relative**; run commands from the **project root**.
- If NOAA changes the column names or CSV structure, update the small helper
  functions in `src/download_noaa.py` and `src/prepare_dataset.py` that parse
  the date/time and water level columns.
- Large datasets do **not** need to be committed to git. You can re-download
  using `download_noaa.py` as long as the station IDs and dates remain valid.
- This project is designed so that Dr. King can:
  - Clone the repo
  - Install requirements
  - Run the three main scripts in order
  - Reproduce your metrics and main plots for grading.
