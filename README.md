# Multi-Station Coastal Water Level Nowcasting  

````markdown

**COSC 6380 – Data Analytics Term Project**

This project uses hourly water level data from **NOAA CO-OPS** for several **Texas Gulf Coast** stations to build a **multi-station, lagged-feature dataset** and train **machine learning models** to nowcast water level at a **target station (Packery Channel, 8775792)**.

Everything runs from simple Python scripts (VS Code or any terminal).

---

## 🔧 1. Setup

```bash
# (optional but recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 📍 2. Station configuration

Configured in `src/config.py`.

**Default target station**

* Packery Channel, TX – **8775792**

**Default neighbor stations**

* Bob Hall Pier, TX – **8775870**
* South Bird Island, TX – **8776139**
* Corpus Christi NAS, TX – **8775421**
* USS Lexington, Corpus Christi Bay, TX – **8775296** (optional)

**Date range (editable in `config.py`):**

```python
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"
```

Change `START_DATE`, `END_DATE`, or the `STATIONS` dict to use different periods or stations.

---

## 🚀 3. How to run

### Step 1 – Download NOAA water level data

Calls the NOAA CO-OPS API and saves CSVs into `data/raw/` for all configured stations.

```bash
python src/download_noaa.py
```

Example log:

```text
Downloading packery (8775792) 2023-01-01 -> 2023-01-31
Saved data/raw/packery_8775792_20230101_20230131.csv
...
```

---

### Step 2 – Build merged, lagged feature dataset

* Loads all raw CSVs from `data/raw/`
* Aligns timestamps across stations
* Creates lag features + time-of-day/day-of-year features
* Writes `data/processed/dataset.parquet`

```bash
python src/prepare_dataset.py
```

---

### Step 3 – Train and evaluate models

* Loads `data/processed/dataset.parquet`
* Splits into train/val/test by time
* Evaluates a **persistence baseline**
* Trains ML models (e.g. Linear Regression, Random Forest, Gradient Boosting / XGBoost)
* Computes **MAE, RMSE, R²**
* Saves metrics/plots (e.g. into `figures/` or `results/`)
* Saves trained models locally in `models/` (ignored by git)

```bash
python src/train_models.py
```

You should see printed metrics for each model and generated plots under `figures/` (and/or `results/`, depending on your script settings).

---

### (Optional) Step 4 – Re-generate figures for report/presentation

If you use a dedicated plotting script or notebook, run it after training, e.g.:

```bash
python src/plot_summary.py
# or open notebooks in VS Code / Jupyter and "Run All"
```

---

## 📁 4. Project structure

```text
waterlevel_term_project/
├─ data/
│  ├─ raw/         # Raw CSVs from NOAA (ignored in git)
│  ├─ processed/   # Feature-engineered datasets (ignored in git)
│  └─ external/    # Small external CSVs used in analysis
├─ figures/        # EDA plots, model comparison, summary figures
├─ models/         # Trained model files (local only, ignored in git)
├─ src/
│  ├─ config.py            # Stations, date range, paths, feature settings
│  ├─ download_noaa.py     # Download hourly water levels from NOAA CO-OPS
│  ├─ prepare_dataset.py   # Clean, merge, and build lag/time features
│  ├─ train_models.py      # Train & evaluate baseline + ML models
│  ├─ analyze_results.py   # Additional analysis / error breakdown
│  ├─ train_multihorizon.py      # (Optional) multi-horizon experiments
│  ├─ plot_multihorizon_results.py
│  └─ plot_summary.py      # Summary plots used in report/presentation
├─ run_all.sh      # Convenience script to run the full pipeline (optional)
├─ requirements.txt
└─ README.md
```

---

## 📝 5. Notes for Run

* All paths are **relative**; run commands from the project root.
* Raw and processed data (`data/raw/`, `data/processed/`) and trained models (`models/`) are **not committed**; they can be recreated with:

  1. `python src/download_noaa.py`
  2. `python src/prepare_dataset.py`
  3. `python src/train_models.py`
* The repository is intended so you can:

  1. Clone the repo
  2. Install requirements
  3. Run the scripts in order
  4. Reproduce key metrics and figures used in the report and presentation.

```
::contentReference[oaicite:0]{index=0}
```
