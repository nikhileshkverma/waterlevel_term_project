#!/usr/bin/env bash
# ================================================================
#  Water Level Term Project – RUN SUMMARY ONLY (No Training)
# ================================================================

set -euo pipefail

# Move to directory where this script is located
cd "$(dirname "$0")"

echo ""
echo "==============================================================="
echo "===  WATER LEVEL TERM PROJECT : SUMMARY + PLOTS ONLY        ==="
echo "==============================================================="
echo ""

# -------------------------------------------------------------------
# 1. Virtual environment setup
# -------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[1] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[2] Activating virtual environment..."
source .venv/bin/activate

echo "[3] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------------------------------------------
# 2. Ensure directory structure
# -------------------------------------------------------------------
echo "[4] Ensuring folders exist..."
mkdir -p data/raw data/external data/processed
mkdir -p figures/{eda,features,models,errors,robustness,independent,multihorizon,interpretability,summary}
mkdir -p models

# -------------------------------------------------------------------
# 3. SKIPPING ALL HEAVY STEPS (commented out)
# -------------------------------------------------------------------
echo ""
echo "[5] SKIPPED: Downloading NOAA data"
# python src/download_noaa.py

echo "[6] SKIPPED: Preparing dataset"
python src/prepare_dataset.py

echo "[7] SKIPPED: Training 0-hour models"
python src/train_models.py

echo "[8] SKIPPED: Generating diagnostic plots"
python src/analyze_results.py

echo "[9] SKIPPED: Training multi-horizon models"
python src/train_multihorizon.py

echo "[10] SKIPPED: Plotting multi-horizon raw plots"
python src/plot_multihorizon_results.py

echo "[11] SKIPPED: Independent 2025 test"
python src/independent_test.py

# -------------------------------------------------------------------
# 4. ONLY RUN SUMMARY PLOTS
# -------------------------------------------------------------------
echo ""
echo "[12] Generating final combined summary plot..."
python src/plot_summary.py || echo "⚠ Summary plot skipped"

echo ""
echo "==============================================================="
echo "===  SUMMARY-ONLY PIPELINE COMPLETE!                         ==="
echo "==============================================================="
echo ""
echo "✔ Final figure saved in ./figures/summary/"
echo "✔ No training or downloading performed."
