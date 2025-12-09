from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist (scripts also call mkdir just in case)
for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# === NOAA CO-OPS configuration ===
NOAA_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Mapping from short station keys to NOAA station IDs
STATIONS = {
    "packery": "8775792",        # Packery Channel, TX (target)
    "bob_hall": "8775870",       # Bob Hall Pier, Corpus Christi, TX
    "lexington": "8775296",      # USS Lexington, Corpus Christi Bay, TX
    "port_aransas": "8775237",   # Port Aransas, TX
}

# Target station key (for prediction)
TARGET_STATION_KEY = "packery"

# Date range for downloads (inclusive)
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
# Random seed for reproducibility
RANDOM_SEED = 42

# Number of hourly lags to use as features for each station
N_LAGS = 24
