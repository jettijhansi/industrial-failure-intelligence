
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# Database configuration
DATABASE_URL = f"sqlite:///{BASE_DIR / 'industrial.db'}"

# Reproducibility
RANDOM_SEED = 20261022

# Dataset configuration
PLANT_COUNT = 3
LINE_COUNT = 12
MACHINE_COUNT = 100
SENSOR_COUNT = 500

TELEMETRY_DAYS = 90
READINGS_PER_SENSOR_PER_HOUR = 1

FAILURE_COUNT_MIN = 100
FAILURE_COUNT_MAX = 200

# Create directories if they don't exist
for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    GROUND_TRUTH_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
