from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"

# Output folders
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Models
MODELS_DIR = BASE_DIR / "models"

# Machine learning settings
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Default prediction threshold
DEFAULT_THRESHOLD = 0.50


def create_directories():
    """Create required project directories."""

    directories = [
        PROCESSED_DATA_PATH.parent,
        CHARTS_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)