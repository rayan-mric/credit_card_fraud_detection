from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

# Model and output directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Model files
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
GRADIENT_BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting.pkl"

# Evaluation results
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"


# Create directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)