"""
Configuration settings for F1 Race Predictor
"""

from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# FastF1 settings
FASTF1_CACHE_DIR = CACHE_DIR / "fastf1"
FASTF1_CACHE_DIR.mkdir(exist_ok=True)

# Data collection settings
MIN_SEASON = 2019  # FastF1 has good data from 2018+
SEASONS = list(range(MIN_SEASON, datetime.now().year + 1))  # Collect up to current year
DEFAULT_DATA_PREFIX = "f1_data"

# The model's feature columns are defined in feature_engineering.get_feature_columns()

# How much more the most recent season in the data counts during training
RECENT_SEASON_WEIGHT = 10.0

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE,
}
