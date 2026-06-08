"""Pipeline configuration and shared constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREP_DIR = OUTPUT_DIR / "prep"

CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"
TRAIN_DATA_PATH = OUTPUT_DIR / "train.csv"
TEST_DATA_PATH = OUTPUT_DIR / "test.csv"
MODEL_PATH = OUTPUT_DIR / "model_random_forest.pkl"
PREP_REPORT_PATH = OUTPUT_DIR / "prep_report.md"
TRAINING_LOG_PATH = OUTPUT_DIR / "training_log.md"
EVALUATION_REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"
FAILURE_REPORT_PATH = OUTPUT_DIR / "failure_report.md"
FEATURE_METADATA_PATH = OUTPUT_DIR / "feature_metadata.json"
STREAMLIT_APP_PATH = OUTPUT_DIR / "app.py"
DEPLOY_REQUIREMENTS_PATH = OUTPUT_DIR / "requirements.txt"

MISSING_THRESHOLD = 0.15
CORR_REDUNDANCY = 0.8
CORR_IRRELEVANCE = 0.05
IMBALANCE_RATIO = 0.4
MIN_PREP_ROWS = 10

PREP_WORKING_PATH = PREP_DIR / "working.csv"
PREP_PRE_ENCODE_PATH = PREP_DIR / "pre_encode.csv"
PREP_STATE_PATH = PREP_DIR / "prep_state.json"
PREP_PROFILE_PATH = PREP_DIR / "profile.json"

DEPLOY_MIN_F1 = 0.70
DEPLOY_MAX_F1_GAP = 0.05

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [5, 10],
    "max_samples": [0.8, 0.9],
    "criterion": ["gini", "entropy"],
}


def ensure_output_dirs() -> None:
    """Create output and plots directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
