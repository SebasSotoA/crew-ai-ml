"""Pipeline configuration and shared constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREP_DIR = OUTPUT_DIR / "prep"
SPLIT_DIR = OUTPUT_DIR / "split"
TRAIN_DIR = OUTPUT_DIR / "train"
EVAL_DIR = OUTPUT_DIR / "eval"
DEPLOY_DIR = OUTPUT_DIR / "deploy"

CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"
TRAIN_DATA_PATH = OUTPUT_DIR / "train.csv"
TEST_DATA_PATH = OUTPUT_DIR / "test.csv"
MODEL_PATH = OUTPUT_DIR / "model.pkl"
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

SPLIT_STATE_PATH = SPLIT_DIR / "split_state.json"
SPLIT_PROFILE_PATH = SPLIT_DIR / "profile.json"
SPLIT_TRAIN_HOLD_PATH = SPLIT_DIR / "train_hold.csv"
SPLIT_TEST_HOLD_PATH = SPLIT_DIR / "test_hold.csv"
SPLIT_REPORT_PATH = OUTPUT_DIR / "split_report.md"

TRAIN_STATE_PATH = TRAIN_DIR / "train_state.json"
TRAIN_PROFILE_PATH = TRAIN_DIR / "profile.json"
TRAIN_CANDIDATES_DIR = TRAIN_DIR / "candidates"

EVAL_STATE_PATH = EVAL_DIR / "eval_state.json"
EVAL_PROFILE_PATH = EVAL_DIR / "profile.json"
EVAL_REPORT_PATH = OUTPUT_DIR / "eval_report.md"

DEPLOY_STATE_PATH = DEPLOY_DIR / "deploy_state.json"
DEPLOY_PROFILE_PATH = DEPLOY_DIR / "profile.json"
DEPLOY_REPORT_PATH = OUTPUT_DIR / "deployment_report.md"
INFERENCE_UTILS_PATH = OUTPUT_DIR / "inference_utils.py"

DEPLOY_MIN_F1 = 0.70
DEPLOY_MAX_F1_GAP = 0.05


def ensure_output_dirs() -> None:
    """Create output and plots directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
