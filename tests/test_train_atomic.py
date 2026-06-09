"""Tests for atomic model training pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    MODEL_PATH,
    PREP_DIR,
    PROJECT_ROOT,
    SPLIT_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_DIR,
    TRAIN_STATE_PATH,
    TRAINING_LOG_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.data_preparation import (
    balance_classes,
    drop_columns,
    encode_features,
    filter_by_correlation,
    finalize_preparation,
    handle_outliers,
    impute_missing,
    load_dataset,
    profile_dataset,
)
from crew_ai_ml.pipeline.model_registry import (
    ALGORITHMS,
    parse_estimator_params,
    parse_param_grid,
)
from crew_ai_ml.pipeline.split import (
    finalize_split,
    profile_split_data,
    split_train_test,
    validate_cleaned_data,
    validate_split,
)
from crew_ai_ml.pipeline.train import (
    TrainingError,
    finalize_training,
    list_training_candidates,
    log_training_decision,
    profile_train_data,
    train_baseline,
    tune_hyperparameters,
    validate_train_data,
)
from crew_ai_ml.pipeline.train_workspace import validate_training_artifacts

DATASET = PROJECT_ROOT / "data" / "titanic.csv"
TARGET = "Sobrevivio"

LR_BASELINE_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}
RF_PARAM_GRID = {"n_estimators": [50, 100], "max_depth": [10, None]}


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _rmtree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _run_atomic_prep(dataset: Path | str, target: str) -> tuple[dict, dict]:
    load_dataset(str(dataset), target)
    profile = profile_dataset(target)
    id_cols = profile["recommendations"]["id_like_columns"]
    if id_cols:
        drop_columns(id_cols, "test id drop", target)
    handle_outliers("iqr_remove", target)
    impute_missing("median", target)
    encode_features("one_hot_drop_first", target)
    filter_by_correlation("both", target)
    balance_classes(profile["recommendations"]["suggested_balance"], target)
    return finalize_preparation(target), profile


def _run_atomic_split(target: str) -> dict:
    validate_cleaned_data(target)
    profile = profile_split_data(target)
    recs = profile["recommendations"]
    split_train_test(
        target,
        test_size=recs["test_size"],
        stratify=recs["stratify_recommended"],
        random_state=recs["random_state"],
    )
    validate_split(target)
    return finalize_split(target)


def _run_atomic_train(target: str) -> dict:
    validate_train_data(target)
    profile = profile_train_data(target)
    recs = profile["recommendations"]
    log_training_decision(
        target,
        issue="algorithm_selection",
        options_considered=profile["supported_algorithms"],
        choice="logistic_regression",
        rationale=(
            f"Baseline logistic_regression for interpretability; profile recommended "
            f"{recs['algorithm']} with tune={recs['tune_recommended']}."
        ),
    )
    train_baseline(target, params=LR_BASELINE_PARAMS, algorithm="logistic_regression")
    tune_hyperparameters(
        target,
        param_grid=RF_PARAM_GRID,
        algorithm="random_forest",
        fixed_params={"random_state": 42, "n_jobs": -1},
    )
    list_training_candidates(target)
    return finalize_training(target)


@pytest.fixture(autouse=True)
def clean_output():
    for path in (
        CLEANED_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        MODEL_PATH,
        TRAINING_LOG_PATH,
    ):
        _unlink_if_exists(path)
    _rmtree_if_exists(PREP_DIR)
    _rmtree_if_exists(SPLIT_DIR)
    _rmtree_if_exists(TRAIN_DIR)
    yield


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_full_atomic_train_flow():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    result = _run_atomic_train(TARGET)
    assert Path(result["model_path"]).is_file()
    bundle = joblib.load(MODEL_PATH)
    assert "algorithm" in bundle
    assert bundle["algorithm"] in {"logistic_regression", "random_forest"}
    ok, _ = validate_training_artifacts()
    assert ok


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_log_training_decision_stored_in_state():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    result = log_training_decision(
        TARGET,
        issue="algorithm_selection",
        options_considered=profile["supported_algorithms"],
        choice="logistic_regression",
        rationale=(
            "Using logistic regression baseline for small dataset interpretability "
            "before tuning random forest."
        ),
    )
    state = json.loads(TRAIN_STATE_PATH.read_text(encoding="utf-8"))
    assert result["decision_count"] == 1
    assert result["issue"] == "algorithm_selection"
    assert result["choice"] == "logistic_regression"
    assert len(state["decisions"]) == 1
    assert state["decisions"][0]["choice"] == "logistic_regression"


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_training_decision():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile_train_data(TARGET)
    train_baseline(TARGET, params=LR_BASELINE_PARAMS, algorithm="logistic_regression")
    tune_hyperparameters(
        TARGET,
        param_grid=RF_PARAM_GRID,
        algorithm="random_forest",
        fixed_params={"random_state": 42, "n_jobs": -1},
    )
    list_training_candidates(TARGET)
    finalize_training(TARGET)
    ok, msg = validate_training_artifacts()
    assert not ok
    assert "log_training_decision" in msg.lower()


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_log_training_decision_before_profile_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    with pytest.raises(TrainingError, match="profile_train_data"):
        log_training_decision(
            TARGET,
            issue="algorithm_selection",
            options_considered=["logistic_regression", "random_forest"],
            choice="logistic_regression",
            rationale="Attempting to log before profiling should fail validation.",
        )


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_finalize():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    log_training_decision(
        TARGET,
        issue="algorithm_selection",
        options_considered=profile["supported_algorithms"],
        choice="logistic_regression",
        rationale=(
            "Baseline logistic_regression for interpretability before finalize guardrail test."
        ),
    )
    train_baseline(TARGET, params=LR_BASELINE_PARAMS, algorithm="logistic_regression")
    ok, msg = validate_training_artifacts()
    assert not ok
    assert "finalize" in msg.lower() or "missing" in msg.lower()


def test_parse_param_grid_null():
    grid = parse_param_grid('{"max_depth": [null, 10]}')
    assert grid["max_depth"][0] is None
    assert grid["max_depth"][1] == 10


def test_parse_estimator_params_null():
    params = parse_estimator_params('{"max_depth": null, "random_state": 42}')
    assert params["max_depth"] is None
    assert params["random_state"] == 42


def _log_algorithm_decision(target: str, profile: dict) -> None:
    recs = profile["recommendations"]
    log_training_decision(
        target,
        issue="algorithm_selection",
        options_considered=profile["supported_algorithms"],
        choice="logistic_regression",
        rationale=(
            f"Baseline logistic_regression for interpretability; profile recommended "
            f"{recs['algorithm']} with tune={recs['tune_recommended']}."
        ),
    )


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_baseline_without_params_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    _log_algorithm_decision(TARGET, profile)
    with pytest.raises(TrainingError, match="params is required"):
        train_baseline(TARGET, params=None, algorithm="logistic_regression")


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_tune_without_grid_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    _log_algorithm_decision(TARGET, profile)
    with pytest.raises(TrainingError, match="param_grid is required"):
        tune_hyperparameters(TARGET, param_grid=None, algorithm="random_forest")


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_grid_over_cap_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    _log_algorithm_decision(TARGET, profile)

    capped = dict(ALGORITHMS)
    capped["random_forest"] = {
        **ALGORITHMS["random_forest"],
        "max_grid_combinations": 1,
    }
    with patch("crew_ai_ml.pipeline.train.ALGORITHMS", capped):
        with pytest.raises(TrainingError, match="exceeding max"):
            tune_hyperparameters(
                TARGET,
                algorithm="random_forest",
                param_grid={"n_estimators": [50, 100]},
            )


def test_validate_train_data_rejects_nan_features():
    ensure_output_dirs()
    pd.DataFrame(
        {
            "feat_num": [1.0, 2.0, float("nan"), 4.0, 5.0] * 2,
            "target": [0, 1] * 5,
        }
    ).to_csv(TRAIN_DATA_PATH, index=False)

    with pytest.raises(TrainingError, match="missing values"):
        validate_train_data("target")


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_train_baseline_rejects_nan_features():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    validate_train_data(TARGET)
    profile = profile_train_data(TARGET)
    _log_algorithm_decision(TARGET, profile)

    df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    df.iloc[0, 0] = float("nan")
    df.to_csv(TRAIN_DATA_PATH, index=False)

    with pytest.raises(TrainingError, match="missing values"):
        train_baseline(
            TARGET,
            params=LR_BASELINE_PARAMS,
            algorithm="logistic_regression",
        )
