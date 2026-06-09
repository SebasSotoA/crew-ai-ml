"""Tests for atomic model evaluation pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import pytest

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    EVAL_DIR,
    EVAL_REPORT_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    PREP_DIR,
    PROJECT_ROOT,
    SPLIT_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_DIR,
    TRAINING_LOG_PATH,
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
from crew_ai_ml.pipeline.evaluate import (
    EvaluationError,
    analyze_deploy_signals,
    compute_eval_metrics,
    finalize_evaluation,
    generate_eval_plots,
    issue_deploy_verdict,
    profile_eval_context,
    validate_eval_inputs,
)
from crew_ai_ml.pipeline.eval_workspace import validate_evaluation_artifacts
from crew_ai_ml.pipeline.split import (
    finalize_split,
    profile_split_data,
    split_train_test,
    validate_cleaned_data,
    validate_split,
)
from crew_ai_ml.pipeline.train import (
    finalize_training,
    list_training_candidates,
    log_training_decision,
    profile_train_data,
    train_baseline,
    tune_hyperparameters,
    validate_train_data,
)

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


def _run_atomic_eval(target: str, verdict: str = "DEPLOY", rationale: str | None = None) -> dict:
    validate_eval_inputs(target)
    profile = profile_eval_context(target)
    compute_eval_metrics(
        target,
        splits=["train", "test"],
        metrics=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
    )
    plots = profile.get("recommendations", {}).get("plots", ["confusion_matrix"])
    if profile.get("has_predict_proba"):
        plots = list(dict.fromkeys([*plots, "roc_curve"]))
    generate_eval_plots(target, plots=plots)
    signals = analyze_deploy_signals(target)
    if rationale is None:
        checks = signals["signals"]["threshold_checks"]
        rationale = (
            f"Agent verdict {verdict}: test F1={checks['test_f1_weighted']:.4f}, "
            f"gap={checks['f1_gap']:.4f}, thresholds min_f1={checks['deploy_min_f1']} "
            f"max_gap={checks['deploy_max_f1_gap']}."
        )
    issue_deploy_verdict(target, verdict=verdict, rationale=rationale)
    return finalize_evaluation(target)


@pytest.fixture(autouse=True)
def clean_output():
    for path in (
        CLEANED_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        MODEL_PATH,
        TRAINING_LOG_PATH,
        EVALUATION_REPORT_PATH,
        EVAL_REPORT_PATH,
        PLOTS_DIR / "confusion_matrix.png",
        PLOTS_DIR / "roc_curve.png",
    ):
        _unlink_if_exists(path)
    for directory in (PREP_DIR, SPLIT_DIR, TRAIN_DIR, EVAL_DIR):
        _rmtree_if_exists(directory)
    yield


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_full_atomic_eval_flow():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    result = _run_atomic_eval(TARGET)
    assert result["verdict"] in {"DEPLOY", "DO NOT DEPLOY"}
    assert Path(result["evaluation_report_path"]).is_file()
    assert Path(result["eval_report_path"]).is_file()
    bundle = joblib.load(MODEL_PATH)
    assert bundle["evaluation_metrics"]["verdict"] == result["verdict"]
    ok, _ = validate_evaluation_artifacts()
    assert ok


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_finalize():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    validate_eval_inputs(TARGET)
    profile_eval_context(TARGET)
    compute_eval_metrics(TARGET, splits=["train", "test"], metrics=["f1_weighted"])
    generate_eval_plots(TARGET, plots=["confusion_matrix"])
    analyze_deploy_signals(TARGET)
    issue_deploy_verdict(TARGET, verdict="DEPLOY", rationale="Test rationale for guardrail check.")
    ok, msg = validate_evaluation_artifacts()
    assert not ok
    assert "finalize" in msg.lower()


def test_metrics_before_profile_raises():
    with pytest.raises(EvaluationError, match="profile"):
        compute_eval_metrics(TARGET, splits=["test"], metrics=["f1_weighted"])


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_verdict_before_signals_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    validate_eval_inputs(TARGET)
    profile_eval_context(TARGET)
    compute_eval_metrics(TARGET, splits=["train", "test"], metrics=["f1_weighted"])
    generate_eval_plots(TARGET, plots=["confusion_matrix"])
    with pytest.raises(EvaluationError, match="signals_analyzed"):
        issue_deploy_verdict(TARGET, verdict="DEPLOY", rationale="Too early verdict attempt.")


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_agent_can_deploy_despite_gap_signal():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    validate_eval_inputs(TARGET)
    profile_eval_context(TARGET)
    compute_eval_metrics(
        TARGET,
        splits=["train", "test"],
        metrics=["f1_weighted"],
    )
    generate_eval_plots(TARGET, plots=["confusion_matrix"])
    signals = analyze_deploy_signals(TARGET)
    rationale = (
        "Agent overrides gap concern: strong test F1 and stable validation delta; "
        f"signals={signals['signals']['threshold_checks']}"
    )
    issue_deploy_verdict(TARGET, verdict="DEPLOY", rationale=rationale)
    result = finalize_evaluation(TARGET)
    ok, _ = validate_evaluation_artifacts()
    assert ok
    assert result["verdict"] == "DEPLOY"
