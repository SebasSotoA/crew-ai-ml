"""Tests for atomic deployment pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    DEPLOY_DIR,
    DEPLOY_REPORT_PATH,
    DEPLOY_REQUIREMENTS_PATH,
    EVAL_DIR,
    EVALUATION_REPORT_PATH,
    FAILURE_REPORT_PATH,
    MODEL_PATH,
    PREP_DIR,
    PROJECT_ROOT,
    SPLIT_DIR,
    STREAMLIT_APP_PATH,
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
from crew_ai_ml.pipeline.deploy import (
    DeploymentError,
    configure_app_ui,
    document_deploy_failure,
    finalize_deployment,
    generate_streamlit_app,
    profile_deploy_context,
    validate_deploy_inputs,
    write_deploy_requirements,
)
from crew_ai_ml.pipeline.deploy_workspace import validate_deployment_artifacts
from crew_ai_ml.pipeline.evaluate import (
    analyze_deploy_signals,
    compute_eval_metrics,
    finalize_evaluation,
    generate_eval_plots,
    issue_deploy_verdict,
    profile_eval_context,
    validate_eval_inputs,
)
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
CUSTOM_TITLE = "Titanic Survival Predictor"

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
    analyze_deploy_signals(target)
    if rationale is None:
        rationale = (
            f"Agent verdict {verdict}: evaluation complete with documented metrics "
            "and threshold review for deployment gate."
        )
    issue_deploy_verdict(target, verdict=verdict, rationale=rationale)
    return finalize_evaluation(target)


def _run_atomic_deploy_deploy_branch(target: str, title: str = CUSTOM_TITLE) -> dict:
    validate_deploy_inputs(target)
    profile_deploy_context(target)
    configure_app_ui(target, ui_config={"page_title": title})
    generate_streamlit_app(target)
    write_deploy_requirements(target)
    return finalize_deployment(target)


def _run_atomic_deploy_reject_branch(target: str) -> dict:
    validate_deploy_inputs(target)
    profile_deploy_context(target)
    document_deploy_failure(
        target,
        rationale="Model failed deployment thresholds during automated test run.",
        remediation_steps=[
            "Review feature engineering and class balance in prep_report.md",
            "Tune hyperparameters or try alternative models",
            "Re-run evaluation after improvements",
        ],
    )
    return finalize_deployment(target)


@pytest.fixture(autouse=True)
def clean_output():
    for path in (
        CLEANED_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        MODEL_PATH,
        TRAINING_LOG_PATH,
        EVALUATION_REPORT_PATH,
        STREAMLIT_APP_PATH,
        DEPLOY_REQUIREMENTS_PATH,
        FAILURE_REPORT_PATH,
        DEPLOY_REPORT_PATH,
    ):
        _unlink_if_exists(path)
    for directory in (PREP_DIR, SPLIT_DIR, TRAIN_DIR, EVAL_DIR, DEPLOY_DIR):
        _rmtree_if_exists(directory)
    yield


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_deploy_branch_with_custom_title():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    _run_atomic_eval(TARGET, verdict="DEPLOY")
    result = _run_atomic_deploy_deploy_branch(TARGET, title=CUSTOM_TITLE)

    assert result["verdict"] == "DEPLOY"
    assert result["deployed"] is True
    artifact_paths = result["artifact_paths"]
    assert Path(artifact_paths["streamlit_app_path"]).is_file()
    assert Path(artifact_paths["requirements_path"]).is_file()
    assert not FAILURE_REPORT_PATH.is_file()

    app_text = STREAMLIT_APP_PATH.read_text(encoding="utf-8")
    assert CUSTOM_TITLE in app_text
    assert f'PAGE_TITLE = "{CUSTOM_TITLE}"' in app_text or f"PAGE_TITLE = '{CUSTOM_TITLE}'" in app_text

    ok, _ = validate_deployment_artifacts()
    assert ok


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_do_not_deploy_branch():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    _run_atomic_eval(
        TARGET,
        verdict="DO NOT DEPLOY",
        rationale="Model failed deployment thresholds during automated test run.",
    )
    result = _run_atomic_deploy_reject_branch(TARGET)

    assert result["verdict"] == "DO NOT DEPLOY"
    assert result["deployed"] is False
    assert Path(result["failure_report_path"]).is_file()
    assert not STREAMLIT_APP_PATH.is_file()
    assert not DEPLOY_REQUIREMENTS_PATH.is_file()

    ok, _ = validate_deployment_artifacts()
    assert ok


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_finalize():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    _run_atomic_eval(TARGET, verdict="DEPLOY")
    validate_deploy_inputs(TARGET)
    profile_deploy_context(TARGET)
    configure_app_ui(TARGET, ui_config={"page_title": CUSTOM_TITLE})
    generate_streamlit_app(TARGET)
    write_deploy_requirements(TARGET)

    ok, msg = validate_deployment_artifacts()
    assert not ok
    assert "finalize" in msg.lower()


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_configure_ui_on_reject_branch_raises():
    _run_atomic_prep(DATASET, TARGET)
    _run_atomic_split(TARGET)
    _run_atomic_train(TARGET)
    _run_atomic_eval(
        TARGET,
        verdict="DO NOT DEPLOY",
        rationale="Reject branch test: model blocked from deployment gate.",
    )
    validate_deploy_inputs(TARGET)
    profile_deploy_context(TARGET)

    with pytest.raises(DeploymentError, match="configure_app_ui"):
        configure_app_ui(TARGET, ui_config={"page_title": CUSTOM_TITLE})
