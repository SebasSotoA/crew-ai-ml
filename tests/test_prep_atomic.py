"""Tests for atomic data preparation pipeline."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    DEPLOY_REQUIREMENTS_PATH,
    EVALUATION_REPORT_PATH,
    FAILURE_REPORT_PATH,
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    PREP_DIR,
    SPLIT_DIR,
    PREP_REPORT_PATH,
    PROJECT_ROOT,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_DIR,
    TRAINING_LOG_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.data_preparation import (
    DataPreparationError,
    balance_classes,
    drop_columns,
    encode_features,
    filter_by_correlation,
    finalize_preparation,
    handle_outliers,
    impute_missing,
    load_dataset,
    profile_dataset,
    resolve_dataset_path,
)
from crew_ai_ml.pipeline.feature_transform import build_input_schema
from crew_ai_ml.pipeline.prep_workspace import (
    prep_tool_guard,
    set_kickoff_inputs,
    validate_prep_artifacts,
)
from crew_ai_ml.pipeline import prep_workspace as prep_ws
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
PASSENGERS_DATASET = PROJECT_ROOT / "data" / "passengers_satisfaction.csv"
PASSENGERS_TARGET = "satisfaction"
PASSENGERS_DROP_COLS = ["Unnamed: 0", "id"]


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


@pytest.fixture(autouse=True)
def clean_output():
    for path in (
        CLEANED_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        MODEL_PATH,
        PREP_REPORT_PATH,
        TRAINING_LOG_PATH,
        EVALUATION_REPORT_PATH,
        FAILURE_REPORT_PATH,
        FEATURE_METADATA_PATH,
        DEPLOY_REQUIREMENTS_PATH,
    ):
        _unlink_if_exists(path)
    _rmtree_if_exists(PLOTS_DIR)
    _rmtree_if_exists(PREP_DIR)
    _rmtree_if_exists(SPLIT_DIR)
    _rmtree_if_exists(TRAIN_DIR)
    yield


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_resolve_dataset_path_from_filename_only():
    set_kickoff_inputs(str(DATASET), TARGET)
    resolved = resolve_dataset_path("titanic.csv")
    assert resolved == DATASET.resolve()


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_load_dataset_accepts_filename_only():
    set_kickoff_inputs(str(DATASET), TARGET)
    result = load_dataset("titanic.csv", TARGET)
    assert "titanic.csv" in result["dataset_path"]


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_atomic_prep_pipeline():
    result, profile = _run_atomic_prep(DATASET, TARGET)
    assert profile["rows"] > 0
    assert "recommendations" in profile
    assert Path(result["cleaned_data_path"]).is_file()
    assert Path(result["feature_metadata_path"]).is_file()
    ok, _ = validate_prep_artifacts()
    assert ok


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


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_downstream_split_train_after_atomic_prep():
    _run_atomic_prep(DATASET, TARGET)
    split_result = _run_atomic_split(TARGET)
    assert split_result["train_rows"] > 0
    train_result = _run_atomic_train(TARGET)
    assert train_result.get("best_cv_score") is not None or train_result.get(
        "validation_f1_weighted"
    ) is not None


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_finalize():
    load_dataset(str(DATASET), TARGET)
    ok, msg = validate_prep_artifacts()
    assert not ok
    assert "finalize" in msg.lower() or "missing" in msg.lower()


@pytest.mark.skipif(
    not PASSENGERS_DATASET.exists(),
    reason="Passengers satisfaction dataset not present",
)
def test_dropped_columns_absent_from_feature_metadata():
    set_kickoff_inputs(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    load_dataset(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    drop_columns(PASSENGERS_DROP_COLS, "test id-like columns", PASSENGERS_TARGET)
    impute_missing("drop_rows", PASSENGERS_TARGET)
    encode_features("one_hot_drop_first", PASSENGERS_TARGET)
    result = finalize_preparation(PASSENGERS_TARGET)

    metadata = json.loads(Path(result["feature_metadata_path"]).read_text(encoding="utf-8"))
    field_names = [entry["name"] for entry in metadata]
    for col in PASSENGERS_DROP_COLS:
        assert col not in field_names


@pytest.mark.skipif(
    not PASSENGERS_DATASET.exists(),
    reason="Passengers satisfaction dataset not present",
)
def test_id_like_columns_auto_excluded_without_drop_tool():
    """encode_features auto-drops id-like cols even when agent skips drop_columns."""
    set_kickoff_inputs(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    load_dataset(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    profile_dataset(PASSENGERS_TARGET)
    impute_missing("drop_rows", PASSENGERS_TARGET)
    encode_features("one_hot_drop_first", PASSENGERS_TARGET)
    result = finalize_preparation(PASSENGERS_TARGET)

    metadata = json.loads(Path(result["feature_metadata_path"]).read_text(encoding="utf-8"))
    field_names = [entry["name"] for entry in metadata]
    cleaned_cols = pd.read_csv(result["cleaned_data_path"], nrows=0).columns.tolist()

    for col in PASSENGERS_DROP_COLS:
        assert col not in field_names
        assert col not in cleaned_cols


def test_guardrail_rejects_non_numeric_cleaned_data():
    ensure_output_dirs()
    pd.DataFrame(
        {
            "feat_num": list(range(10)),
            "feat_str": ["a"] * 10,
            "target": [0, 1] * 5,
        }
    ).to_csv(CLEANED_DATA_PATH, index=False)
    state = prep_ws.load_state()
    state["target_column"] = "target"
    state["finalized"] = True
    prep_ws.save_state(state)

    ok, msg = validate_prep_artifacts()
    assert not ok
    assert "non-numeric" in msg.lower()


def test_guardrail_rejects_nan_cleaned_data():
    ensure_output_dirs()
    pd.DataFrame(
        {
            "feat_num": [1.0, 2.0, float("nan"), 4.0, 5.0] * 2,
            "target": [0, 1] * 5,
        }
    ).to_csv(CLEANED_DATA_PATH, index=False)
    state = prep_ws.load_state()
    state["target_column"] = "target"
    state["finalized"] = True
    prep_ws.save_state(state)

    ok, msg = validate_prep_artifacts()
    assert not ok
    assert "missing values" in msg.lower()


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_encode_features_rejects_row_count_mismatch():
    load_dataset(str(DATASET), TARGET)
    profile = profile_dataset(TARGET)
    id_cols = profile["recommendations"]["id_like_columns"]
    if id_cols:
        drop_columns(id_cols, "test id drop", TARGET)
    impute_missing("drop_rows", TARGET)

    working = prep_ws.read_working()
    corrupt = pd.concat([working, working.iloc[:5]], ignore_index=True)
    prep_ws.write_working(corrupt)

    with pytest.raises(DataPreparationError, match="working.csv has"):
        encode_features("one_hot_drop_first", TARGET)


def test_atomic_csv_write_overwrites_existing(tmp_path):
    path = tmp_path / "working.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    prep_ws._atomic_csv_write(pd.DataFrame({"a": [9, 8], "b": [1, 2]}), path)
    result = pd.read_csv(path)
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_finalize_preparation_raises_on_nan_features():
    load_dataset(str(DATASET), TARGET)
    profile = profile_dataset(TARGET)
    id_cols = profile["recommendations"]["id_like_columns"]
    if id_cols:
        drop_columns(id_cols, "test id drop", TARGET)
    encode_features("one_hot_drop_first", TARGET)

    working = prep_ws.read_working()
    working.iloc[0, 0] = float("nan")
    prep_ws.write_working(working)

    state = prep_ws.load_state()
    state["steps_applied"].append(
        {
            "tool": "encode_features",
            "params": {"strategy": "one_hot_drop_first"},
            "rows_before": len(working),
            "rows_after": len(working),
            "summary": "test stub",
        }
    )
    prep_ws.save_state(state)

    with pytest.raises(DataPreparationError, match="missing values"):
        finalize_preparation(TARGET)


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_load_dataset_clears_stale_cleaned_data():
    ensure_output_dirs()
    CLEANED_DATA_PATH.write_text("stale", encoding="utf-8")
    FEATURE_METADATA_PATH.write_text("[]", encoding="utf-8")
    split_ws = __import__(
        "crew_ai_ml.pipeline.split_workspace", fromlist=["save_state", "load_state"]
    )
    split_state = split_ws.load_state()
    split_state["stage"] = "validated_split"
    split_state["finalized"] = True
    split_ws.save_state(split_state)

    load_dataset(str(DATASET), TARGET)

    assert not CLEANED_DATA_PATH.exists()
    assert not FEATURE_METADATA_PATH.exists()
    assert split_ws.load_state()["stage"] == "uninitialized"


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_finalize_preparation_raises_on_corrupt_tail():
    load_dataset(str(DATASET), TARGET)
    profile = profile_dataset(TARGET)
    id_cols = profile["recommendations"]["id_like_columns"]
    if id_cols:
        drop_columns(id_cols, "test id drop", TARGET)
    encode_features("one_hot_drop_first", TARGET)

    working = prep_ws.read_working()
    corrupt = working.copy()
    tail = pd.DataFrame(
        {
            col: ["bad_value"] * 3 if col != TARGET else working[TARGET].iloc[:3].tolist()
            for col in working.columns
        }
    )
    corrupt = pd.concat([corrupt, tail], ignore_index=True)
    prep_ws.write_working(corrupt)

    state = prep_ws.load_state()
    state["steps_applied"].append(
        {
            "tool": "balance_classes",
            "params": {"method": "none"},
            "rows_before": len(working),
            "rows_after": len(working),
            "summary": "test stub",
        }
    )
    prep_ws.save_state(state)

    with pytest.raises(DataPreparationError, match="Row count|Non-numeric"):
        finalize_preparation(TARGET)


def test_write_working_uses_memory_cache():
    prep_ws.reset_workspace("test.csv", "target")
    df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
    prep_ws.write_working(df)

    from crew_ai_ml.pipeline.config import PREP_WORKING_PATH

    PREP_WORKING_PATH.unlink(missing_ok=True)

    with patch("pandas.read_csv") as mock_read_csv:
        result = prep_ws.read_working()
        mock_read_csv.assert_not_called()

    assert len(result) == 3


def test_concurrent_write_working_serializes():
    prep_ws.reset_workspace("test.csv", "target")
    results: list[int] = []
    errors: list[Exception] = []

    def writer(row_count: int, delay: float) -> None:
        try:
            with prep_tool_guard():
                df = pd.DataFrame(
                    {"feat": list(range(row_count)), "target": [0] * row_count}
                )
                prep_ws.write_working(df)
                if delay:
                    threading.Event().wait(delay)
                results.append(len(prep_ws.read_working()))
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=(5, 0.15))
    t2 = threading.Thread(target=writer, args=(9, 0.0))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors
    assert sorted(results) == [5, 9]
    assert len(prep_ws.read_working()) == 9


@pytest.mark.skipif(
    not PASSENGERS_DATASET.exists(),
    reason="Passengers satisfaction dataset not present",
)
def test_encode_rejects_when_imputation_required_but_skipped():
    set_kickoff_inputs(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    load_dataset(str(PASSENGERS_DATASET), PASSENGERS_TARGET)
    profile = profile_dataset(PASSENGERS_TARGET)
    assert profile["recommendations"]["apply_imputation"] is True

    with pytest.raises(DataPreparationError, match="impute_missing"):
        encode_features("one_hot_drop_first", PASSENGERS_TARGET)


def test_build_input_schema_excludes_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [25.0, 30.0, 35.0],
            "color": ["red", "blue", "red"],
        }
    )
    dummy_groups = {"color": ["color_blue"]}
    final_feature_columns = ["age", "color_blue"]

    schema = build_input_schema(
        df,
        dummy_groups,
        final_feature_columns,
        excluded_columns=["id"],
    )
    field_names = [entry["name"] for entry in schema]

    assert "id" not in field_names
    assert "age" in field_names
    assert "color" in field_names
