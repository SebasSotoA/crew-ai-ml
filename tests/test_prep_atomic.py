"""Tests for atomic data preparation pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
    PREP_REPORT_PATH,
    PROJECT_ROOT,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
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
    resolve_dataset_path,
)
from crew_ai_ml.pipeline.feature_transform import build_input_schema
from crew_ai_ml.pipeline.prep_workspace import set_kickoff_inputs
from crew_ai_ml.pipeline.prep_workspace import validate_prep_artifacts
from crew_ai_ml.pipeline.split import run_split
from crew_ai_ml.pipeline.train import run_training

DATASET = PROJECT_ROOT / "data" / "titanic.csv"
TARGET = "Sobrevivio"
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


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_downstream_split_train_after_atomic_prep():
    _run_atomic_prep(DATASET, TARGET)
    split_result = run_split(TARGET)
    assert split_result["train_rows"] > 0
    train_result = run_training(TARGET)
    assert train_result["best_cv_score"] >= 0


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
    encode_features("one_hot_drop_first", PASSENGERS_TARGET)
    result = finalize_preparation(PASSENGERS_TARGET)

    metadata = json.loads(Path(result["feature_metadata_path"]).read_text(encoding="utf-8"))
    field_names = [entry["name"] for entry in metadata]
    cleaned_cols = pd.read_csv(result["cleaned_data_path"], nrows=0).columns.tolist()

    for col in PASSENGERS_DROP_COLS:
        assert col not in field_names
        assert col not in cleaned_cols


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
