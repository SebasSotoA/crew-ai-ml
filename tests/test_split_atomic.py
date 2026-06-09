"""Tests for atomic train/test split pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    PREP_DIR,
    PROJECT_ROOT,
    SPLIT_DIR,
    SPLIT_REPORT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
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
from crew_ai_ml.pipeline.split import (
    SplitError,
    finalize_split,
    profile_split_data,
    split_train_test,
    validate_cleaned_data,
    validate_split,
)
from crew_ai_ml.pipeline.split_workspace import validate_split_artifacts
from crew_ai_ml.pipeline import split_workspace as split_ws

DATASET = PROJECT_ROOT / "data" / "titanic.csv"
TARGET = "Sobrevivio"


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
        SPLIT_REPORT_PATH,
    ):
        _unlink_if_exists(path)
    _rmtree_if_exists(PREP_DIR)
    _rmtree_if_exists(SPLIT_DIR)
    yield


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_full_atomic_split_flow():
    _run_atomic_prep(DATASET, TARGET)
    validate_cleaned_data(TARGET)
    profile = profile_split_data(TARGET)
    recs = profile["recommendations"]
    split_train_test(
        TARGET,
        test_size=recs["test_size"],
        stratify=recs["stratify_recommended"],
        random_state=recs["random_state"],
    )
    validate_split(TARGET)
    result = finalize_split(TARGET)
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert Path(result["train_path"]).is_file()
    assert Path(result["test_path"]).is_file()
    ok, _ = validate_split_artifacts()
    assert ok


@pytest.mark.skipif(not DATASET.exists(), reason="Titanic dataset not present")
def test_guardrail_fails_without_finalize():
    _run_atomic_prep(DATASET, TARGET)
    validate_cleaned_data(TARGET)
    profile = profile_split_data(TARGET)
    recs = profile["recommendations"]
    split_train_test(
        TARGET,
        test_size=recs["test_size"],
        stratify=recs["stratify_recommended"],
        random_state=recs["random_state"],
    )
    validate_split(TARGET)
    ok, msg = validate_split_artifacts()
    assert not ok
    assert "finalize" in msg.lower() or "missing" in msg.lower()


def test_validate_cleaned_data_rejects_string_features():
    df = pd.DataFrame(
        {
            "feat_num": list(range(10)),
            "feat_str": ["a"] * 10,
            "target": [0, 1] * 5,
        }
    )
    ensure_output_dirs()
    df.to_csv(CLEANED_DATA_PATH, index=False)

    with pytest.raises(SplitError, match="numeric"):
        validate_cleaned_data("target")


def test_validate_cleaned_data_rejects_nan_features():
    df = pd.DataFrame(
        {
            "feat_num": [1.0, 2.0, float("nan"), 4.0, 5.0] * 2,
            "target": [0, 1] * 5,
        }
    )
    ensure_output_dirs()
    df.to_csv(CLEANED_DATA_PATH, index=False)

    with pytest.raises(SplitError, match="missing values"):
        validate_cleaned_data("target")


def test_validate_cleaned_data_resets_stale_split_state_on_failure():
    df = pd.DataFrame(
        {
            "feat_num": list(range(10)),
            "feat_str": ["a"] * 10,
            "target": [0, 1] * 5,
        }
    )
    ensure_output_dirs()
    df.to_csv(CLEANED_DATA_PATH, index=False)

    stale = split_ws.load_state()
    stale["stage"] = "validated_split"
    stale["finalized"] = True
    stale["target_column"] = "target"
    split_ws.save_state(stale)

    with pytest.raises(SplitError, match="numeric"):
        validate_cleaned_data("target")

    assert split_ws.load_state()["stage"] == "uninitialized"
