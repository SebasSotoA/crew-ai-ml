"""Shared workspace for agentic data preparation steps."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from crewai.tasks.task_output import TaskOutput

import pandas as pd

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    FEATURE_METADATA_PATH,
    PREP_PRE_ENCODE_PATH,
    PREP_PROFILE_PATH,
    PREP_STATE_PATH,
    PREP_WORKING_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.validators import feature_null_report, null_report_message


class PrepWorkspaceError(Exception):
    """Raised when prep workspace operations fail."""


WORKSPACE_LOCK_TIMEOUT = 30.0

_workspace_lock = threading.RLock()
_working_df: pd.DataFrame | None = None
_pre_encode_df: pd.DataFrame | None = None

_kickoff_dataset_path: str | None = None
_kickoff_target_column: str | None = None


def set_kickoff_inputs(dataset_path: str | None, target_column: str | None) -> None:
    """Store crew kickoff paths so tools can fall back if the agent omits them."""
    global _kickoff_dataset_path, _kickoff_target_column
    _kickoff_dataset_path = dataset_path.strip() if dataset_path else None
    _kickoff_target_column = target_column.strip() if target_column else None


def get_kickoff_dataset_path() -> str | None:
    return _kickoff_dataset_path


def get_kickoff_target_column() -> str | None:
    return _kickoff_target_column


@contextmanager
def prep_tool_guard() -> Iterator[None]:
    """Serialize prep tool execution so only one runs at a time."""
    acquired = _workspace_lock.acquire(timeout=WORKSPACE_LOCK_TIMEOUT)
    if not acquired:
        raise PrepWorkspaceError(
            "Another prep tool is still running. Call one prep tool at a time."
        )
    try:
        yield
    finally:
        _workspace_lock.release()


def _default_state() -> dict[str, Any]:
    return {
        "dataset_path": None,
        "target_column": None,
        "stage": "uninitialized",
        "dummy_groups": {},
        "label_classes": [],
        "steps_applied": [],
        "decisions": [],
        "dropped_columns": [],
        "finalized": False,
    }


def load_state() -> dict[str, Any]:
    ensure_output_dirs()
    if not PREP_STATE_PATH.exists():
        return _default_state()
    with PREP_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    ensure_output_dirs()
    PREP_STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _invalidate_downstream_artifacts(target_column: str | None = None) -> None:
    """Remove stale prep/split outputs so a fresh prep run cannot mask failures."""
    for path in (CLEANED_DATA_PATH, FEATURE_METADATA_PATH):
        if path.exists():
            path.unlink()

    from crew_ai_ml.pipeline import split_workspace as split_ws

    split_ws.reset_workspace(target_column or "")


def _atomic_csv_write(df: pd.DataFrame, path: Path) -> None:
    """Write CSV in place with row-count verify; retry on Windows file locks."""
    ensure_output_dirs()
    expected_rows = len(df)
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            df.to_csv(path, index=False)
            line_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
            if line_count != expected_rows:
                raise OSError(
                    f"CSV row count mismatch after write: expected {expected_rows}, "
                    f"got {line_count}"
                )
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))

    raise PrepWorkspaceError(
        f"Failed to write {path} after retries: {last_error}"
    ) from last_error


def reset_workspace(dataset_path: str, target_column: str) -> dict[str, Any]:
    global _working_df, _pre_encode_df
    _working_df = None
    _pre_encode_df = None
    ensure_output_dirs()
    state = _default_state()
    state["dataset_path"] = dataset_path
    state["target_column"] = target_column
    state["stage"] = "raw"
    save_state(state)
    for path in (PREP_WORKING_PATH, PREP_PRE_ENCODE_PATH, PREP_PROFILE_PATH):
        if path.exists():
            path.unlink()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    _invalidate_downstream_artifacts(target_column)
    return state


def read_working() -> pd.DataFrame:
    global _working_df
    if _working_df is not None:
        return _working_df.copy()
    if not PREP_WORKING_PATH.exists():
        raise PrepWorkspaceError(
            "Working dataset not found. Call load_dataset before other prep tools."
        )
    _working_df = pd.read_csv(PREP_WORKING_PATH, low_memory=False)
    return _working_df.copy()


def write_working(df: pd.DataFrame) -> None:
    global _working_df
    _working_df = df.copy()
    _atomic_csv_write(df, PREP_WORKING_PATH)


def verify_working_write(expected_rows: int) -> None:
    """Assert row count matches what was just written (memory or disk)."""
    if _working_df is not None and len(_working_df) == expected_rows:
        return
    actual = read_working()
    if len(actual) != expected_rows:
        raise PrepWorkspaceError(
            f"working.csv row count mismatch after write: expected {expected_rows}, "
            f"got {len(actual)}. Re-run load_dataset to reset the workspace."
        )


def read_pre_encode() -> pd.DataFrame:
    global _pre_encode_df
    if _pre_encode_df is not None:
        return _pre_encode_df.copy()
    if not PREP_PRE_ENCODE_PATH.exists():
        raise PrepWorkspaceError(
            "Pre-encode snapshot not found. Call encode_features before finalize."
        )
    _pre_encode_df = pd.read_csv(PREP_PRE_ENCODE_PATH, low_memory=False)
    return _pre_encode_df.copy()


def write_pre_encode(df: pd.DataFrame) -> None:
    global _pre_encode_df
    _pre_encode_df = df.copy()
    _atomic_csv_write(df, PREP_PRE_ENCODE_PATH)


def validate_working_df(
    df: pd.DataFrame,
    target_column: str,
    *,
    require_numeric_features: bool = False,
) -> None:
    """Validate working dataframe structure before persisting downstream artifacts."""
    if target_column not in df.columns:
        raise PrepWorkspaceError(f"Target column '{target_column}' missing from working data.")

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise PrepWorkspaceError("No feature columns found in working data.")

    if not require_numeric_features:
        return

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise PrepWorkspaceError(
            f"All features must be numeric after encoding. Non-numeric: {non_numeric}"
        )


def write_profile(profile: dict[str, Any]) -> None:
    ensure_output_dirs()
    PREP_PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def read_profile() -> dict[str, Any]:
    if not PREP_PROFILE_PATH.exists():
        raise PrepWorkspaceError("Profile not found. Call profile_dataset first.")
    with PREP_PROFILE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def log_step(
    tool: str,
    params: dict[str, Any],
    rows_before: int,
    rows_after: int,
    summary: str,
) -> dict[str, Any]:
    state = load_state()
    entry = {
        "tool": tool,
        "params": params,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state["steps_applied"].append(entry)
    save_state(state)
    return entry


def log_decision(issue: str, options_considered: list[str], choice: str, rationale: str) -> None:
    state = load_state()
    state["decisions"].append(
        {
            "issue": issue,
            "options_considered": options_considered,
            "choice": choice,
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_state(state)


def validate_prep_artifacts() -> tuple[bool, str]:
    """Validate required prep outputs exist and are minimally valid."""
    errors: list[str] = []

    if not CLEANED_DATA_PATH.is_file():
        errors.append(f"Missing {CLEANED_DATA_PATH}")
    if not FEATURE_METADATA_PATH.is_file():
        errors.append(f"Missing {FEATURE_METADATA_PATH}")

    state = load_state()
    target_column = state.get("target_column")

    if CLEANED_DATA_PATH.is_file():
        try:
            df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
            if df.empty:
                errors.append("cleaned_data.csv is empty")
            elif target_column and target_column in df.columns:
                feature_cols = [c for c in df.columns if c != target_column]
                non_numeric = [
                    c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
                ]
                if non_numeric:
                    errors.append(
                        "cleaned_data.csv has non-numeric feature columns: "
                        f"{non_numeric}. Re-run encode_features and finalize_preparation."
                    )
                nulls = feature_null_report(df, target_column)
                if nulls:
                    errors.append(
                        null_report_message(
                            nulls,
                            "Call impute_missing before finalize_preparation.",
                        )
                    )
        except Exception as exc:
            errors.append(f"cleaned_data.csv unreadable: {exc}")

    if FEATURE_METADATA_PATH.is_file():
        try:
            with FEATURE_METADATA_PATH.open(encoding="utf-8") as f:
                schema = json.load(f)
            if not isinstance(schema, list) or not schema:
                errors.append("feature_metadata.json must be a non-empty list")
        except Exception as exc:
            errors.append(f"feature_metadata.json invalid: {exc}")

    if not state.get("finalized"):
        errors.append("finalize_preparation was not completed (prep_state.finalized is false)")

    if errors:
        return False, "; ".join(errors)
    return True, "Prep artifacts validated successfully."


def validate_prep_task_output(result: TaskOutput):
    """CrewAI task guardrail for data preparation task completion."""
    ok, message = validate_prep_artifacts()
    if not ok:
        return False, message
    return True, result.raw
