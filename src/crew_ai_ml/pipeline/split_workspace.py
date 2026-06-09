"""Shared workspace for agentic train/test split steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from crewai.tasks.task_output import TaskOutput

from crew_ai_ml.pipeline.config import (
    SPLIT_PROFILE_PATH,
    SPLIT_STATE_PATH,
    SPLIT_TEST_HOLD_PATH,
    SPLIT_TRAIN_HOLD_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.prep_workspace import get_kickoff_target_column

__all__ = ["SplitWorkspaceError", "get_kickoff_target_column"]


class SplitWorkspaceError(Exception):
    """Raised when split workspace operations fail."""


def _default_state() -> dict[str, Any]:
    return {
        "target_column": None,
        "stage": "uninitialized",
        "steps_applied": [],
        "decisions": [],
        "finalized": False,
        "test_size": None,
        "stratify": None,
        "random_state": None,
    }


def load_state() -> dict[str, Any]:
    ensure_output_dirs()
    if not SPLIT_STATE_PATH.exists():
        return _default_state()
    with SPLIT_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    ensure_output_dirs()
    SPLIT_STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def reset_workspace(target_column: str) -> dict[str, Any]:
    ensure_output_dirs()
    state = _default_state()
    state["target_column"] = target_column
    save_state(state)
    for path in (SPLIT_TRAIN_HOLD_PATH, SPLIT_TEST_HOLD_PATH, SPLIT_PROFILE_PATH):
        if path.exists():
            path.unlink()
    return state


def read_train_hold() -> pd.DataFrame:
    if not SPLIT_TRAIN_HOLD_PATH.exists():
        raise SplitWorkspaceError(
            "Train hold not found. Call split_train_test before reading holds."
        )
    return pd.read_csv(SPLIT_TRAIN_HOLD_PATH)


def write_train_hold(df: pd.DataFrame) -> None:
    ensure_output_dirs()
    df.to_csv(SPLIT_TRAIN_HOLD_PATH, index=False)


def read_test_hold() -> pd.DataFrame:
    if not SPLIT_TEST_HOLD_PATH.exists():
        raise SplitWorkspaceError(
            "Test hold not found. Call split_train_test before reading holds."
        )
    return pd.read_csv(SPLIT_TEST_HOLD_PATH)


def write_test_hold(df: pd.DataFrame) -> None:
    ensure_output_dirs()
    df.to_csv(SPLIT_TEST_HOLD_PATH, index=False)


def write_profile(profile: dict[str, Any]) -> None:
    ensure_output_dirs()
    SPLIT_PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def read_profile() -> dict[str, Any]:
    if not SPLIT_PROFILE_PATH.exists():
        raise SplitWorkspaceError("Profile not found. Call profile_split_data first.")
    with SPLIT_PROFILE_PATH.open(encoding="utf-8") as f:
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


def validate_split_artifacts() -> tuple[bool, str]:
    """Validate required split outputs exist and are minimally valid."""
    errors: list[str] = []

    if not TRAIN_DATA_PATH.is_file():
        errors.append(f"Missing {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.is_file():
        errors.append(f"Missing {TEST_DATA_PATH}")

    state = load_state()
    target_column = state.get("target_column")

    for label, path in (("train", TRAIN_DATA_PATH), ("test", TEST_DATA_PATH)):
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                errors.append(f"{label}.csv is empty")
            elif target_column and target_column not in df.columns:
                errors.append(f"{label}.csv missing target column '{target_column}'")
        except Exception as exc:
            errors.append(f"{label}.csv unreadable: {exc}")

    if not state.get("finalized"):
        errors.append("finalize_split was not completed (split_state.finalized is false)")

    if errors:
        return False, "; ".join(errors)
    return True, "Split artifacts validated successfully."


def validate_split_task_output(result: TaskOutput):
    """CrewAI task guardrail for data split task completion."""
    ok, message = validate_split_artifacts()
    if not ok:
        return False, message
    return True, result.raw
