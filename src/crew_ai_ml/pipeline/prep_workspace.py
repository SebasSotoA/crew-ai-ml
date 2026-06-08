"""Shared workspace for agentic data preparation steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

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


class PrepWorkspaceError(Exception):
    """Raised when prep workspace operations fail."""


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


def reset_workspace(dataset_path: str, target_column: str) -> dict[str, Any]:
    ensure_output_dirs()
    state = _default_state()
    state["dataset_path"] = dataset_path
    state["target_column"] = target_column
    state["stage"] = "raw"
    save_state(state)
    if PREP_WORKING_PATH.exists():
        PREP_WORKING_PATH.unlink()
    if PREP_PRE_ENCODE_PATH.exists():
        PREP_PRE_ENCODE_PATH.unlink()
    if PREP_PROFILE_PATH.exists():
        PREP_PROFILE_PATH.unlink()
    return state


def read_working() -> pd.DataFrame:
    if not PREP_WORKING_PATH.exists():
        raise PrepWorkspaceError(
            "Working dataset not found. Call load_dataset before other prep tools."
        )
    return pd.read_csv(PREP_WORKING_PATH)


def write_working(df: pd.DataFrame) -> None:
    ensure_output_dirs()
    df.to_csv(PREP_WORKING_PATH, index=False)


def read_pre_encode() -> pd.DataFrame:
    if not PREP_PRE_ENCODE_PATH.exists():
        raise PrepWorkspaceError(
            "Pre-encode snapshot not found. Call encode_features before finalize."
        )
    return pd.read_csv(PREP_PRE_ENCODE_PATH)


def write_pre_encode(df: pd.DataFrame) -> None:
    ensure_output_dirs()
    df.to_csv(PREP_PRE_ENCODE_PATH, index=False)


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

    if CLEANED_DATA_PATH.is_file():
        try:
            df = pd.read_csv(CLEANED_DATA_PATH)
            if df.empty:
                errors.append("cleaned_data.csv is empty")
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

    state = load_state()
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
