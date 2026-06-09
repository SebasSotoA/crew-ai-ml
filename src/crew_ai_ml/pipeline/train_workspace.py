"""Shared workspace for agentic model training steps."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from crewai.tasks.task_output import TaskOutput

from crew_ai_ml.pipeline.config import (
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    TRAIN_CANDIDATES_DIR,
    TRAIN_DATA_PATH,
    TRAIN_PROFILE_PATH,
    TRAIN_STATE_PATH,
    TRAINING_LOG_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.prep_workspace import get_kickoff_target_column
from crew_ai_ml.pipeline.validators import feature_null_report, null_report_message

__all__ = [
    "MAX_CANDIDATES",
    "TrainWorkspaceError",
    "get_kickoff_target_column",
]

MAX_CANDIDATES = 5


class TrainWorkspaceError(Exception):
    """Raised when training workspace operations fail."""


def _default_state() -> dict[str, Any]:
    return {
        "target_column": None,
        "stage": "uninitialized",
        "steps_applied": [],
        "decisions": [],
        "candidates": [],
        "best_candidate_id": None,
        "finalized": False,
    }


def load_state() -> dict[str, Any]:
    ensure_output_dirs()
    if not TRAIN_STATE_PATH.exists():
        return _default_state()
    with TRAIN_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    ensure_output_dirs()
    TRAIN_STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def reset_workspace(target_column: str) -> dict[str, Any]:
    ensure_output_dirs()
    state = _default_state()
    state["target_column"] = target_column
    save_state(state)
    if TRAIN_PROFILE_PATH.exists():
        TRAIN_PROFILE_PATH.unlink()
    if TRAIN_CANDIDATES_DIR.exists():
        shutil.rmtree(TRAIN_CANDIDATES_DIR, ignore_errors=True)
    TRAIN_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    return state


def write_profile(profile: dict[str, Any]) -> None:
    ensure_output_dirs()
    TRAIN_PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def read_profile() -> dict[str, Any]:
    if not TRAIN_PROFILE_PATH.exists():
        raise TrainWorkspaceError("Profile not found. Call profile_train_data first.")
    with TRAIN_PROFILE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def candidate_artifact_path(candidate_id: str) -> Path:
    return TRAIN_CANDIDATES_DIR / f"{candidate_id}.joblib"


def save_candidate(candidate_id: str, bundle: dict[str, Any]) -> Path:
    ensure_output_dirs()
    TRAIN_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path = candidate_artifact_path(candidate_id)
    joblib.dump(bundle, path)
    return path


def load_candidate(candidate_id: str) -> dict[str, Any]:
    path = candidate_artifact_path(candidate_id)
    if not path.exists():
        raise TrainWorkspaceError(f"Candidate '{candidate_id}' not found at {path}")
    return joblib.load(path)


def _next_candidate_id(state: dict[str, Any]) -> str:
    existing = {c["id"] for c in state.get("candidates", [])}
    index = 1
    while f"candidate_{index}" in existing:
        index += 1
    return f"candidate_{index}"


def register_candidate(metadata: dict[str, Any]) -> dict[str, Any]:
    """Append candidate metadata to state; enforce MAX_CANDIDATES cap."""
    state = load_state()
    candidates = state.get("candidates", [])
    if len(candidates) >= MAX_CANDIDATES:
        raise TrainWorkspaceError(
            f"Maximum of {MAX_CANDIDATES} training candidates reached. "
            "Call finalize_training or remove candidates before adding more."
        )
    candidate_id = metadata.get("id") or _next_candidate_id(state)
    entry = dict(metadata)
    entry["id"] = candidate_id
    entry.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    candidates.append(entry)
    state["candidates"] = candidates
    save_state(state)
    return entry


def list_candidates() -> list[dict[str, Any]]:
    return list(load_state().get("candidates", []))


def set_best_candidate(candidate_id: str) -> None:
    state = load_state()
    ids = {c["id"] for c in state.get("candidates", [])}
    if candidate_id not in ids:
        raise TrainWorkspaceError(f"Unknown candidate id '{candidate_id}'")
    state["best_candidate_id"] = candidate_id
    save_state(state)


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


def validate_training_artifacts() -> tuple[bool, str]:
    """Validate required training outputs exist and are minimally valid."""
    errors: list[str] = []

    if not MODEL_PATH.is_file():
        errors.append(f"Missing {MODEL_PATH}")
    if not TRAINING_LOG_PATH.is_file():
        errors.append(f"Missing {TRAINING_LOG_PATH}")

    state = load_state()
    target_column = state.get("target_column")

    if MODEL_PATH.is_file():
        try:
            bundle = joblib.load(MODEL_PATH)
            required = {"model", "label_encoder", "feature_columns", "target_column"}
            missing = required - set(bundle.keys())
            if missing:
                errors.append(f"model.pkl missing keys: {sorted(missing)}")
            elif target_column and bundle.get("target_column") != target_column:
                errors.append(
                    f"model.pkl target '{bundle.get('target_column')}' "
                    f"!= workspace target '{target_column}'"
                )
        except Exception as exc:
            errors.append(f"model.pkl unreadable: {exc}")

    if not TRAIN_DATA_PATH.is_file():
        errors.append(f"Missing {TRAIN_DATA_PATH}")
    elif target_column:
        try:
            df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
            if target_column not in df.columns:
                errors.append(f"train.csv missing target column '{target_column}'")
            else:
                nulls = feature_null_report(df, target_column)
                if nulls:
                    errors.append(
                        null_report_message(
                            nulls,
                            "Re-run data preparation with impute_missing.",
                        )
                    )
        except Exception as exc:
            errors.append(f"train.csv unreadable: {exc}")

    if not FEATURE_METADATA_PATH.is_file():
        errors.append(f"Missing {FEATURE_METADATA_PATH}")

    if not state.get("finalized"):
        errors.append("finalize_training was not completed (train_state.finalized is false)")

    if not state.get("decisions"):
        errors.append(
            "No training decisions logged (call log_training_decision at least once)"
        )

    if errors:
        return False, "; ".join(errors)
    return True, "Training artifacts validated successfully."


def validate_training_task_output(result: TaskOutput):
    """CrewAI task guardrail for model training task completion."""
    ok, message = validate_training_artifacts()
    if not ok:
        return False, message
    return True, result.raw
