"""Shared workspace for agentic deployment steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from crewai.tasks.task_output import TaskOutput

from crew_ai_ml.pipeline.config import (
    DEPLOY_PROFILE_PATH,
    DEPLOY_REPORT_PATH,
    DEPLOY_REQUIREMENTS_PATH,
    DEPLOY_STATE_PATH,
    EVALUATION_REPORT_PATH,
    FAILURE_REPORT_PATH,
    MODEL_PATH,
    STREAMLIT_APP_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.prep_workspace import get_kickoff_target_column

__all__ = [
    "DeployWorkspaceError",
    "get_kickoff_target_column",
    "load_state",
    "log_decision",
    "log_step",
    "read_profile",
    "reset_workspace",
    "save_state",
    "validate_deployment_artifacts",
    "validate_deployment_task_output",
    "write_profile",
]

VALID_VERDICTS = frozenset({"DEPLOY", "DO NOT DEPLOY"})


class DeployWorkspaceError(Exception):
    """Raised when deployment workspace operations fail."""


def _default_state() -> dict[str, Any]:
    return {
        "target_column": None,
        "stage": "uninitialized",
        "verdict": None,
        "branch": None,
        "steps_applied": [],
        "decisions": [],
        "ui_config": {},
        "artifacts": {},
        "finalized": False,
    }


def load_state() -> dict[str, Any]:
    ensure_output_dirs()
    if not DEPLOY_STATE_PATH.exists():
        return _default_state()
    with DEPLOY_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    ensure_output_dirs()
    DEPLOY_STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def reset_workspace(target_column: str, verdict: str | None = None) -> dict[str, Any]:
    ensure_output_dirs()
    state = _default_state()
    state["target_column"] = target_column
    if verdict is not None:
        state["verdict"] = verdict
        state["branch"] = "deploy" if verdict == "DEPLOY" else "do_not_deploy"
    save_state(state)
    if DEPLOY_PROFILE_PATH.exists():
        DEPLOY_PROFILE_PATH.unlink()
    return state


def write_profile(profile: dict[str, Any]) -> None:
    ensure_output_dirs()
    DEPLOY_PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def read_profile() -> dict[str, Any]:
    if not DEPLOY_PROFILE_PATH.exists():
        raise DeployWorkspaceError("Profile not found. Call profile_deploy_context first.")
    with DEPLOY_PROFILE_PATH.open(encoding="utf-8") as f:
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


def validate_deployment_artifacts() -> tuple[bool, str]:
    """Validate required deployment outputs exist and are minimally valid."""
    errors: list[str] = []
    state = load_state()

    if not state.get("finalized"):
        errors.append("finalize_deployment was not completed (deploy_state.finalized is false)")

    verdict = state.get("verdict")
    if verdict not in VALID_VERDICTS:
        errors.append(
            f"deploy_state.verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}"
        )

    if not DEPLOY_REPORT_PATH.is_file():
        errors.append(f"Missing {DEPLOY_REPORT_PATH}")

    if not EVALUATION_REPORT_PATH.is_file():
        errors.append(f"Missing {EVALUATION_REPORT_PATH}")

    if not MODEL_PATH.is_file():
        errors.append(f"Missing {MODEL_PATH}")

    branch = state.get("branch")
    if verdict == "DEPLOY" or branch == "deploy":
        if not STREAMLIT_APP_PATH.is_file():
            errors.append(f"Missing {STREAMLIT_APP_PATH}")
        if not DEPLOY_REQUIREMENTS_PATH.is_file():
            errors.append(f"Missing {DEPLOY_REQUIREMENTS_PATH}")
        if FAILURE_REPORT_PATH.is_file():
            errors.append(
                f"Unexpected {FAILURE_REPORT_PATH} on DEPLOY branch"
            )
    elif verdict == "DO NOT DEPLOY" or branch in {"reject", "do_not_deploy"}:
        if not FAILURE_REPORT_PATH.is_file():
            errors.append(f"Missing {FAILURE_REPORT_PATH}")
        if STREAMLIT_APP_PATH.is_file():
            errors.append(
                f"Unexpected {STREAMLIT_APP_PATH} on DO NOT DEPLOY branch"
            )
        if DEPLOY_REQUIREMENTS_PATH.is_file():
            errors.append(
                f"Unexpected {DEPLOY_REQUIREMENTS_PATH} on DO NOT DEPLOY branch"
            )

    if errors:
        return False, "; ".join(errors)
    return True, "Deployment artifacts validated successfully."


def validate_deployment_task_output(result: TaskOutput):
    """CrewAI task guardrail for deployment task completion."""
    ok, message = validate_deployment_artifacts()
    if not ok:
        return False, message
    return True, result.raw
