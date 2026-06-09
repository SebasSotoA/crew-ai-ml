"""Shared workspace for agentic model evaluation steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
from crewai.tasks.task_output import TaskOutput

from crew_ai_ml.pipeline.config import (
    EVAL_PROFILE_PATH,
    EVAL_REPORT_PATH,
    EVAL_STATE_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.prep_workspace import get_kickoff_target_column

__all__ = [
    "EvalWorkspaceError",
    "get_kickoff_target_column",
    "load_state",
    "log_decision",
    "log_step",
    "read_profile",
    "reset_workspace",
    "save_state",
    "validate_evaluation_artifacts",
    "validate_evaluation_task_output",
    "write_profile",
]

VALID_VERDICTS = frozenset({"DEPLOY", "DO NOT DEPLOY"})

EVAL_STAGES = (
    "uninitialized",
    "validated",
    "profiled",
    "metrics_computed",
    "plots_generated",
    "signals_analyzed",
    "verdict_issued",
    "finalized",
)


class EvalWorkspaceError(Exception):
    """Raised when evaluation workspace operations fail."""


def _default_state() -> dict[str, Any]:
    return {
        "target_column": None,
        "stage": "uninitialized",
        "steps_applied": [],
        "decisions": [],
        "metrics": {"train": None, "test": None},
        "signals": [],
        "plots": {},
        "verdict": None,
        "rationale": "",
        "finalized": False,
    }


def load_state() -> dict[str, Any]:
    ensure_output_dirs()
    if not EVAL_STATE_PATH.exists():
        return _default_state()
    with EVAL_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    ensure_output_dirs()
    EVAL_STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def reset_workspace(target_column: str) -> dict[str, Any]:
    ensure_output_dirs()
    state = _default_state()
    state["target_column"] = target_column
    save_state(state)
    if EVAL_PROFILE_PATH.exists():
        EVAL_PROFILE_PATH.unlink()
    return state


def write_profile(profile: dict[str, Any]) -> None:
    ensure_output_dirs()
    EVAL_PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")


def read_profile() -> dict[str, Any]:
    if not EVAL_PROFILE_PATH.exists():
        raise EvalWorkspaceError("Profile not found. Call profile_eval_data first.")
    with EVAL_PROFILE_PATH.open(encoding="utf-8") as f:
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


def _report_contains_verdict(text: str) -> bool:
    return "**DEPLOY**" in text or "**DO NOT DEPLOY**" in text


def validate_evaluation_artifacts() -> tuple[bool, str]:
    """Validate required evaluation outputs exist and are minimally valid."""
    errors: list[str] = []
    state = load_state()

    if not state.get("finalized"):
        errors.append("finalize_evaluation was not completed (eval_state.finalized is false)")

    verdict = state.get("verdict")
    if verdict not in VALID_VERDICTS:
        errors.append(
            f"eval_state.verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}"
        )

    rationale = state.get("rationale")
    if not rationale or not str(rationale).strip():
        errors.append("eval_state.rationale is empty")

    if not EVAL_REPORT_PATH.is_file():
        errors.append(f"Missing {EVAL_REPORT_PATH}")
    if not EVALUATION_REPORT_PATH.is_file():
        errors.append(f"Missing {EVALUATION_REPORT_PATH}")

    report_path = EVAL_REPORT_PATH if EVAL_REPORT_PATH.is_file() else EVALUATION_REPORT_PATH
    if report_path.is_file():
        try:
            report_text = report_path.read_text(encoding="utf-8")
            if not _report_contains_verdict(report_text):
                errors.append(
                    f"{report_path} must contain **DEPLOY** or **DO NOT DEPLOY**"
                )
        except Exception as exc:
            errors.append(f"{report_path} unreadable: {exc}")

    confusion_path = PLOTS_DIR / "confusion_matrix.png"
    if not confusion_path.is_file():
        errors.append(f"Missing {confusion_path}")

    if not MODEL_PATH.is_file():
        errors.append(f"Missing {MODEL_PATH}")
    else:
        try:
            bundle = joblib.load(MODEL_PATH)
            eval_metrics = bundle.get("evaluation_metrics")
            if not isinstance(eval_metrics, dict):
                errors.append("model.pkl missing evaluation_metrics dict")
            else:
                if "verdict" not in eval_metrics:
                    errors.append("model.pkl evaluation_metrics missing verdict")
                elif eval_metrics["verdict"] not in VALID_VERDICTS:
                    errors.append(
                        f"model.pkl evaluation_metrics.verdict invalid: {eval_metrics['verdict']!r}"
                    )
                if "f1_weighted" not in eval_metrics:
                    errors.append("model.pkl evaluation_metrics missing f1_weighted")
        except Exception as exc:
            errors.append(f"model.pkl unreadable: {exc}")

    if errors:
        return False, "; ".join(errors)
    return True, "Evaluation artifacts validated successfully."


def validate_evaluation_task_output(result: TaskOutput):
    """CrewAI task guardrail for model evaluation task completion."""
    ok, message = validate_evaluation_artifacts()
    if not ok:
        return False, message
    return True, result.raw
