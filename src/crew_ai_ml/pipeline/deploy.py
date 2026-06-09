"""Deployment pipeline: atomic deployment steps."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import joblib

from crew_ai_ml.pipeline import deploy_workspace as ws
from crew_ai_ml.pipeline.config import (
    DEPLOY_REPORT_PATH,
    DEPLOY_REQUIREMENTS_PATH,
    DEPLOY_STATE_PATH,
    EVAL_STATE_PATH,
    EVALUATION_REPORT_PATH,
    FAILURE_REPORT_PATH,
    INFERENCE_UTILS_PATH,
    MODEL_PATH,
    STREAMLIT_APP_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.streamlit_app_builder import (
    build_inference_utils_source,
    build_streamlit_app_source,
)

__all__ = [
    "DeploymentError",
    "configure_app_ui",
    "document_deploy_failure",
    "finalize_deployment",
    "generate_streamlit_app",
    "profile_deploy_context",
    "validate_deploy_inputs",
    "write_deploy_requirements",
]

DEFAULT_PACKAGES = [
    "streamlit>=1.28.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.0",
]

UI_CONFIG_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "page_title": str,
    "page_caption": str,
    "page_icon": str,
    "confidence_threshold": (int, float),
    "tab_manual_label": str,
    "tab_csv_label": str,
    "sidebar_metrics": bool,
    "show_probability_table": bool,
    "show_metrics_banner": bool,
    "disclaimer_text": str,
}

DEFAULT_UI_CONFIG: dict[str, Any] = {
    "page_title": "ML Prediction App",
    "page_caption": "Deployed from crew_ai_ml pipeline",
    "page_icon": "🫀",
    "confidence_threshold": 0.60,
    "tab_manual_label": "Manual Input",
    "tab_csv_label": "CSV Upload",
    "sidebar_metrics": True,
    "show_probability_table": True,
    "show_metrics_banner": True,
    "disclaimer_text": "Predictions are estimates, not certainties.",
}


class DeploymentError(Exception):
    """Raised when deployment preparation fails."""


def _parse_verdict(report_text: str) -> str:
    match = re.search(r"\*\*(DEPLOY|DO NOT DEPLOY)\*\*", report_text)
    if match:
        return match.group(1)

    if "DO NOT DEPLOY" in report_text.upper():
        return "DO NOT DEPLOY"
    if "DEPLOY" in report_text.upper():
        return "DEPLOY"

    raise DeploymentError(
        "Could not parse deployment verdict from evaluation_report.md. "
        "Expected '**DEPLOY**' or '**DO NOT DEPLOY**'."
    )


def _extract_section_lines(report_text: str, heading: str) -> list[str]:
    lines = report_text.splitlines()
    collected: list[str] = []
    in_section = False

    for line in lines:
        if line.strip().startswith("## ") and heading.lower() in line.lower():
            in_section = True
            continue
        if in_section and line.strip().startswith("## "):
            break
        if in_section:
            collected.append(line)

    return collected


def _require_target(target_column: str) -> str:
    requested = target_column.strip() if target_column else ""
    if not requested:
        kickoff = ws.get_kickoff_target_column()
        if kickoff:
            requested = kickoff
    if not requested:
        raise DeploymentError(
            "target_column must be a non-empty string (tool arg or crew kickoff inputs)."
        )
    return requested.strip()


def _require_stage(actual: str | None, expected: str, tool_name: str) -> None:
    if actual != expected:
        raise DeploymentError(
            f"Expected stage '{expected}', got '{actual}'. "
            f"Complete prior deployment steps before calling {tool_name}."
        )


def _load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise DeploymentError(
            f"Trained model not found at {MODEL_PATH}. Run training and evaluation first."
        )
    return joblib.load(MODEL_PATH)


def _load_eval_state() -> dict[str, Any]:
    if not EVAL_STATE_PATH.exists():
        raise DeploymentError(
            f"Evaluation state not found at {EVAL_STATE_PATH}. Run evaluation first."
        )
    with EVAL_STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_verdict(eval_state: dict[str, Any], report_text: str) -> str:
    verdict = eval_state.get("verdict")
    if verdict in ws.VALID_VERDICTS:
        return str(verdict)
    return _parse_verdict(report_text)


def _algorithm_display_name(algorithm: str | None) -> str:
    if not algorithm:
        return "ML"
    return algorithm.replace("_", " ").title()


def _summarize_input_schema(input_schema: Any) -> dict[str, Any]:
    if not isinstance(input_schema, list):
        return {"field_count": 0, "numeric_fields": [], "categorical_fields": []}

    numeric_fields = [f["name"] for f in input_schema if f.get("type") == "numeric"]
    categorical_fields = [
        {
            "name": f["name"],
            "category_count": len(f.get("categories", [])),
        }
        for f in input_schema
        if f.get("type") == "categorical"
    ]
    return {
        "field_count": len(input_schema),
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
    }


def _parse_ui_config(ui_config: dict[str, Any] | str | None) -> dict[str, Any]:
    if ui_config is None:
        return dict(DEFAULT_UI_CONFIG)

    parsed: dict[str, Any]
    if isinstance(ui_config, str):
        if not ui_config.strip():
            return dict(DEFAULT_UI_CONFIG)
        try:
            loaded = json.loads(ui_config)
        except json.JSONDecodeError as exc:
            raise DeploymentError(f"ui_config must be valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise DeploymentError("ui_config JSON must decode to an object.")
        parsed = loaded
    else:
        parsed = dict(ui_config)

    merged = dict(DEFAULT_UI_CONFIG)
    merged.update(parsed)

    for key, expected_type in UI_CONFIG_SCHEMA.items():
        value = merged.get(key)
        if value is None:
            continue
        if not isinstance(value, expected_type):
            type_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " or ".join(t.__name__ for t in expected_type)
            )
            raise DeploymentError(
                f"ui_config['{key}'] must be {type_name}, got {type(value).__name__}"
            )

    threshold = float(merged["confidence_threshold"])
    if not 0.0 < threshold < 1.0:
        raise DeploymentError("ui_config['confidence_threshold'] must be between 0 and 1.")

    return merged


def _build_ui_recommendations(
    *,
    target_column: str,
    algorithm: str | None,
    has_predict_proba: bool,
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    algorithm_name = _algorithm_display_name(algorithm)
    precision = eval_metrics.get("precision_weighted")
    f1 = eval_metrics.get("f1_weighted")

    disclaimer = DEFAULT_UI_CONFIG["disclaimer_text"]
    if precision is not None and f1 is not None:
        disclaimer = (
            f"Model test precision (weighted): {precision:.1%}. "
            f"F1: {f1:.1%}. Predictions are estimates, not certainties."
        )
    elif f1 is not None:
        disclaimer = (
            f"Model test F1 (weighted): {f1:.1%}. "
            "Predictions are estimates, not certainties."
        )

    return {
        "page_title": f"{algorithm_name} — {target_column} Prediction",
        "page_caption": f"Binary classifier for `{target_column}` (algorithm: {algorithm or 'unknown'})",
        "page_icon": "🫀",
        "confidence_threshold": 0.60,
        "tab_manual_label": "Manual Input",
        "tab_csv_label": "CSV Upload",
        "sidebar_metrics": True,
        "show_probability_table": has_predict_proba,
        "show_metrics_banner": True,
        "disclaimer_text": disclaimer,
    }


def validate_deploy_inputs(target_column: str) -> dict[str, Any]:
    """Validate evaluation artifacts and reset the deployment workspace."""
    ensure_output_dirs()
    target_column = _require_target(target_column)

    if not EVALUATION_REPORT_PATH.exists():
        raise DeploymentError(
            f"Evaluation report not found at {EVALUATION_REPORT_PATH}. Run evaluation first."
        )

    eval_state = _load_eval_state()
    if not eval_state.get("finalized"):
        raise DeploymentError(
            "Evaluation was not finalized. Call finalize_evaluation before deployment."
        )

    if eval_state.get("target_column") and eval_state["target_column"] != target_column:
        raise DeploymentError(
            f"Evaluation target '{eval_state['target_column']}' "
            f"does not match requested target '{target_column}'."
        )

    bundle = _load_model_bundle()
    if bundle.get("target_column") != target_column:
        raise DeploymentError(
            f"Model target '{bundle.get('target_column')}' "
            f"does not match requested target '{target_column}'."
        )

    report_text = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    verdict = _resolve_verdict(eval_state, report_text)

    ws.reset_workspace(target_column)
    state = ws.load_state()
    state["stage"] = "validated"
    state["verdict"] = verdict
    state["branch"] = "deploy" if verdict == "DEPLOY" else "do_not_deploy"
    ws.save_state(state)

    ws.log_step(
        "validate_deploy_inputs",
        {"target_column": target_column, "verdict": verdict},
        0,
        0,
        f"Validated deployment inputs; verdict={verdict}",
    )
    return {
        "target_column": target_column,
        "verdict": verdict,
        "branch": state["branch"],
        "model_path": str(MODEL_PATH),
        "evaluation_report_path": str(EVALUATION_REPORT_PATH),
        "stage": "validated",
    }


def profile_deploy_context(target_column: str) -> dict[str, Any]:
    """Profile deployment context: verdict, algorithm, schema, metrics, UI hints."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "validated", "profile_deploy_context")

    bundle = _load_model_bundle()
    model = bundle["model"]
    algorithm = bundle.get("algorithm")
    input_schema = bundle.get("input_schema")
    eval_metrics = bundle.get("evaluation_metrics") or {}
    has_predict_proba = hasattr(model, "predict_proba")
    verdict = state.get("verdict") or eval_metrics.get("verdict")

    profile = {
        "target_column": target_column,
        "verdict": verdict,
        "branch": state.get("branch"),
        "algorithm": algorithm,
        "algorithm_display": _algorithm_display_name(algorithm),
        "input_schema_summary": _summarize_input_schema(input_schema),
        "evaluation_metrics": eval_metrics,
        "has_predict_proba": has_predict_proba,
        "ui_recommendations": _build_ui_recommendations(
            target_column=target_column,
            algorithm=algorithm,
            has_predict_proba=has_predict_proba,
            eval_metrics=eval_metrics if isinstance(eval_metrics, dict) else {},
        ),
    }
    ws.write_profile(profile)

    state = ws.load_state()
    state["stage"] = "profiled"
    ws.save_state(state)

    ws.log_step(
        "profile_deploy_context",
        {"target_column": target_column},
        0,
        0,
        f"Profiled deploy context; algorithm={algorithm}, predict_proba={has_predict_proba}",
    )
    return profile


def configure_app_ui(
    target_column: str,
    ui_config: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Validate and store Streamlit UI configuration (DEPLOY branch only)."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "profiled", "configure_app_ui")

    if state.get("verdict") != "DEPLOY":
        raise DeploymentError(
            f"configure_app_ui is only valid for DEPLOY verdict, got '{state.get('verdict')}'."
        )

    parsed_config = _parse_ui_config(ui_config)
    if not parsed_config.get("page_title", "").strip():
        raise DeploymentError("ui_config['page_title'] must be a non-empty string.")

    state = ws.load_state()
    state["ui_config"] = parsed_config
    state["stage"] = "ui_configured"
    ws.save_state(state)

    ws.log_decision(
        issue="streamlit_ui",
        options_considered=["agent_custom", "profile_recommendations"],
        choice="agent_custom",
        rationale=f"Configured UI: title='{parsed_config['page_title']}'",
    )
    ws.log_step(
        "configure_app_ui",
        {"target_column": target_column, "page_title": parsed_config["page_title"]},
        0,
        0,
        "Stored Streamlit UI configuration",
    )
    return {
        "target_column": target_column,
        "ui_config": parsed_config,
        "stage": "ui_configured",
    }


def generate_streamlit_app(target_column: str) -> dict[str, Any]:
    """Write output/app.py and output/inference_utils.py (DEPLOY branch only)."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "ui_configured", "generate_streamlit_app")

    if state.get("verdict") != "DEPLOY":
        raise DeploymentError(
            f"generate_streamlit_app is only valid for DEPLOY verdict, got '{state.get('verdict')}'."
        )

    ui_config = state.get("ui_config")
    if not isinstance(ui_config, dict):
        raise DeploymentError("Missing ui_config. Call configure_app_ui first.")

    ensure_output_dirs()
    inference_source = build_inference_utils_source()
    app_source = build_streamlit_app_source(ui_config)

    INFERENCE_UTILS_PATH.write_text(inference_source, encoding="utf-8")
    STREAMLIT_APP_PATH.write_text(app_source, encoding="utf-8")

    state = ws.load_state()
    state["stage"] = "app_generated"
    state["artifact_paths"] = {
        **state.get("artifact_paths", {}),
        "streamlit_app_path": str(STREAMLIT_APP_PATH),
        "inference_utils_path": str(INFERENCE_UTILS_PATH),
    }
    ws.save_state(state)

    ws.log_step(
        "generate_streamlit_app",
        {"target_column": target_column},
        0,
        0,
        f"Wrote {STREAMLIT_APP_PATH.name} and {INFERENCE_UTILS_PATH.name}",
    )
    return {
        "target_column": target_column,
        "streamlit_app_path": str(STREAMLIT_APP_PATH),
        "inference_utils_path": str(INFERENCE_UTILS_PATH),
        "stage": "app_generated",
    }


def write_deploy_requirements(
    target_column: str,
    packages: list[str] | str | None = None,
) -> dict[str, Any]:
    """Write output/requirements.txt (DEPLOY branch only)."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "app_generated", "write_deploy_requirements")

    if state.get("verdict") != "DEPLOY":
        raise DeploymentError(
            f"write_deploy_requirements is only valid for DEPLOY verdict, "
            f"got '{state.get('verdict')}'."
        )

    resolved_packages: list[str]
    if packages is None:
        resolved_packages = list(DEFAULT_PACKAGES)
    elif isinstance(packages, str):
        if not packages.strip():
            resolved_packages = list(DEFAULT_PACKAGES)
        else:
            try:
                parsed = json.loads(packages)
            except json.JSONDecodeError:
                parsed = [line.strip() for line in packages.splitlines() if line.strip()]
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                resolved_packages = parsed
            else:
                raise DeploymentError(
                    "packages must be a list of strings or JSON array of package specifiers."
                )
    else:
        if not packages:
            resolved_packages = list(DEFAULT_PACKAGES)
        elif not all(isinstance(item, str) for item in packages):
            raise DeploymentError("packages must be a list of strings.")
        else:
            resolved_packages = list(packages)

    ensure_output_dirs()
    requirements_text = "\n".join(resolved_packages) + "\n"
    DEPLOY_REQUIREMENTS_PATH.write_text(requirements_text, encoding="utf-8")

    state = ws.load_state()
    state["stage"] = "requirements_written"
    state["artifact_paths"] = {
        **state.get("artifact_paths", {}),
        "requirements_path": str(DEPLOY_REQUIREMENTS_PATH),
    }
    ws.save_state(state)

    ws.log_step(
        "write_deploy_requirements",
        {"target_column": target_column, "package_count": len(resolved_packages)},
        0,
        0,
        f"Wrote {DEPLOY_REQUIREMENTS_PATH.name} ({len(resolved_packages)} packages)",
    )
    return {
        "target_column": target_column,
        "requirements_path": str(DEPLOY_REQUIREMENTS_PATH),
        "packages": resolved_packages,
        "stage": "requirements_written",
    }


def document_deploy_failure(
    target_column: str,
    remediation_steps: list[str] | str,
    rationale: str,
) -> dict[str, Any]:
    """Write output/failure_report.md (DO NOT DEPLOY branch only)."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "profiled", "document_deploy_failure")

    if state.get("verdict") != "DO NOT DEPLOY":
        raise DeploymentError(
            f"document_deploy_failure is only valid for DO NOT DEPLOY verdict, "
            f"got '{state.get('verdict')}'."
        )

    cleaned_rationale = rationale.strip() if rationale else ""
    if len(cleaned_rationale) < 20:
        raise DeploymentError("rationale must be at least 20 characters.")

    if isinstance(remediation_steps, str):
        if not remediation_steps.strip():
            steps = [
                "Review feature engineering and class balance in prep_report.md",
                "Tune hyperparameters or try alternative models",
                "Collect more data for underrepresented classes",
                "Re-run the pipeline after adjustments",
            ]
        else:
            try:
                parsed = json.loads(remediation_steps)
                steps = parsed if isinstance(parsed, list) else [remediation_steps.strip()]
            except json.JSONDecodeError:
                steps = [line.strip() for line in remediation_steps.splitlines() if line.strip()]
    else:
        steps = [s.strip() for s in remediation_steps if s and str(s).strip()]

    if not steps:
        raise DeploymentError("remediation_steps must contain at least one step.")

    report_text = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    metrics_lines = _extract_section_lines(report_text, "Metrics Comparison")
    verdict_lines = _extract_section_lines(report_text, "Verdict")

    lines = [
        "# Deployment Failure Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Target column: `{target_column}`",
        f"- Verdict: **DO NOT DEPLOY**",
        "",
        "## Evaluation Metrics",
        "",
    ]
    lines.extend(metrics_lines or ["- Metrics section not found in evaluation report."])
    lines.extend(["", "## Evaluation Verdict", ""])
    lines.extend(verdict_lines or ["- Verdict section not found in evaluation report."])
    lines.extend(["", "## Agent Rationale", "", cleaned_rationale, "", "## Recommended Next Steps", ""])
    for step in steps:
        lines.append(f"- {step}")
    lines.append("")

    content = "\n".join(lines)
    ensure_output_dirs()
    FAILURE_REPORT_PATH.write_text(content, encoding="utf-8")

    state = ws.load_state()
    state["stage"] = "failure_documented"
    state["remediation_steps"] = steps
    state["failure_rationale"] = cleaned_rationale
    state["artifact_paths"] = {
        **state.get("artifact_paths", {}),
        "failure_report_path": str(FAILURE_REPORT_PATH),
    }
    ws.save_state(state)

    ws.log_decision(
        issue="deployment_blocked",
        options_considered=["DEPLOY", "DO NOT DEPLOY"],
        choice="DO NOT DEPLOY",
        rationale=cleaned_rationale,
    )
    ws.log_step(
        "document_deploy_failure",
        {"target_column": target_column, "remediation_count": len(steps)},
        0,
        0,
        f"Wrote {FAILURE_REPORT_PATH.name}",
    )
    return {
        "target_column": target_column,
        "failure_report_path": str(FAILURE_REPORT_PATH),
        "remediation_steps": steps,
        "rationale": cleaned_rationale,
        "stage": "failure_documented",
    }


def _build_deployment_report(state: dict[str, Any], profile: dict[str, Any]) -> str:
    target_column = state.get("target_column", "unknown")
    verdict = state.get("verdict", "unknown")
    branch = state.get("branch", "unknown")

    lines = [
        "# Deployment Report",
        "",
        f"- Target column: `{target_column}`",
        f"- Verdict: **{verdict}**",
        f"- Branch: `{branch}`",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Context",
        "",
        f"- Algorithm: `{profile.get('algorithm', 'unknown')}`",
        f"- Has predict_proba: `{profile.get('has_predict_proba')}`",
        "",
        "## Artifacts",
        "",
    ]

    artifact_paths = state.get("artifact_paths", {})
    if branch == "deploy":
        for key in ("streamlit_app_path", "inference_utils_path", "requirements_path"):
            path = artifact_paths.get(key)
            if path:
                lines.append(f"- {key}: `{path}`")
    else:
        path = artifact_paths.get("failure_report_path")
        if path:
            lines.append(f"- failure_report_path: `{path}`")

    if state.get("ui_config"):
        lines.extend(["", "## UI Configuration", ""])
        for key, value in state["ui_config"].items():
            lines.append(f"- {key}: {value!r}")

    if state.get("remediation_steps"):
        lines.extend(["", "## Remediation Steps", ""])
        for step in state["remediation_steps"]:
            lines.append(f"- {step}")

    if state.get("failure_rationale"):
        lines.extend(["", "## Failure Rationale", "", state["failure_rationale"]])

    if state.get("steps_applied"):
        lines.extend(["", "## Steps Applied", ""])
        for step in state["steps_applied"]:
            lines.append(f"- **{step['tool']}**: {step['summary']}")

    if state.get("decisions"):
        lines.extend(["", "## Agent Decisions", ""])
        for decision in state["decisions"]:
            lines.append(
                f"- **{decision['issue']}** → {decision['choice']}: {decision['rationale']}"
            )

    return "\n".join(lines)


def finalize_deployment(target_column: str) -> dict[str, Any]:
    """Write deployment report and mark deployment workspace finalized."""
    target_column = _require_target(target_column)
    ensure_output_dirs()
    state = ws.load_state()

    verdict = state.get("verdict")
    branch = state.get("branch")

    if verdict == "DEPLOY":
        _require_stage(state.get("stage"), "requirements_written", "finalize_deployment")
        for path in (STREAMLIT_APP_PATH, INFERENCE_UTILS_PATH, DEPLOY_REQUIREMENTS_PATH):
            if not path.exists():
                raise DeploymentError(f"Missing required artifact: {path}")
    elif verdict == "DO NOT DEPLOY":
        _require_stage(state.get("stage"), "failure_documented", "finalize_deployment")
        if not FAILURE_REPORT_PATH.exists():
            raise DeploymentError(f"Missing required artifact: {FAILURE_REPORT_PATH}")
    else:
        raise DeploymentError(f"Invalid verdict '{verdict}'. Run validate_deploy_inputs first.")

    profile = ws.read_profile()
    report_text = _build_deployment_report(state, profile)
    DEPLOY_REPORT_PATH.write_text(report_text, encoding="utf-8")

    state = ws.load_state()
    state["finalized"] = True
    state["stage"] = "finalized"
    state["artifact_paths"] = {
        **state.get("artifact_paths", {}),
        "deployment_report_path": str(DEPLOY_REPORT_PATH),
    }
    ws.save_state(state)

    ws.log_step(
        "finalize_deployment",
        {"target_column": target_column, "verdict": verdict},
        0,
        0,
        f"Finalized deployment branch={branch}",
    )

    result: dict[str, Any] = {
        "target_column": target_column,
        "verdict": verdict,
        "branch": branch,
        "deployed": verdict == "DEPLOY",
        "deployment_report_path": str(DEPLOY_REPORT_PATH),
        "deploy_state_path": str(DEPLOY_STATE_PATH),
        "finalized": True,
        "stage": "finalized",
        "artifact_paths": state["artifact_paths"],
    }
    if verdict == "DEPLOY":
        result["message"] = "Streamlit app and requirements generated successfully."
    else:
        result["failure_report_path"] = str(FAILURE_REPORT_PATH)
        result["message"] = "Model did not meet deployment criteria."
    return result
