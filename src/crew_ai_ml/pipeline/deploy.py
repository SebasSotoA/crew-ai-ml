"""Deployment module: Streamlit app generation or failure report."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from crew_ai_ml.pipeline.config import (
    DEPLOY_REQUIREMENTS_PATH,
    EVALUATION_REPORT_PATH,
    FAILURE_REPORT_PATH,
    MODEL_PATH,
    STREAMLIT_APP_PATH,
    ensure_output_dirs,
)


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


def _generate_streamlit_app() -> str:
    return r'''"""Streamlit inference app for the trained Random Forest model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CONFIDENCE_THRESHOLD = 0.60

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_random_forest.pkl"
EVALUATION_REPORT_PATH = APP_DIR / "evaluation_report.md"


# ---------------------------------------------------------------------------
# Feature transformation (self-contained; mirrors feature_transform.py)
# ---------------------------------------------------------------------------

def _normalize_categorical_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _encode_categorical_field(
    value: Any,
    field: dict[str, Any],
    row_data: dict[str, float],
) -> None:
    normalized = _normalize_categorical_value(value)
    categories = field["categories"]
    drop_first = field["drop_first"]
    dummy_columns = field["dummy_columns"]
    prefix = field["name"]

    if drop_first:
        reference = _normalize_categorical_value(categories[0])
        if normalized == reference:
            return
        for dummy in dummy_columns:
            if dummy in row_data:
                row_data[dummy] = 1.0
        return

    for category in categories:
        dummy_name = f"{prefix}_{category}"
        if dummy_name in row_data and normalized == _normalize_categorical_value(category):
            row_data[dummy_name] = 1.0


def transform_raw_input(
    raw_row: dict[str, Any],
    input_schema: list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform one raw input row into a model-ready feature DataFrame."""
    row_data: dict[str, float] = {col: 0.0 for col in feature_columns}

    for field in input_schema:
        name = field["name"]
        if name not in raw_row:
            continue
        value = raw_row[name]
        if field["type"] == "numeric":
            if name in row_data:
                row_data[name] = float(value) if value is not None and value != "" else 0.0
        elif field["type"] == "categorical":
            _encode_categorical_field(value, field, row_data)

    return pd.DataFrame([row_data], columns=feature_columns)


def transform_raw_dataframe(
    df: pd.DataFrame,
    input_schema: list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform a raw DataFrame into model-ready features."""
    rows = [
        transform_raw_input(df.iloc[i].to_dict(), input_schema, feature_columns).iloc[0]
        for i in range(len(df))
    ]
    return pd.DataFrame(rows, columns=feature_columns)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _parse_evaluation_report(report_path: Path) -> dict[str, Any]:
    """Parse evaluation_report.md when evaluation_metrics is not in the bundle."""
    if not report_path.exists():
        return {}

    text = report_path.read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}

    verdict_match = re.search(r"\*\*(DEPLOY|DO NOT DEPLOY)\*\*", text)
    if verdict_match:
        metrics["verdict"] = verdict_match.group(1)

    for line in text.splitlines():
        if "|" not in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3 or parts[0].lower() == "metric":
            continue
        name, test_val = parts[0].lower(), parts[2]
        try:
            val = float(test_val)
        except ValueError:
            continue
        if "accuracy" in name:
            metrics["accuracy"] = val
        elif "precision" in name:
            metrics["precision_weighted"] = val
        elif "recall" in name:
            metrics["recall_weighted"] = val
        elif "f1" in name:
            metrics["f1_weighted"] = val

    bullet_patterns = [
        (r"\*\*Accuracy:\*\*\s*([\d.]+)", "accuracy"),
        (r"\*\*Weighted Precision:\*\*\s*([\d.]+)", "precision_weighted"),
        (r"\*\*Weighted Recall:\*\*\s*([\d.]+)", "recall_weighted"),
        (r"\*\*Weighted F1(?: Score)?:\*\*\s*([\d.]+)", "f1_weighted"),
        (r"\*\*Precision \(weighted\):\*\*\s*([\d.]+)", "precision_weighted"),
        (r"\*\*Recall \(weighted\):\*\*\s*([\d.]+)", "recall_weighted"),
        (r"\*\*F1 \(weighted\):\*\*\s*([\d.]+)", "f1_weighted"),
    ]
    for pattern, key in bullet_patterns:
        if key not in metrics:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

    roc_match = re.search(r"(?:Test )?ROC AUC[:\s*]*([\d.]+)", text, re.IGNORECASE)
    if roc_match:
        metrics["roc_auc"] = float(roc_match.group(1))

    return metrics


def _resolve_evaluation_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    metrics = bundle.get("evaluation_metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    return _parse_evaluation_report(EVALUATION_REPORT_PATH)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _show_confidence_result(label: str, target_column: str, confidence: float | None) -> None:
    if confidence is None:
        st.info(f"Predicted {target_column}: **{label}**")
        return
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"Low confidence prediction ({confidence:.1%}). "
            f"Predicted {target_column}: **{label}**. Review inputs or treat with caution."
        )
    else:
        st.success(
            f"Predicted {target_column}: **{label}** (confidence: {confidence:.1%})"
        )


def main() -> None:
    st.set_page_config(page_title="ML Prediction App", page_icon="🫀", layout="wide")
    st.title("Random Forest Prediction App")
    st.caption("Deployed from crew_ai_ml pipeline")

    bundle = load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]
    target_column: str = bundle["target_column"]
    input_schema: list[dict[str, Any]] = bundle.get("input_schema") or []
    eval_metrics = _resolve_evaluation_metrics(bundle)

    precision = eval_metrics.get("precision_weighted")
    f1 = eval_metrics.get("f1_weighted")
    if precision is not None and f1 is not None:
        st.warning(
            f"Model test precision (weighted): {precision:.1%}. "
            f"F1: {f1:.1%}. Predictions are estimates, not certainties."
        )
    elif precision is not None:
        st.warning(
            f"Model test precision (weighted): {precision:.1%}. "
            "Predictions are estimates, not certainties."
        )
    elif f1 is not None:
        st.warning(
            f"Model test F1 (weighted): {f1:.1%}. "
            "Predictions are estimates, not certainties."
        )

    with st.sidebar:
        st.header("Model Performance")
        if eval_metrics:
            if "accuracy" in eval_metrics:
                st.metric("Test Accuracy", f"{eval_metrics['accuracy']:.1%}")
            if "precision_weighted" in eval_metrics:
                st.metric("Test Precision (weighted)", f"{eval_metrics['precision_weighted']:.1%}")
            if "recall_weighted" in eval_metrics:
                st.metric("Test Recall (weighted)", f"{eval_metrics['recall_weighted']:.1%}")
            if "f1_weighted" in eval_metrics:
                st.metric("Test F1 (weighted)", f"{eval_metrics['f1_weighted']:.1%}")
            if "roc_auc" in eval_metrics:
                st.metric("Test ROC AUC", f"{eval_metrics['roc_auc']:.3f}")
            verdict = eval_metrics.get("verdict", "Unknown")
            st.markdown(f"**Deployment verdict:** {verdict}")
        else:
            st.info("Evaluation metrics not available. Run evaluation and redeploy.")

    tab_manual, tab_csv = st.tabs(["Manual Input", "CSV Upload"])

    with tab_manual:
        st.subheader("Enter Feature Values")
        raw_input: dict[str, Any] = {}

        if input_schema:
            for field in input_schema:
                name = field["name"]
                if field["type"] == "categorical":
                    categories = [str(c) for c in field["categories"]]
                    raw_input[name] = st.selectbox(
                        label=name,
                        options=categories,
                        key=f"manual_{name}",
                    )
                elif field["type"] == "numeric":
                    raw_input[name] = st.number_input(
                        label=name,
                        value=0.0,
                        format="%.4f",
                        key=f"manual_{name}",
                    )
        else:
            st.info(
                "No input schema in model bundle; showing encoded feature columns directly."
            )
            for feature in feature_columns:
                raw_input[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.4f",
                    key=f"manual_{feature}",
                )

        if st.button("Predict", type="primary", key="manual_predict"):
            if input_schema:
                input_df = transform_raw_input(raw_input, input_schema, feature_columns)
            else:
                input_df = pd.DataFrame([raw_input], columns=feature_columns)

            prediction = model.predict(input_df)[0]
            label = label_encoder.inverse_transform([prediction])[0]
            confidence = None
            probabilities = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                confidence = float(probabilities.max())

            _show_confidence_result(label, target_column, confidence)

            if probabilities is not None:
                prob_df = pd.DataFrame(
                    {
                        "class": label_encoder.classes_,
                        "probability": probabilities,
                    }
                ).sort_values("probability", ascending=False)
                st.subheader("Class Probabilities")
                st.dataframe(prob_df, use_container_width=True)

    with tab_csv:
        st.subheader("Batch Prediction from CSV")
        if input_schema:
            expected_cols = [f["name"] for f in input_schema]
            st.caption(f"Expected columns: {', '.join(expected_cols)}")
        else:
            st.caption(f"Expected columns: {', '.join(feature_columns)}")

        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")

        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                return

            if input_schema:
                missing = [c for c in expected_cols if c not in raw_df.columns]
                if missing:
                    st.error(f"CSV is missing required columns: {missing}")
                else:
                    feature_df = transform_raw_dataframe(
                        raw_df[expected_cols], input_schema, feature_columns
                    )
                    predictions = model.predict(feature_df)
                    labels = label_encoder.inverse_transform(predictions)

                    result_df = raw_df.copy()
                    result_df["prediction"] = labels

                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(feature_df)
                        result_df["confidence"] = probas.max(axis=1)
                        low_conf = (result_df["confidence"] < CONFIDENCE_THRESHOLD).sum()
                        if low_conf:
                            st.warning(
                                f"{low_conf} row(s) have confidence below "
                                f"{CONFIDENCE_THRESHOLD:.0%}."
                            )

                    st.dataframe(result_df, use_container_width=True)
            else:
                missing = [c for c in feature_columns if c not in raw_df.columns]
                if missing:
                    st.error(f"CSV is missing required columns: {missing}")
                else:
                    feature_df = raw_df[feature_columns]
                    predictions = model.predict(feature_df)
                    labels = label_encoder.inverse_transform(predictions)

                    result_df = raw_df.copy()
                    result_df["prediction"] = labels

                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(feature_df)
                        result_df["confidence"] = probas.max(axis=1)
                        low_conf = (result_df["confidence"] < CONFIDENCE_THRESHOLD).sum()
                        if low_conf:
                            st.warning(
                                f"{low_conf} row(s) have confidence below "
                                f"{CONFIDENCE_THRESHOLD:.0%}."
                            )

                    st.dataframe(result_df, use_container_width=True)


if __name__ == "__main__":
    main()
'''


def _generate_requirements() -> str:
    return "\n".join(
        [
            "streamlit>=1.28.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
        ]
    )


def _write_failure_report(report_text: str, verdict: str) -> str:
    reasons = _extract_section_lines(report_text, "Deployment Verdict")
    metrics = _extract_section_lines(report_text, "Metrics Comparison")

    lines = [
        "# Deployment Failure Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Verdict: **{verdict}**",
        "",
        "## Evaluation Metrics",
        "",
    ]
    lines.extend(metrics or ["- Metrics section not found in evaluation report."])
    lines.extend(["", "## Reasons", ""])
    lines.extend(reasons or ["- Model did not meet deployment thresholds."])
    lines.extend(
        [
            "",
            "## Recommended Next Steps",
            "",
            "- Review feature engineering and class balance in prep_report.md",
            "- Tune hyperparameters or try alternative models",
            "- Collect more data for underrepresented classes",
            "- Re-run the pipeline after adjustments",
            "",
        ]
    )
    content = "\n".join(lines)
    FAILURE_REPORT_PATH.write_text(content, encoding="utf-8")
    return content


def run_deployment() -> dict[str, Any]:
    """
    Parse evaluation_report.md and deploy or record failure.

    On DEPLOY: writes output/app.py and output/requirements.txt.
    Otherwise: writes output/failure_report.md.
    """
    ensure_output_dirs()

    if not EVALUATION_REPORT_PATH.exists():
        raise DeploymentError(
            f"Evaluation report not found at {EVALUATION_REPORT_PATH}. "
            "Run evaluation first."
        )

    if not MODEL_PATH.exists():
        raise DeploymentError(
            f"Trained model not found at {MODEL_PATH}. Run training first."
        )

    report_text = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    verdict = _parse_verdict(report_text)

    if verdict == "DEPLOY":
        app_code = _generate_streamlit_app()
        requirements = _generate_requirements()

        STREAMLIT_APP_PATH.write_text(app_code, encoding="utf-8")
        DEPLOY_REQUIREMENTS_PATH.write_text(requirements, encoding="utf-8")

        return {
            "verdict": verdict,
            "deployed": True,
            "app_path": str(STREAMLIT_APP_PATH),
            "requirements_path": str(DEPLOY_REQUIREMENTS_PATH),
            "failure_report_path": None,
            "message": "Streamlit app and requirements generated successfully.",
        }

    failure_content = _write_failure_report(report_text, verdict)
    return {
        "verdict": verdict,
        "deployed": False,
        "app_path": None,
        "requirements_path": None,
        "failure_report_path": str(FAILURE_REPORT_PATH),
        "message": "Model did not meet deployment criteria.",
        "failure_report_preview": failure_content[:500],
    }
