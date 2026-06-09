"""Model evaluation pipeline: atomic evaluation steps."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from crew_ai_ml.pipeline import eval_workspace as ws
from crew_ai_ml.pipeline.config import (
    DEPLOY_MAX_F1_GAP,
    DEPLOY_MIN_F1,
    EVAL_REPORT_PATH,
    EVAL_STATE_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_STATE_PATH,
    ensure_output_dirs,
)

__all__ = [
    "EvaluationError",
    "analyze_deploy_signals",
    "compute_eval_metrics",
    "finalize_evaluation",
    "generate_eval_plots",
    "issue_deploy_verdict",
    "profile_eval_context",
    "validate_eval_inputs",
]

SUPPORTED_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
SUPPORTED_PLOTS = ("confusion_matrix", "roc_curve")
STAGE_ORDER = (
    "uninitialized",
    "validated",
    "profiled",
    "metrics_computed",
    "plots_generated",
    "signals_analyzed",
    "verdict_issued",
    "finalized",
)


class EvaluationError(Exception):
    """Raised when model evaluation fails."""


def _require_target(target_column: str) -> str:
    requested = target_column.strip() if target_column else ""
    if not requested:
        kickoff = ws.get_kickoff_target_column()
        if kickoff:
            requested = kickoff
    if not requested:
        raise EvaluationError(
            "target_column must be a non-empty string (tool arg or crew kickoff inputs)."
        )
    return requested.strip()


def _stage_index(stage: str | None) -> int:
    if stage in STAGE_ORDER:
        return STAGE_ORDER.index(stage)
    return -1


def _require_stage(current: str | None, minimum: str, tool_name: str) -> None:
    if _stage_index(current) < _stage_index(minimum):
        raise EvaluationError(
            f"Expected stage '{minimum}' or later, got '{current}'. "
            f"Complete prior evaluation steps before calling {tool_name}."
        )


def _load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise EvaluationError(
            f"Model not found at {MODEL_PATH}. Run training first."
        )
    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "label_encoder", "feature_columns", "target_column"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise EvaluationError(f"Model bundle missing keys: {sorted(missing)}")
    return bundle


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    label_encoder,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise EvaluationError(
            f"Dataset missing expected features: {missing_features[:5]}"
            + ("..." if len(missing_features) > 5 else "")
        )

    X = df[feature_columns]
    y = label_encoder.transform(df[target_column].astype(str))
    return X, y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_roc_curve(
    model,
    X: pd.DataFrame,
    y_true: np.ndarray,
    class_names: list[str],
    output_path,
) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    y_proba = model.predict_proba(X)
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        roc_auc = roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr")
        for idx, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
            plt.plot(fpr, tpr, label=name)
        plt.plot([], [], " ", label=f"Weighted AUC = {roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return float(roc_auc)


def _read_train_candidate_metrics() -> dict[str, Any] | None:
    if not TRAIN_STATE_PATH.exists():
        return None
    with TRAIN_STATE_PATH.open(encoding="utf-8") as f:
        train_state = json.load(f)

    candidates = train_state.get("candidates", [])
    if not candidates:
        return None

    best_id = train_state.get("best_candidate_id")
    candidate = None
    if best_id:
        candidate = next((c for c in candidates if c["id"] == best_id), None)

    if candidate is None:
        def _score(entry: dict[str, Any]) -> float:
            metrics = entry.get("metrics", {})
            return float(
                metrics.get("validation_f1_weighted")
                or metrics.get("best_cv_score")
                or -1.0
            )

        candidate = max(candidates, key=_score)

    return {
        "candidate_id": candidate.get("id"),
        "algorithm": candidate.get("algorithm"),
        "training_method": candidate.get("training_method"),
        "metrics": candidate.get("metrics", {}),
    }


def _load_split_frame(split: str) -> pd.DataFrame:
    if split == "train":
        path = TRAIN_DATA_PATH
    elif split == "test":
        path = TEST_DATA_PATH
    else:
        raise EvaluationError(
            f"Unknown split '{split}'. Supported splits: train, test."
        )
    if not path.exists():
        raise EvaluationError(f"{split}.csv not found at {path}.")
    return pd.read_csv(path)


def _validate_binary_target(df: pd.DataFrame, target_column: str, label: str) -> None:
    if target_column not in df.columns:
        raise EvaluationError(
            f"Target column '{target_column}' missing from {label} data."
        )
    n_classes = df[target_column].nunique(dropna=True)
    if n_classes != 2:
        raise EvaluationError(
            f"Binary classification requires exactly 2 classes in {label} data, "
            f"found {n_classes}."
        )


def validate_eval_inputs(target_column: str) -> dict[str, Any]:
    """Validate model and datasets are ready for evaluation; reset eval workspace."""
    ensure_output_dirs()
    target_column = _require_target(target_column)

    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise EvaluationError(
            "Train/test CSV files not found. Run data preparation and split first."
        )

    bundle = _load_model_bundle()
    resolved_target = target_column
    if bundle["target_column"] != resolved_target:
        raise EvaluationError(
            f"Model target '{bundle['target_column']}' does not match "
            f"requested target '{resolved_target}'."
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    _validate_binary_target(train_df, resolved_target, "train")
    _validate_binary_target(test_df, resolved_target, "test")

    ws.reset_workspace(resolved_target)
    state = ws.load_state()
    state["stage"] = "validated"
    ws.save_state(state)

    ws.log_step(
        "validate_eval_inputs",
        {"target_column": resolved_target},
        len(train_df) + len(test_df),
        len(train_df) + len(test_df),
        f"Validated model bundle and binary target on train ({len(train_df)}) "
        f"and test ({len(test_df)}) rows",
    )
    return {
        "target_column": resolved_target,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_path": str(MODEL_PATH),
        "feature_count": len(bundle["feature_columns"]),
        "stage": "validated",
    }


def profile_eval_context(target_column: str) -> dict[str, Any]:
    """Profile evaluation context: train candidate metrics, test balance, deploy thresholds."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "validated", "profile_eval_context")

    bundle = _load_model_bundle()
    model = bundle["model"]
    test_df = pd.read_csv(TEST_DATA_PATH)
    _validate_binary_target(test_df, target_column, "test")

    class_counts = test_df[target_column].value_counts().to_dict()
    majority = max(class_counts.values()) if class_counts else 0
    minority = min(class_counts.values()) if class_counts else 0

    candidate_context = _read_train_candidate_metrics()
    has_predict_proba = hasattr(model, "predict_proba")

    profile = {
        "target_column": target_column,
        "test_rows": len(test_df),
        "test_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "test_class_balance": {
            "majority_count": int(majority),
            "minority_count": int(minority),
            "minority_to_majority_ratio": round(minority / majority, 4) if majority else 1.0,
        },
        "selected_candidate": candidate_context,
        "deploy_thresholds": {
            "deploy_min_f1": DEPLOY_MIN_F1,
            "deploy_max_f1_gap": DEPLOY_MAX_F1_GAP,
        },
        "has_predict_proba": has_predict_proba,
        "supported_metrics": list(SUPPORTED_METRICS),
        "supported_plots": list(SUPPORTED_PLOTS),
    }
    ws.write_profile(profile)

    state = ws.load_state()
    state["stage"] = "profiled"
    ws.save_state(state)

    ws.log_step(
        "profile_eval_context",
        {"target_column": target_column},
        len(test_df),
        len(test_df),
        f"Profiled eval context; predict_proba={has_predict_proba}",
    )
    return profile


def compute_eval_metrics(
    target_column: str,
    splits: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    """Compute classification metrics on requested train/test splits."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "profiled", "compute_eval_metrics")

    if not splits:
        raise EvaluationError("splits must be a non-empty list (e.g. ['train', 'test']).")

    invalid_splits = [s for s in splits if s not in {"train", "test"}]
    if invalid_splits:
        raise EvaluationError(
            f"Invalid splits: {invalid_splits}. Supported: train, test."
        )

    if not metrics:
        raise EvaluationError(
            f"metrics must be a non-empty subset of {list(SUPPORTED_METRICS)}."
        )

    invalid_metrics = [m for m in metrics if m not in SUPPORTED_METRICS]
    if invalid_metrics:
        raise EvaluationError(
            f"Invalid metrics: {invalid_metrics}. Supported: {list(SUPPORTED_METRICS)}."
        )

    bundle = _load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]

    computed: dict[str, dict[str, float]] = {}
    total_rows = 0

    for split in splits:
        df = _load_split_frame(split)
        _validate_binary_target(df, target_column, split)
        X, y = _prepare_xy(df, feature_columns, target_column, label_encoder)
        y_pred = model.predict(X)
        split_metrics = _compute_metrics(y, y_pred)
        computed[split] = {m: split_metrics[m] for m in metrics}
        total_rows += len(df)

    state = ws.load_state()
    state["metrics"] = computed
    state["splits_computed"] = list(splits)
    state["stage"] = "metrics_computed"
    ws.save_state(state)

    ws.log_step(
        "compute_eval_metrics",
        {"target_column": target_column, "splits": splits, "metrics": metrics},
        total_rows,
        total_rows,
        f"Computed {len(metrics)} metrics on {', '.join(splits)}",
    )
    return {
        "target_column": target_column,
        "metrics": computed,
        "splits_computed": splits,
        "stage": "metrics_computed",
    }


def generate_eval_plots(target_column: str, plots: list[str]) -> dict[str, Any]:
    """Generate evaluation plots on the test set and store artifact paths."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "metrics_computed", "generate_eval_plots")

    if not plots:
        raise EvaluationError(
            f"plots must be a non-empty subset of {list(SUPPORTED_PLOTS)}."
        )

    invalid_plots = [p for p in plots if p not in SUPPORTED_PLOTS]
    if invalid_plots:
        raise EvaluationError(
            f"Invalid plots: {invalid_plots}. Supported: {list(SUPPORTED_PLOTS)}."
        )

    profile = ws.read_profile()
    bundle = _load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test, y_test = _prepare_xy(
        test_df, feature_columns, target_column, label_encoder
    )
    y_test_pred = model.predict(X_test)
    class_names = [str(c) for c in label_encoder.classes_]

    plot_paths: dict[str, str] = {}
    roc_auc: float | None = None

    if "confusion_matrix" in plots:
        confusion_path = PLOTS_DIR / "confusion_matrix.png"
        _plot_confusion_matrix(y_test, y_test_pred, class_names, confusion_path)
        plot_paths["confusion_matrix"] = str(confusion_path)

    if "roc_curve" in plots:
        if not profile.get("has_predict_proba", hasattr(model, "predict_proba")):
            raise EvaluationError(
                "roc_curve plot requires a model with predict_proba support."
            )
        roc_path = PLOTS_DIR / "roc_curve.png"
        roc_auc = _plot_roc_curve(model, X_test, y_test, class_names, roc_path)
        plot_paths["roc_curve"] = str(roc_path)

    state = ws.load_state()
    state["plots"] = plot_paths
    state["roc_auc"] = roc_auc
    state["stage"] = "plots_generated"
    ws.save_state(state)

    ws.log_step(
        "generate_eval_plots",
        {"target_column": target_column, "plots": plots},
        len(test_df),
        len(test_df),
        f"Generated plots: {', '.join(plot_paths.keys())}",
    )
    return {
        "target_column": target_column,
        "plots": plot_paths,
        "roc_auc": roc_auc,
        "stage": "plots_generated",
    }


def analyze_deploy_signals(target_column: str) -> dict[str, Any]:
    """Compute deployment signals for agent review; does not issue a verdict."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "plots_generated", "analyze_deploy_signals")

    metrics = state.get("metrics", {})
    if "train" not in metrics or "test" not in metrics:
        raise EvaluationError(
            "Train and test metrics required. Call compute_eval_metrics with "
            "splits=['train', 'test'] and include f1_weighted."
        )

    train_f1 = metrics["train"].get("f1_weighted")
    test_f1 = metrics["test"].get("f1_weighted")
    if train_f1 is None or test_f1 is None:
        raise EvaluationError(
            "f1_weighted must be computed for both train and test splits."
        )

    f1_gap = abs(float(train_f1) - float(test_f1))
    test_df = pd.read_csv(TEST_DATA_PATH)
    class_counts = test_df[target_column].value_counts().to_dict()
    majority = max(class_counts.values()) if class_counts else 0
    minority = min(class_counts.values()) if class_counts else 0

    candidate_context = _read_train_candidate_metrics() or {}
    candidate_metrics = candidate_context.get("metrics", {})
    validation_f1 = candidate_metrics.get("validation_f1_weighted")
    validation_to_test_f1_delta = None
    if validation_f1 is not None:
        validation_to_test_f1_delta = float(validation_f1) - float(test_f1)

    threshold_checks = {
        "deploy_min_f1": DEPLOY_MIN_F1,
        "deploy_max_f1_gap": DEPLOY_MAX_F1_GAP,
        "test_f1_weighted": float(test_f1),
        "f1_gap": f1_gap,
        "test_f1_meets_minimum": float(test_f1) >= DEPLOY_MIN_F1,
        "f1_gap_within_limit": f1_gap <= DEPLOY_MAX_F1_GAP,
    }

    signals = {
        "f1_gap": f1_gap,
        "threshold_checks": threshold_checks,
        "class_balance": {
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "majority_count": int(majority),
            "minority_count": int(minority),
            "minority_to_majority_ratio": round(minority / majority, 4) if majority else 1.0,
        },
        "validation_to_test_f1_delta": validation_to_test_f1_delta,
        "selected_candidate": candidate_context,
    }

    state = ws.load_state()
    state["signals"] = signals
    state["stage"] = "signals_analyzed"
    ws.save_state(state)

    ws.log_step(
        "analyze_deploy_signals",
        {"target_column": target_column},
        len(test_df),
        len(test_df),
        f"Analyzed deploy signals; test F1={float(test_f1):.4f}, gap={f1_gap:.4f}",
    )
    return {
        "target_column": target_column,
        "signals": signals,
        "stage": "signals_analyzed",
    }


def issue_deploy_verdict(
    target_column: str,
    verdict: str,
    rationale: str,
) -> dict[str, Any]:
    """Record the agent's deployment verdict and rationale."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    _require_stage(state.get("stage"), "signals_analyzed", "issue_deploy_verdict")

    normalized = verdict.strip().upper()
    if normalized == "DO NOT DEPLOY":
        verdict_value = "DO NOT DEPLOY"
    elif normalized == "DEPLOY":
        verdict_value = "DEPLOY"
    else:
        raise EvaluationError(
            "verdict must be 'DEPLOY' or 'DO NOT DEPLOY'."
        )

    cleaned_rationale = rationale.strip()
    if len(cleaned_rationale) < 20:
        raise EvaluationError("rationale must be at least 20 characters.")

    state = ws.load_state()
    state["verdict"] = verdict_value
    state["rationale"] = cleaned_rationale
    state["stage"] = "verdict_issued"
    ws.save_state(state)

    ws.log_decision(
        issue="deployment_verdict",
        options_considered=["DEPLOY", "DO NOT DEPLOY"],
        choice=verdict_value,
        rationale=cleaned_rationale,
    )

    ws.log_step(
        "issue_deploy_verdict",
        {"target_column": target_column, "verdict": verdict_value},
        0,
        0,
        f"Agent verdict: {verdict_value}",
    )
    return {
        "target_column": target_column,
        "verdict": verdict_value,
        "rationale": cleaned_rationale,
        "stage": "verdict_issued",
    }


def finalize_evaluation(target_column: str) -> dict[str, Any]:
    """Write evaluation report and persist agent verdict to model.pkl."""
    target_column = _require_target(target_column)
    ensure_output_dirs()
    state = ws.load_state()
    _require_stage(state.get("stage"), "verdict_issued", "finalize_evaluation")

    verdict = state.get("verdict")
    rationale = state.get("rationale")
    if not verdict or not rationale:
        raise EvaluationError(
            "Missing agent verdict or rationale. Call issue_deploy_verdict first."
        )

    metrics = state.get("metrics", {})
    train_metrics = metrics.get("train", {})
    test_metrics = metrics.get("test", {})
    if not train_metrics or not test_metrics:
        raise EvaluationError(
            "Missing train/test metrics. Call compute_eval_metrics first."
        )

    bundle = _load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test, y_test = _prepare_xy(
        test_df, feature_columns, target_column, label_encoder
    )
    y_test_pred = model.predict(X_test)
    class_names = [str(c) for c in label_encoder.classes_]

    f1_gap = abs(
        float(train_metrics.get("f1_weighted", 0.0))
        - float(test_metrics.get("f1_weighted", 0.0))
    )
    roc_auc = state.get("roc_auc")
    plot_paths = state.get("plots", {})

    report_lines = [
        "# Evaluation Report",
        "",
        f"- Target column: `{target_column}`",
        f"- Model path: `{MODEL_PATH}`",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | Train | Test |",
        "| --- | ---: | ---: |",
    ]

    metric_labels = {
        "accuracy": "Accuracy",
        "precision_weighted": "Precision (weighted)",
        "recall_weighted": "Recall (weighted)",
        "f1_weighted": "F1 (weighted)",
    }
    for key, label in metric_labels.items():
        if key in train_metrics or key in test_metrics:
            train_val = train_metrics.get(key, 0.0)
            test_val = test_metrics.get(key, 0.0)
            report_lines.append(f"| {label} | {train_val:.4f} | {test_val:.4f} |")

    report_lines.append("")
    report_lines.append(f"- Train-test F1 gap: {f1_gap:.4f}")
    if roc_auc is not None:
        report_lines.append(f"- Test ROC AUC: {roc_auc:.4f}")

    report_lines.extend(
        [
            "",
            "## Classification Report (Test)",
            "",
            "```",
            classification_report(
                y_test,
                y_test_pred,
                target_names=class_names,
                zero_division=0,
            ),
            "```",
            "",
            "## Verdict",
            "",
            f"**{verdict}**",
            "",
            f"- Agent rationale: {rationale}",
            "",
        ]
    )

    signals = state.get("signals", {})
    if signals:
        report_lines.extend(
            [
                "## Deploy Signals",
                "",
                f"- Suggested verdict (threshold-based): **{signals.get('suggested_verdict', 'N/A')}**",
                f"- Threshold guidance: {signals.get('threshold_guidance', '')}",
            ]
        )
        for reason in signals.get("signal_reasons", []):
            report_lines.append(f"- {reason}")
        report_lines.append("")

    if state.get("decisions"):
        report_lines.extend(["## Agent Decisions", ""])
        for decision in state["decisions"]:
            report_lines.append(
                f"- **{decision['issue']}** → {decision['choice']}: {decision['rationale']}"
            )
        report_lines.append("")

    if state.get("steps_applied"):
        report_lines.extend(["## Steps Applied (from eval_state.json)", ""])
        for step in state["steps_applied"]:
            report_lines.append(f"- **{step['tool']}**: {step['summary']}")
        report_lines.append("")

    report_lines.extend(["## Artifacts", ""])
    confusion_path = plot_paths.get("confusion_matrix")
    roc_path = plot_paths.get("roc_curve")
    if confusion_path:
        report_lines.append(f"- Confusion matrix: `{confusion_path}`")
    if roc_path:
        report_lines.append(f"- ROC curve: `{roc_path}`")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    EVAL_REPORT_PATH.write_text(report_text, encoding="utf-8")
    EVALUATION_REPORT_PATH.write_text(report_text, encoding="utf-8")

    evaluation_metrics: dict[str, Any] = {
        "accuracy": test_metrics.get("accuracy"),
        "precision_weighted": test_metrics.get("precision_weighted"),
        "recall_weighted": test_metrics.get("recall_weighted"),
        "f1_weighted": test_metrics.get("f1_weighted"),
        "verdict": verdict,
        "rationale": rationale,
    }
    if roc_auc is not None:
        evaluation_metrics["roc_auc"] = roc_auc

    bundle["evaluation_metrics"] = evaluation_metrics
    joblib.dump(bundle, MODEL_PATH)

    state = ws.load_state()
    state["finalized"] = True
    state["stage"] = "finalized"
    ws.save_state(state)

    ws.log_step(
        "finalize_evaluation",
        {"target_column": target_column, "verdict": verdict},
        len(test_df),
        len(test_df),
        f"Wrote {EVALUATION_REPORT_PATH.name} with verdict {verdict}",
    )
    return {
        "eval_report_path": str(EVAL_REPORT_PATH),
        "evaluation_report_path": str(EVALUATION_REPORT_PATH),
        "eval_state_path": str(EVAL_STATE_PATH),
        "model_path": str(MODEL_PATH),
        "target_column": target_column,
        "verdict": verdict,
        "rationale": rationale,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "evaluation_metrics": evaluation_metrics,
        "f1_gap": f1_gap,
        "roc_auc": roc_auc,
        "plots": plot_paths,
        "finalized": True,
        "stage": "finalized",
    }
