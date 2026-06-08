"""Model evaluation module with deployment verdict."""

from __future__ import annotations

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

from crew_ai_ml.pipeline.config import (
    DEPLOY_MAX_F1_GAP,
    DEPLOY_MIN_F1,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_dirs,
)


class EvaluationError(Exception):
    """Raised when model evaluation fails."""


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
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
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


def _determine_verdict(test_f1: float, f1_gap: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    deploy = True

    if test_f1 < DEPLOY_MIN_F1:
        deploy = False
        reasons.append(
            f"Test F1 ({test_f1:.4f}) is below minimum threshold ({DEPLOY_MIN_F1:.2f})"
        )

    if f1_gap > DEPLOY_MAX_F1_GAP:
        deploy = False
        reasons.append(
            f"Train-test F1 gap ({f1_gap:.4f}) exceeds maximum allowed "
            f"({DEPLOY_MAX_F1_GAP:.2f}), indicating possible overfitting"
        )

    if deploy:
        reasons.append(
            f"Test F1 ({test_f1:.4f}) >= {DEPLOY_MIN_F1:.2f} and "
            f"F1 gap ({f1_gap:.4f}) <= {DEPLOY_MAX_F1_GAP:.2f}"
        )

    verdict = "DEPLOY" if deploy else "DO NOT DEPLOY"
    return verdict, reasons


def run_evaluation(target_column: str | None = None) -> dict[str, Any]:
    """
    Evaluate the trained model on train and test sets.

    Generates metrics, plots, and an evaluation report with a deployment verdict.
    """
    ensure_output_dirs()

    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise EvaluationError(
            "Train/test CSV files not found. Run data preparation and split first."
        )

    bundle = _load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]
    resolved_target = target_column or bundle["target_column"]

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    if resolved_target not in train_df.columns or resolved_target not in test_df.columns:
        raise EvaluationError(
            f"Target column '{resolved_target}' missing from train or test data."
        )

    X_train, y_train = _prepare_xy(
        train_df, feature_columns, resolved_target, label_encoder
    )
    X_test, y_test = _prepare_xy(
        test_df, feature_columns, resolved_target, label_encoder
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = _compute_metrics(y_train, y_train_pred)
    test_metrics = _compute_metrics(y_test, y_test_pred)
    f1_gap = abs(train_metrics["f1_weighted"] - test_metrics["f1_weighted"])

    class_names = [str(c) for c in label_encoder.classes_]
    confusion_path = PLOTS_DIR / "confusion_matrix.png"
    roc_path = PLOTS_DIR / "roc_curve.png"

    _plot_confusion_matrix(y_test, y_test_pred, class_names, confusion_path)
    roc_auc = _plot_roc_curve(model, X_test, y_test, class_names, roc_path)

    verdict, verdict_reasons = _determine_verdict(test_metrics["f1_weighted"], f1_gap)

    report_lines = [
        "# Evaluation Report",
        "",
        f"- Target column: `{resolved_target}`",
        f"- Model path: `{MODEL_PATH}`",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | Train | Test |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {train_metrics['accuracy']:.4f} | {test_metrics['accuracy']:.4f} |",
        f"| Precision (weighted) | {train_metrics['precision_weighted']:.4f} | {test_metrics['precision_weighted']:.4f} |",
        f"| Recall (weighted) | {train_metrics['recall_weighted']:.4f} | {test_metrics['recall_weighted']:.4f} |",
        f"| F1 (weighted) | {train_metrics['f1_weighted']:.4f} | {test_metrics['f1_weighted']:.4f} |",
        "",
        f"- Train-test F1 gap: {f1_gap:.4f}",
    ]

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
            "## Deployment Verdict",
            "",
            f"**{verdict}**",
            "",
        ]
    )
    for reason in verdict_reasons:
        report_lines.append(f"- {reason}")

    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Confusion matrix: `{confusion_path}`",
            f"- ROC curve: `{roc_path}`",
            "",
        ]
    )

    EVALUATION_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    evaluation_metrics: dict[str, Any] = {
        "accuracy": test_metrics["accuracy"],
        "precision_weighted": test_metrics["precision_weighted"],
        "recall_weighted": test_metrics["recall_weighted"],
        "f1_weighted": test_metrics["f1_weighted"],
        "verdict": verdict,
    }
    if roc_auc is not None:
        evaluation_metrics["roc_auc"] = roc_auc

    bundle["evaluation_metrics"] = evaluation_metrics
    joblib.dump(bundle, MODEL_PATH)

    return {
        "evaluation_report_path": str(EVALUATION_REPORT_PATH),
        "verdict": verdict,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "evaluation_metrics": evaluation_metrics,
        "f1_gap": f1_gap,
        "roc_auc": roc_auc,
        "confusion_matrix_path": str(confusion_path),
        "roc_curve_path": str(roc_path),
        "verdict_reasons": verdict_reasons,
    }
