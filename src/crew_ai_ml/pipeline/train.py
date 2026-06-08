"""Model training module with GridSearchCV."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from crew_ai_ml.pipeline.config import (
    DEFAULT_PARAM_GRID,
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
    TRAINING_LOG_PATH,
    ensure_output_dirs,
)


class TrainingError(Exception):
    """Raised when model training fails."""


def _load_input_schema() -> dict[str, Any] | None:
    if not FEATURE_METADATA_PATH.exists():
        return None
    with FEATURE_METADATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _build_training_report(
    *,
    target_column: str,
    train_rows: int,
    val_rows: int,
    feature_columns: list[str],
    class_distribution: dict[str, int],
    best_params: dict[str, Any],
    best_cv_score: float,
    param_grid: dict[str, list[Any]],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Training Log",
        "",
        f"- Timestamp: {timestamp}",
        f"- Target column: `{target_column}`",
        f"- Algorithm: RandomForestClassifier",
        f"- Training rows (80% of train.csv): {train_rows}",
        f"- Validation rows (20% of train.csv): {val_rows}",
        f"- Feature count: {len(feature_columns)}",
        "",
        "## Class Distribution (training subset)",
        "",
    ]
    for label, count in class_distribution.items():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Hyperparameter Search",
            "",
            f"- Search strategy: GridSearchCV with 5-fold stratified CV",
            f"- Scoring metric: f1_weighted",
            f"- Parameter grid:",
            "",
            "```json",
            str(param_grid),
            "```",
            "",
            "## Best Model",
            "",
            f"- Best CV F1 (weighted): {best_cv_score:.4f}",
            f"- Best parameters: {best_params}",
            f"- Model saved to: `{MODEL_PATH}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_training(
    target_column: str,
    param_grid: dict[str, list[Any]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a RandomForest model with GridSearchCV on train.csv.

    Uses an internal 80/20 stratified validation split and saves the best
    estimator bundle to output/model_random_forest.pkl.
    """
    ensure_output_dirs()

    if not target_column or not str(target_column).strip():
        raise TrainingError("target_column must be a non-empty string.")

    target_column = target_column.strip()

    if not TRAIN_DATA_PATH.exists():
        raise TrainingError(
            f"Training data not found at {TRAIN_DATA_PATH}. Run split first."
        )

    df = pd.read_csv(TRAIN_DATA_PATH)
    if target_column not in df.columns:
        raise TrainingError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )

    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        raise TrainingError("No feature columns available for training.")

    X = df[feature_columns]
    y_raw = df[target_column].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if len(set(y)) < 2:
        raise TrainingError("Training data must contain at least two classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    search_grid = param_grid or DEFAULT_PARAM_GRID
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=search_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    val_predictions = grid_search.best_estimator_.predict(X_val)
    from sklearn.metrics import f1_score

    val_f1 = f1_score(y_val, val_predictions, average="weighted")

    input_schema = _load_input_schema()

    model_bundle = {
        "model": grid_search.best_estimator_,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "validation_f1_weighted": float(val_f1),
    }
    if input_schema is not None:
        model_bundle["input_schema"] = input_schema

    joblib.dump(model_bundle, MODEL_PATH)

    class_distribution = (
        pd.Series(label_encoder.inverse_transform(y_train))
        .value_counts()
        .to_dict()
    )

    report = _build_training_report(
        target_column=target_column,
        train_rows=len(X_train),
        val_rows=len(X_val),
        feature_columns=feature_columns,
        class_distribution=class_distribution,
        best_params=grid_search.best_params_,
        best_cv_score=float(grid_search.best_score_),
        param_grid=search_grid,
    )
    TRAINING_LOG_PATH.write_text(report, encoding="utf-8")

    summary: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "training_log_path": str(TRAINING_LOG_PATH),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "validation_f1_weighted": float(val_f1),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "classes": list(label_encoder.classes_),
    }
    if input_schema is not None:
        summary["input_schema"] = input_schema
    return summary
