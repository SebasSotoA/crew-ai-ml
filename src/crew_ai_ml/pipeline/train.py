"""Model training pipeline: atomic training steps."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from crew_ai_ml.pipeline import model_registry as registry
from crew_ai_ml.pipeline import train_workspace as ws
from crew_ai_ml.pipeline.config import (
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
    TRAIN_STATE_PATH,
    TRAINING_LOG_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.model_registry import ALGORITHMS
from crew_ai_ml.pipeline.validators import (
    feature_frame_null_report,
    feature_null_report,
    format_null_report,
    null_report_message,
)

__all__ = [
    "TrainingError",
    "finalize_training",
    "list_training_candidates",
    "log_training_decision",
    "profile_train_data",
    "train_baseline",
    "tune_hyperparameters",
    "validate_train_data",
]


class TrainingError(Exception):
    """Raised when model training fails."""


def _require_target(target_column: str) -> str:
    requested = target_column.strip() if target_column else ""
    if not requested:
        kickoff = ws.get_kickoff_target_column()
        if kickoff:
            requested = kickoff
    if not requested:
        raise TrainingError(
            "target_column must be a non-empty string (tool arg or crew kickoff inputs)."
        )
    return requested.strip()


def _load_input_schema() -> dict[str, Any] | list[Any] | None:
    if not FEATURE_METADATA_PATH.exists():
        return None
    with FEATURE_METADATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_train_frame(target_column: str) -> tuple[pd.DataFrame, list[str]]:
    if not TRAIN_DATA_PATH.exists():
        raise TrainingError(
            f"Training data not found at {TRAIN_DATA_PATH}. Run split first."
        )
    df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    if target_column not in df.columns:
        raise TrainingError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )
    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        raise TrainingError("No feature columns available for training.")
    return df, feature_columns


def _assert_finite_features(X: pd.DataFrame) -> None:
    nulls = feature_frame_null_report(X)
    if nulls:
        raise TrainingError(
            f"Feature matrix contains missing values: {format_null_report(nulls)}. "
            "Re-run data preparation with impute_missing before training."
        )


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, Any, LabelEncoder]:
    X = df[feature_columns]
    y_raw = df[target_column].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X, y, label_encoder


def _internal_split(
    X: pd.DataFrame,
    y: Any,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    if len(set(y)) < 2:
        raise TrainingError("Training data must contain at least two classes.")
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )


def _require_estimator_params(params: dict[str, Any] | str | None) -> dict[str, Any]:
    if params is None:
        raise TrainingError("params is required (non-empty dict or JSON string).")
    if isinstance(params, str):
        if not params.strip():
            raise TrainingError("params is required (non-empty dict or JSON string).")
        try:
            parsed = registry.parse_estimator_params(params)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc
        if not parsed:
            raise TrainingError("params must decode to a non-empty object.")
        return parsed
    if not params:
        raise TrainingError("params is required (non-empty dict or JSON string).")
    return dict(params)


def _parse_options_considered(options_considered: list[str] | str) -> list[str]:
    if isinstance(options_considered, str):
        if not options_considered.strip():
            raise TrainingError(
                "options_considered must be a non-empty list or JSON string."
            )
        try:
            parsed = json.loads(options_considered)
        except json.JSONDecodeError as exc:
            raise TrainingError(f"options_considered JSON invalid: {exc}") from exc
        if not isinstance(parsed, list):
            raise TrainingError("options_considered must decode to a JSON array.")
        return [str(option) for option in parsed]
    return [str(option) for option in options_considered]


def _parse_fixed_params(fixed_params: dict[str, Any] | str | None) -> dict[str, Any]:
    if fixed_params is None:
        return {}
    if isinstance(fixed_params, str):
        if not fixed_params.strip():
            return {}
        try:
            return registry.parse_estimator_params(fixed_params)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc
    return dict(fixed_params)


def _require_param_grid(
    param_grid: dict[str, list[Any]] | str | None,
) -> dict[str, list[Any]]:
    if param_grid is None:
        raise TrainingError("param_grid is required (non-empty dict or JSON string).")
    if isinstance(param_grid, str):
        if not param_grid.strip():
            raise TrainingError("param_grid is required (non-empty dict or JSON string).")
        try:
            parsed = registry.parse_param_grid(param_grid)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc
        return parsed
    if not param_grid:
        raise TrainingError("param_grid is required (non-empty dict or JSON string).")
    return param_grid


def _fit_and_score_baseline(
    algorithm: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: Any,
    y_val: Any,
) -> tuple[Any, dict[str, Any], float, float]:
    estimator = registry.get_estimator(algorithm, params)
    estimator.fit(X_train, y_train)
    val_predictions = estimator.predict(X_val)
    val_f1 = float(f1_score(y_val, val_predictions, average="weighted"))
    train_predictions = estimator.predict(X_train)
    train_f1 = float(f1_score(y_train, train_predictions, average="weighted"))
    params = dict(estimator.get_params())
    return estimator, params, train_f1, val_f1


def _run_grid_search(
    algorithm: str,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: Any,
    X_val: pd.DataFrame,
    y_val: Any,
    *,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    fixed_params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any], float, float, float]:
    key = registry.validate_algorithm(algorithm)
    max_combos = ALGORITHMS[key]["max_grid_combinations"]
    combo_count = registry.count_grid_combinations(param_grid)
    if combo_count > max_combos:
        raise TrainingError(
            f"Parameter grid has {combo_count} combinations, "
            f"exceeding max {max_combos} for '{algorithm}'."
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_model = registry.get_estimator(algorithm, fixed_params or {"random_state": random_state})
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    val_predictions = grid_search.best_estimator_.predict(X_val)
    val_f1 = float(f1_score(y_val, val_predictions, average="weighted"))

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        float(grid_search.best_score_),
        val_f1,
        combo_count,
    )


def _build_candidate_bundle(
    *,
    model: Any,
    label_encoder: LabelEncoder,
    feature_columns: list[str],
    target_column: str,
    algorithm: str,
    training_method: str,
    params: dict[str, Any],
    train_f1: float | None,
    val_f1: float,
    cv_score: float | None = None,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "algorithm": algorithm,
        "training_method": training_method,
        "params": params,
        "train_f1_weighted": train_f1,
        "validation_f1_weighted": val_f1,
    }
    if cv_score is not None:
        bundle["best_cv_score"] = cv_score
    input_schema = _load_input_schema()
    if input_schema is not None:
        bundle["input_schema"] = input_schema
    return bundle


def _save_candidate(
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        entry = ws.register_candidate(metadata)
    except ws.TrainWorkspaceError as exc:
        raise TrainingError(str(exc)) from exc
    candidate_id = entry["id"]
    ws.save_candidate(candidate_id, bundle)
    entry["artifact_path"] = str(ws.candidate_artifact_path(candidate_id))
    state = ws.load_state()
    for idx, candidate in enumerate(state["candidates"]):
        if candidate["id"] == candidate_id:
            state["candidates"][idx] = {**candidate, **entry}
            break
    ws.save_state(state)
    return entry


def validate_train_data(target_column: str) -> dict[str, Any]:
    """Validate train.csv is ready for modeling and reset training workspace."""
    ensure_output_dirs()
    target_column = _require_target(target_column)

    df, feature_columns = _load_train_frame(target_column)

    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TrainingError(
            f"All features must be numeric. Non-numeric: {non_numeric}"
        )

    nulls = feature_null_report(df, target_column)
    if nulls:
        raise TrainingError(
            null_report_message(
                nulls,
                "Re-run data preparation with impute_missing before training.",
            )
        )

    n_classes = df[target_column].nunique(dropna=True)
    if n_classes != 2:
        raise TrainingError(
            f"Binary classification requires exactly 2 classes, found {n_classes}."
        )

    ws.reset_workspace(target_column)
    state = ws.load_state()
    state["stage"] = "validated"
    ws.save_state(state)

    ws.log_step(
        "validate_train_data",
        {"target_column": target_column},
        len(df),
        len(df),
        f"Validated {len(df)} rows, {len(feature_columns)} numeric features, binary target",
    )
    return {
        "target_column": target_column,
        "rows": len(df),
        "feature_count": len(feature_columns),
        "class_counts": df[target_column].value_counts().to_dict(),
        "stage": "validated",
    }


def profile_train_data(target_column: str) -> dict[str, Any]:
    """Profile training data and recommend algorithm and tuning strategy."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") != "validated":
        raise TrainingError(
            f"Expected stage 'validated', got '{state.get('stage')}'. "
            "Call validate_train_data first."
        )

    df, feature_columns = _load_train_frame(target_column)
    class_counts = df[target_column].value_counts().to_dict()
    rows = len(df)
    feature_count = len(feature_columns)
    missing_features = feature_null_report(df, target_column)

    if rows < 1000:
        recommended_algorithm = "logistic_regression"
        tune_recommended = False
    elif rows < 5000:
        recommended_algorithm = "random_forest"
        tune_recommended = True
    else:
        recommended_algorithm = "gradient_boosting"
        tune_recommended = True

    algorithm_catalog = {
        name: {
            "tunable_params": entry["tunable_params"],
            "max_grid_combinations": entry["max_grid_combinations"],
        }
        for name, entry in registry.ALGORITHMS.items()
    }
    profile = {
        "rows": rows,
        "target_column": target_column,
        "feature_count": feature_count,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "has_missing_features": bool(missing_features),
        "missing_feature_columns": missing_features,
        "recommendations": {
            "algorithm": recommended_algorithm,
            "tune_recommended": tune_recommended,
            "random_state": 42,
            "internal_val_size": 0.2,
            "scoring": "f1_weighted",
        },
        "supported_algorithms": sorted(registry.ALGORITHMS),
        "algorithm_catalog": algorithm_catalog,
    }
    ws.write_profile(profile)

    state = ws.load_state()
    state["stage"] = "profiled"
    ws.save_state(state)

    ws.log_step(
        "profile_train_data",
        {"target_column": target_column},
        rows,
        rows,
        f"Profiled training data; recommended algorithm={recommended_algorithm}",
    )
    return profile


def log_training_decision(
    target_column: str,
    issue: str,
    options_considered: list[str] | str,
    choice: str,
    rationale: str,
) -> dict[str, Any]:
    """Record an agent training decision with rationale."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") not in {"profiled", "trained"}:
        raise TrainingError(
            f"Expected stage 'profiled' or 'trained', got '{state.get('stage')}'. "
            "Call profile_train_data first."
        )

    cleaned_issue = issue.strip()
    if not cleaned_issue:
        raise TrainingError("issue must be a non-empty string.")

    cleaned_choice = choice.strip()
    if not cleaned_choice:
        raise TrainingError("choice must be a non-empty string.")

    cleaned_rationale = rationale.strip()
    if len(cleaned_rationale) < 20:
        raise TrainingError("rationale must be at least 20 characters.")

    parsed_options = _parse_options_considered(options_considered)
    decision_count_before = len(state.get("decisions", []))

    ws.log_decision(
        issue=cleaned_issue,
        options_considered=parsed_options,
        choice=cleaned_choice,
        rationale=cleaned_rationale,
    )

    state = ws.load_state()
    decision_count = len(state.get("decisions", []))
    ws.log_step(
        "log_training_decision",
        {
            "target_column": target_column,
            "issue": cleaned_issue,
            "choice": cleaned_choice,
            "options_count": len(parsed_options),
        },
        decision_count_before,
        decision_count,
        f"Logged decision: {cleaned_issue} → {cleaned_choice}",
    )
    return {
        "decision_count": decision_count,
        "issue": cleaned_issue,
        "choice": cleaned_choice,
    }


def train_baseline(
    target_column: str,
    params: dict[str, Any] | str,
    algorithm: str = "random_forest",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a baseline model with agent-defined hyperparameters and store as candidate."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") not in {"profiled", "trained"}:
        raise TrainingError(
            f"Expected stage 'profiled' or 'trained', got '{state.get('stage')}'. "
            "Call profile_train_data first."
        )

    algorithm = registry.validate_algorithm(algorithm)
    estimator_params = _require_estimator_params(params)
    df, feature_columns = _load_train_frame(target_column)
    X, y, label_encoder = _prepare_xy(df, feature_columns, target_column)
    _assert_finite_features(X)
    X_train, X_val, y_train, y_val = _internal_split(X, y, random_state=random_state)

    model, fitted_params, train_f1, val_f1 = _fit_and_score_baseline(
        algorithm, estimator_params, X_train, X_val, y_train, y_val
    )

    bundle = _build_candidate_bundle(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        target_column=target_column,
        algorithm=algorithm,
        training_method="baseline",
        params=fitted_params,
        train_f1=train_f1,
        val_f1=val_f1,
    )

    entry = _save_candidate(
        bundle,
        {
            "algorithm": algorithm,
            "training_method": "baseline",
            "params": fitted_params,
            "metrics": {
                "train_f1_weighted": train_f1,
                "validation_f1_weighted": val_f1,
            },
        },
    )

    state = ws.load_state()
    state["stage"] = "trained"
    if state.get("best_candidate_id") is None:
        state["best_candidate_id"] = entry["id"]
    ws.save_state(state)

    ws.log_step(
        "train_baseline",
        {"target_column": target_column, "algorithm": algorithm, "random_state": random_state},
        len(df),
        len(df),
        f"Baseline {algorithm}: val F1={val_f1:.4f}",
    )
    return {
        "candidate_id": entry["id"],
        "algorithm": algorithm,
        "training_method": "baseline",
        "params": fitted_params,
        "train_f1_weighted": train_f1,
        "validation_f1_weighted": val_f1,
        "artifact_path": entry.get("artifact_path"),
        "stage": "trained",
    }


def tune_hyperparameters(
    target_column: str,
    param_grid: dict[str, list[Any]] | str,
    algorithm: str = "random_forest",
    fixed_params: dict[str, Any] | str | None = None,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
) -> dict[str, Any]:
    """Run GridSearchCV and store the best estimator as a candidate."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") not in {"profiled", "trained"}:
        raise TrainingError(
            f"Expected stage 'profiled' or 'trained', got '{state.get('stage')}'. "
            "Call profile_train_data first."
        )

    algorithm = registry.validate_algorithm(algorithm)
    search_grid = _require_param_grid(param_grid)
    constant_params = _parse_fixed_params(fixed_params)
    if "random_state" not in constant_params:
        constant_params["random_state"] = random_state

    df, feature_columns = _load_train_frame(target_column)
    X, y, label_encoder = _prepare_xy(df, feature_columns, target_column)
    _assert_finite_features(X)
    X_train, X_val, y_train, y_val = _internal_split(X, y, random_state=random_state)

    model, best_params, cv_score, val_f1, combo_count = _run_grid_search(
        algorithm,
        search_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        cv_folds=cv_folds,
        scoring=scoring,
        fixed_params=constant_params,
        random_state=random_state,
    )

    bundle = _build_candidate_bundle(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        target_column=target_column,
        algorithm=algorithm,
        training_method="tuned",
        params=best_params,
        train_f1=None,
        val_f1=val_f1,
        cv_score=cv_score,
    )

    entry = _save_candidate(
        bundle,
        {
            "algorithm": algorithm,
            "training_method": "tuned",
            "params": best_params,
            "metrics": {
                "best_cv_score": cv_score,
                "validation_f1_weighted": val_f1,
            },
            "grid_combinations": combo_count,
        },
    )

    state = ws.load_state()
    state["stage"] = "trained"
    current_best = state.get("best_candidate_id")
    if current_best:
        candidates = {c["id"]: c for c in state.get("candidates", [])}
        current_val = candidates.get(current_best, {}).get("metrics", {}).get(
            "validation_f1_weighted", -1.0
        )
        if val_f1 >= current_val:
            state["best_candidate_id"] = entry["id"]
    else:
        state["best_candidate_id"] = entry["id"]
    ws.save_state(state)

    ws.log_step(
        "tune_hyperparameters",
        {
            "target_column": target_column,
            "algorithm": algorithm,
            "random_state": random_state,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "grid_combinations": combo_count,
        },
        len(df),
        len(df),
        f"Tuned {algorithm}: CV F1={cv_score:.4f}, val F1={val_f1:.4f}",
    )
    return {
        "candidate_id": entry["id"],
        "algorithm": algorithm,
        "training_method": "tuned",
        "best_params": best_params,
        "best_cv_score": cv_score,
        "validation_f1_weighted": val_f1,
        "grid_combinations": combo_count,
        "artifact_path": entry.get("artifact_path"),
        "stage": "trained",
    }


def list_training_candidates(target_column: str) -> dict[str, Any]:
    """List stored training candidates and the current best selection."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") not in {"trained", "profiled"}:
        raise TrainingError(
            f"Expected stage 'profiled' or 'trained', got '{state.get('stage')}'. "
            "Call train_baseline or tune_hyperparameters first."
        )

    candidates = ws.list_candidates()
    return {
        "target_column": target_column,
        "candidates": candidates,
        "best_candidate_id": state.get("best_candidate_id"),
        "candidate_count": len(candidates),
        "max_candidates": ws.MAX_CANDIDATES,
    }


def _select_best_candidate(state: dict[str, Any]) -> str:
    candidates = state.get("candidates", [])
    if not candidates:
        raise TrainingError("No training candidates found. Train at least one model first.")

    best_id = state.get("best_candidate_id")
    if best_id and any(c["id"] == best_id for c in candidates):
        return best_id

    def _score(candidate: dict[str, Any]) -> float:
        metrics = candidate.get("metrics", {})
        return float(
            metrics.get("validation_f1_weighted")
            or metrics.get("best_cv_score")
            or -1.0
        )

    best = max(candidates, key=_score)
    return best["id"]


def _build_training_report(
    *,
    target_column: str,
    best_candidate: dict[str, Any],
    bundle: dict[str, Any],
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = best_candidate.get("metrics", {})
    lines = [
        "# Training Log",
        "",
        f"- Timestamp: {timestamp}",
        f"- Target column: `{target_column}`",
        f"- Selected candidate: `{best_candidate['id']}`",
        f"- Algorithm: {bundle.get('algorithm', best_candidate.get('algorithm'))}",
        f"- Training method: {bundle.get('training_method', best_candidate.get('training_method'))}",
        f"- Feature count: {len(bundle.get('feature_columns', []))}",
        "",
        "## Selected Model Metrics",
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            f"- Best parameters: {bundle.get('params', best_candidate.get('params'))}",
            f"- Model saved to: `{MODEL_PATH}`",
            "",
            "## Candidates",
            "",
        ]
    )
    for candidate in candidates:
        cand_metrics = candidate.get("metrics", {})
        val_f1 = cand_metrics.get("validation_f1_weighted", "n/a")
        cv = cand_metrics.get("best_cv_score", "n/a")
        marker = " (selected)" if candidate["id"] == best_candidate["id"] else ""
        lines.append(
            f"- **{candidate['id']}** [{candidate.get('algorithm')}, "
            f"{candidate.get('training_method')}]: val F1={val_f1}, CV={cv}{marker}"
        )

    if state.get("steps_applied"):
        lines.extend(["", "## Steps Applied", ""])
        for step in state["steps_applied"]:
            lines.append(
                f"- **{step['tool']}**: {step['summary']} "
                f"({step['rows_before']} → {step['rows_after']} rows)"
            )

    if state.get("decisions"):
        lines.extend(["", "## Agent Decisions", ""])
        for decision in state["decisions"]:
            lines.append(
                f"- **{decision['issue']}** → {decision['choice']}: {decision['rationale']}"
            )

    return "\n".join(lines)


def finalize_training(
    target_column: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Promote the best candidate to model.pkl and write the training log."""
    target_column = _require_target(target_column)
    ensure_output_dirs()
    state = ws.load_state()

    if state.get("stage") != "trained":
        raise TrainingError(
            f"Expected stage 'trained', got '{state.get('stage')}'. "
            "Call train_baseline or tune_hyperparameters first."
        )

    selected_id = candidate_id or _select_best_candidate(state)
    ws.set_best_candidate(selected_id)

    candidates = ws.list_candidates()
    best_candidate = next((c for c in candidates if c["id"] == selected_id), None)
    if best_candidate is None:
        raise TrainingError(f"Candidate '{selected_id}' not found in workspace state.")

    bundle = ws.load_candidate(selected_id)
    bundle["best_params"] = bundle.get("params", best_candidate.get("params"))
    if "best_cv_score" not in bundle:
        bundle["best_cv_score"] = best_candidate.get("metrics", {}).get("best_cv_score")

    joblib.dump(bundle, MODEL_PATH)

    report = _build_training_report(
        target_column=target_column,
        best_candidate=best_candidate,
        bundle=bundle,
        candidates=candidates,
        state=state,
    )
    TRAINING_LOG_PATH.write_text(report, encoding="utf-8")

    state = ws.load_state()
    state["finalized"] = True
    state["best_candidate_id"] = selected_id
    ws.save_state(state)

    ws.log_step(
        "finalize_training",
        {"target_column": target_column, "candidate_id": selected_id},
        len(candidates),
        len(candidates),
        f"Wrote {MODEL_PATH.name} from {selected_id}",
    )
    return {
        "model_path": str(MODEL_PATH),
        "training_log_path": str(TRAINING_LOG_PATH),
        "train_state_path": str(TRAIN_STATE_PATH),
        "target_column": target_column,
        "candidate_id": selected_id,
        "algorithm": bundle.get("algorithm"),
        "training_method": bundle.get("training_method"),
        "best_params": bundle.get("best_params"),
        "validation_f1_weighted": bundle.get("validation_f1_weighted"),
        "best_cv_score": bundle.get("best_cv_score"),
        "finalized": True,
    }
