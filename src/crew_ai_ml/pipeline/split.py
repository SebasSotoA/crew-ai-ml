"""Train/test split pipeline: atomic split steps."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    MIN_PREP_ROWS,
    SPLIT_REPORT_PATH,
    SPLIT_STATE_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline import split_workspace as ws
from crew_ai_ml.pipeline.validators import feature_null_report, null_report_message


class SplitError(Exception):
    """Raised when data splitting fails."""


def _require_target(target_column: str) -> str:
    requested = target_column.strip() if target_column else ""
    if not requested:
        kickoff = ws.get_kickoff_target_column()
        if kickoff:
            requested = kickoff
    if not requested:
        raise SplitError(
            "target_column must be a non-empty string (tool arg or crew kickoff inputs)."
        )
    return requested.strip()


def validate_cleaned_data(target_column: str) -> dict[str, Any]:
    """Validate cleaned data is ready for splitting and reset split workspace."""
    ensure_output_dirs()
    target_column = _require_target(target_column)
    ws.reset_workspace(target_column)

    if not CLEANED_DATA_PATH.exists():
        raise SplitError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}. Run data preparation first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    if target_column not in df.columns:
        raise SplitError(
            f"Target column '{target_column}' not found in cleaned data. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < MIN_PREP_ROWS:
        raise SplitError(f"Need at least {MIN_PREP_ROWS} rows, got {len(df)}.")

    n_classes = df[target_column].nunique(dropna=True)
    if n_classes != 2:
        raise SplitError(
            f"Binary classification requires exactly 2 classes, found {n_classes}."
        )

    feature_cols = [c for c in df.columns if c != target_column]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SplitError(
            f"All features must be numeric after preparation. Non-numeric: {non_numeric}"
        )

    if not feature_cols:
        raise SplitError("No feature columns found in cleaned data.")

    nulls = feature_null_report(df, target_column)
    if nulls:
        raise SplitError(
            null_report_message(
                nulls,
                "Re-run data preparation and call impute_missing before finalize_preparation.",
            )
        )

    state = ws.load_state()
    state["stage"] = "validated"
    ws.save_state(state)

    ws.log_step(
        "validate_cleaned_data",
        {"target_column": target_column},
        len(df),
        len(df),
        f"Validated {len(df)} rows, {len(feature_cols)} numeric features, binary target",
    )
    return {
        "target_column": target_column,
        "rows": len(df),
        "feature_count": len(feature_cols),
        "class_counts": df[target_column].value_counts().to_dict(),
        "stage": "validated",
    }


def profile_split_data(target_column: str) -> dict[str, Any]:
    """Profile cleaned data and recommend split parameters."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    stage = state.get("stage")
    if stage != "validated":
        hint = (
            "Stale split state from a prior run was cleared; call validate_cleaned_data first."
            if stage == "uninitialized"
            else "Call validate_cleaned_data first."
        )
        raise SplitError(
            f"Expected stage 'validated', got '{stage}'. {hint}"
        )

    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    if target_column not in df.columns:
        raise SplitError(f"Target '{target_column}' not in cleaned data.")

    class_counts = df[target_column].value_counts().to_dict()
    majority = max(class_counts.values()) if class_counts else 0
    minority = min(class_counts.values()) if class_counts else 0
    rows = len(df)

    if rows < 5000:
        recommended_test_size = 0.2
    elif rows < 20000:
        recommended_test_size = 0.25
    else:
        recommended_test_size = 0.3

    stratify_recommended = minority >= 2

    profile = {
        "rows": rows,
        "target_column": target_column,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "majority_count": int(majority),
        "minority_count": int(minority),
        "minority_to_majority_ratio": round(minority / majority, 4) if majority else 1.0,
        "recommendations": {
            "test_size": recommended_test_size,
            "stratify_recommended": stratify_recommended,
            "random_state": 42,
        },
    }
    ws.write_profile(profile)

    state = ws.load_state()
    state["stage"] = "profiled"
    ws.save_state(state)

    ws.log_step(
        "profile_split_data",
        {"target_column": target_column},
        rows,
        rows,
        f"Profiled split; recommended test_size={recommended_test_size}, "
        f"stratify={stratify_recommended}",
    )
    return profile


def split_train_test(
    target_column: str,
    test_size: float,
    stratify: bool,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split cleaned data into train/test holds."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") not in {"profiled", "validated_split"}:
        raise SplitError(
            f"Expected stage 'profiled' or 'validated_split', got '{state.get('stage')}'. "
            "Call profile_split_data first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    if target_column not in df.columns:
        raise SplitError(f"Target '{target_column}' not in cleaned data.")

    class_counts = df[target_column].value_counts().to_dict()
    min_class = df[target_column].value_counts().min()
    if stratify and min_class < 2:
        raise SplitError(
            f"Cannot stratify: class with only {min_class} sample(s). "
            f"Distribution: {class_counts}"
        )

    stratify_series = df[target_column] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_series,
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    ws.write_train_hold(train_df)
    ws.write_test_hold(test_df)

    state = ws.load_state()
    state["stage"] = "split"
    state["test_size"] = test_size
    state["stratify"] = stratify
    state["random_state"] = random_state
    ws.save_state(state)

    ws.log_step(
        "split_train_test",
        {
            "target_column": target_column,
            "test_size": test_size,
            "stratify": stratify,
            "random_state": random_state,
        },
        len(df),
        len(train_df) + len(test_df),
        f"Split into {len(train_df)} train / {len(test_df)} test rows",
    )
    return {
        "target_column": target_column,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": test_size,
        "stratify": stratify,
        "random_state": random_state,
        "train_class_counts": train_df[target_column].value_counts().to_dict(),
        "test_class_counts": test_df[target_column].value_counts().to_dict(),
    }


def validate_split(target_column: str, max_class_drift: float = 0.05) -> dict[str, Any]:
    """Validate train/test class proportions from holds are within drift tolerance."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") != "split":
        raise SplitError(
            f"Expected stage 'split', got '{state.get('stage')}'. "
            "Call split_train_test first."
        )

    train_df = ws.read_train_hold()
    test_df = ws.read_test_hold()
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise SplitError(f"Target '{target_column}' missing from train or test hold.")

    train_props = train_df[target_column].value_counts(normalize=True)
    test_props = test_df[target_column].value_counts(normalize=True)
    all_classes = sorted(set(train_props.index) | set(test_props.index), key=str)

    drift_by_class: dict[str, float] = {}
    for cls in all_classes:
        train_pct = float(train_props.get(cls, 0.0))
        test_pct = float(test_props.get(cls, 0.0))
        drift_by_class[str(cls)] = round(abs(train_pct - test_pct), 4)

    max_drift = max(drift_by_class.values()) if drift_by_class else 0.0
    if max_drift > max_class_drift:
        raise SplitError(
            f"Class drift {max_drift:.4f} exceeds max {max_class_drift}. "
            f"Per-class drift: {drift_by_class}"
        )

    state = ws.load_state()
    state["stage"] = "validated_split"
    ws.save_state(state)

    ws.log_step(
        "validate_split",
        {"target_column": target_column, "max_class_drift": max_class_drift},
        len(train_df) + len(test_df),
        len(train_df) + len(test_df),
        f"Validated split; max class drift={max_drift:.4f}",
    )
    return {
        "target_column": target_column,
        "max_class_drift": round(max_drift, 4),
        "max_allowed_drift": max_class_drift,
        "drift_by_class": drift_by_class,
        "train_class_distribution": train_props.round(4).to_dict(),
        "test_class_distribution": test_props.round(4).to_dict(),
        "stage": "validated_split",
    }


def finalize_split(target_column: str) -> dict[str, Any]:
    """Copy holds to final train/test CSVs and write split report."""
    target_column = _require_target(target_column)
    ensure_output_dirs()
    state = ws.load_state()

    if state.get("stage") != "validated_split":
        raise SplitError(
            f"Expected stage 'validated_split', got '{state.get('stage')}'. "
            "Call validate_split first."
        )

    train_df = ws.read_train_hold()
    test_df = ws.read_test_hold()
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise SplitError(f"Target '{target_column}' missing from train or test hold.")

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    profile: dict[str, Any] = {}
    try:
        profile = ws.read_profile()
    except ws.SplitWorkspaceError:
        pass

    report_lines = [
        "# Train/Test Split Report",
        "",
        f"- Target column: `{target_column}`",
        f"- Train rows: {len(train_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Test size: {state.get('test_size')}",
        f"- Stratified: {state.get('stratify')}",
        f"- Random state: {state.get('random_state')}",
        "",
        "## Class Counts",
        "",
        f"- Train: {train_df[target_column].value_counts().to_dict()}",
        f"- Test: {test_df[target_column].value_counts().to_dict()}",
        "",
        "## Steps Applied (from split_state.json)",
        "",
    ]
    for step in state.get("steps_applied", []):
        report_lines.append(
            f"- **{step['tool']}**: {step['summary']} "
            f"({step['rows_before']} → {step['rows_after']} rows)"
        )

    if state.get("decisions"):
        report_lines.extend(["", "## Agent Decisions", ""])
        for decision in state["decisions"]:
            report_lines.append(
                f"- **{decision['issue']}** → {decision['choice']}: {decision['rationale']}"
            )

    if profile:
        report_lines.extend(["", "## Profile Recommendations", ""])
        recs = profile.get("recommendations", {})
        report_lines.append(f"- Recommended test_size: {recs.get('test_size')}")
        report_lines.append(f"- Stratify recommended: {recs.get('stratify_recommended')}")

    SPLIT_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    state["finalized"] = True
    ws.save_state(state)

    ws.log_step(
        "finalize_split",
        {"target_column": target_column},
        len(train_df) + len(test_df),
        len(train_df) + len(test_df),
        f"Wrote {TRAIN_DATA_PATH.name} and {TEST_DATA_PATH.name}",
    )
    return {
        "train_path": str(TRAIN_DATA_PATH),
        "test_path": str(TEST_DATA_PATH),
        "split_report_path": str(SPLIT_REPORT_PATH),
        "split_state_path": str(SPLIT_STATE_PATH),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "finalized": True,
    }
