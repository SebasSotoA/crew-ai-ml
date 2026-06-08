"""Data preparation pipeline: atomic data preparation steps."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    CORR_IRRELEVANCE,
    CORR_REDUNDANCY,
    DATA_DIR,
    FEATURE_METADATA_PATH,
    IMBALANCE_RATIO,
    MIN_PREP_ROWS,
    PREP_REPORT_PATH,
    PREP_STATE_PATH,
    PREP_WORKING_PATH,
    PROJECT_ROOT,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.feature_transform import build_input_schema
from crew_ai_ml.pipeline import prep_workspace as ws

ID_COLUMN_PATTERNS = (
    r"\bid\b",
    r"\bname\b",
    r"\bphone\b",
    r"\baddress\b",
    r"\bdocument\b",
    r"\bdoc\b",
    r"\bemail\b",
    r"\bpassport\b",
    r"\bssn\b",
    r"\bcustomer_?id\b",
    r"\buser_?id\b",
    r"\bpatient_?id\b",
    r"^unnamed",
)


class DataPreparationError(Exception):
    """Raised when data preparation fails."""


def resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve dataset path from agent input, cwd, project root, or data/."""
    requested = dataset_path.strip() if dataset_path else ""
    if not requested:
        requested = os.getenv("DATASET_PATH", "").strip()
    if not requested:
        kickoff = ws.get_kickoff_dataset_path()
        if kickoff:
            requested = kickoff

    if not requested:
        raise DataPreparationError(
            "dataset_path is required (tool arg, DATASET_PATH env, or crew kickoff inputs)."
        )

    raw = Path(requested)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                raw,
                Path.cwd() / raw,
                PROJECT_ROOT / raw,
                DATA_DIR / raw.name,
                PROJECT_ROOT / "data" / raw.name,
            ]
        )

    for extra in (
        os.getenv("DATASET_PATH", "").strip(),
        ws.get_kickoff_dataset_path() or "",
    ):
        if not extra:
            continue
        extra_path = Path(extra)
        candidates.extend(
            [
                extra_path,
                Path.cwd() / extra_path,
                PROJECT_ROOT / extra_path,
                DATA_DIR / extra_path.name,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    tried = ", ".join(str(c) for c in candidates[:8])
    raise DataPreparationError(f"Dataset not found: '{requested}'. Tried: {tried}")


def _load_dataset_file(dataset_path: str) -> pd.DataFrame:
    path = resolve_dataset_path(dataset_path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise DataPreparationError(
            f"Unsupported file format '{path.suffix}'. Use CSV or Excel."
        )

    if df.empty:
        raise DataPreparationError("Dataset is empty after loading.")

    return df


def _integrate_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(col).strip() for col in data.columns]

    for col in data.columns:
        if data[col].dtype == object or str(data[col].dtype) == "category":
            data[col] = (
                data[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            )
    return data


def _detect_id_like_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    dropped: list[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        normalized = col.strip().lower()
        if any(re.search(pattern, normalized) for pattern in ID_COLUMN_PATTERNS):
            dropped.append(col)
    return dropped


def _columns_to_exclude(
    df: pd.DataFrame,
    target_column: str,
    state: dict[str, Any],
) -> list[str]:
    """Union of agent-dropped columns and auto-detected id-like columns."""
    manual = state.get("dropped_columns", [])
    auto = _detect_id_like_columns(df, target_column)
    return list(dict.fromkeys(manual + auto))


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _encode_target_for_correlation(data: pd.DataFrame, target_column: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(data[target_column]):
        return data[target_column].astype(float).values
    return LabelEncoder().fit_transform(data[target_column].astype(str))


def _one_hot_encode(
    df: pd.DataFrame,
    target_column: str,
    strategy: str = "one_hot_drop_first",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    data = df.copy()
    categorical_cols = [c for c in data.columns if c != target_column and _is_categorical(data[c])]
    dummy_groups: dict[str, list[str]] = {}

    for col in categorical_cols:
        n_unique = data[col].nunique(dropna=True)
        if strategy == "one_hot_full":
            drop_first = False
        else:
            drop_first = n_unique == 2
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=drop_first, dtype=int)
        dummy_groups[col] = list(dummies.columns)
        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

    feature_cols = [c for c in data.columns if c != target_column]
    return data[feature_cols + [target_column]], dummy_groups


def _drop_redundant_features(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = CORR_REDUNDANCY,
) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    feature_cols = [
        c
        for c in data.columns
        if c != target_column and pd.api.types.is_numeric_dtype(data[c])
    ]
    if len(feature_cols) < 2:
        return data, []

    encoded_target = _encode_target_for_correlation(data, target_column)
    corr = data[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()

    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for partner in high_corr:
            if col in to_drop or partner in to_drop:
                continue
            target_corr_col = abs(np.corrcoef(data[col].values, encoded_target)[0, 1])
            target_corr_partner = abs(np.corrcoef(data[partner].values, encoded_target)[0, 1])
            drop_col = col if target_corr_col <= target_corr_partner else partner
            to_drop.add(drop_col)

    dropped = sorted(to_drop)
    if dropped:
        data = data.drop(columns=dropped)
    return data, dropped


def _drop_irrelevant_features(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = CORR_IRRELEVANCE,
) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    dropped: list[str] = []
    encoded_target = _encode_target_for_correlation(data, target_column)

    for col in [c for c in data.columns if c != target_column]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        corr = abs(np.corrcoef(data[col].values, encoded_target)[0, 1])
        if np.isnan(corr) or corr < threshold:
            dropped.append(col)

    if dropped:
        data = data.drop(columns=dropped)
    return data, dropped


def _check_imbalance(y: pd.Series, imbalance_ratio: float) -> tuple[bool, dict[str, Any]]:
    counts = y.value_counts()
    if len(counts) < 2:
        return False, {"class_counts": counts.to_dict(), "imbalanced": False}

    majority = int(counts.max())
    minority = int(counts.min())
    ratio = minority / majority if majority else 1.0
    imbalanced = ratio < imbalance_ratio
    return imbalanced, {
        "class_counts": counts.to_dict(),
        "majority_count": majority,
        "minority_count": minority,
        "minority_to_majority_ratio": round(ratio, 4),
        "imbalanced": imbalanced,
        "threshold": imbalance_ratio,
    }


def _apply_balance(
    X: pd.DataFrame,
    y: np.ndarray,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    minority_count = int(np.bincount(y).min())
    k_neighbors = max(1, min(5, minority_count - 1))

    if method == "none":
        return X, y, "none"

    if method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, "undersample"

    cat_indices = [
        idx for idx, col in enumerate(X.columns) if set(X[col].unique()) <= {0, 1}
    ]

    if method == "smotenc" or (method == "smote" and cat_indices):
        if cat_indices:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                k_neighbors=k_neighbors,
                random_state=42,
            )
            applied = "smotenc"
        else:
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
            applied = "smote"
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, applied

    sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, "smote"


def _require_target(target_column: str) -> str:
    if not target_column or not str(target_column).strip():
        raise DataPreparationError("target_column must be a non-empty string.")
    return target_column.strip()


def _require_raw_stage() -> dict[str, Any]:
    state = ws.load_state()
    if state.get("stage") != "raw":
        raise DataPreparationError(
            f"Expected stage 'raw', got '{state.get('stage')}'. "
            "encode_features locks the dataset to encoded stage."
        )
    return state


def load_dataset(dataset_path: str, target_column: str) -> dict[str, Any]:
    """Load dataset into prep workspace."""
    ensure_output_dirs()
    target_column = _require_target(target_column)
    resolved_path = resolve_dataset_path(dataset_path)
    resolved_str = str(resolved_path)
    ws.reset_workspace(resolved_str, target_column)

    raw_df = _load_dataset_file(resolved_str)
    if target_column not in raw_df.columns:
        raise DataPreparationError(
            f"Target column '{target_column}' not found. Available: {list(raw_df.columns)}"
        )

    df = _integrate_data(raw_df)
    ws.write_working(df)

    summary = (
        f"Loaded {len(df)} rows, {len(df.columns)} columns from {resolved_str}"
    )
    ws.log_step(
        "load_dataset",
        {
            "dataset_path": resolved_str,
            "dataset_path_requested": dataset_path,
            "target_column": target_column,
        },
        len(raw_df),
        len(df),
        summary,
    )
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "target_column": target_column,
        "dataset_path": resolved_str,
        "dataset_path_requested": dataset_path,
        "workspace_path": str(PREP_WORKING_PATH),
    }


def profile_dataset(target_column: str) -> dict[str, Any]:
    """Profile working dataset and cache EDA to profile.json."""
    target_column = _require_target(target_column)
    df = ws.read_working()
    if target_column not in df.columns:
        raise DataPreparationError(f"Target '{target_column}' not in working data.")

    columns_info: list[dict[str, Any]] = []
    for col in df.columns:
        missing_pct = float(df[col].isna().mean())
        info: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(missing_pct, 4),
            "nunique": int(df[col].nunique(dropna=True)),
        }
        if col != target_column and _is_categorical(df[col]):
            info["type"] = "categorical"
            info["top_values"] = df[col].value_counts(dropna=False).head(5).to_dict()
        elif col != target_column:
            info["type"] = "numeric"
            info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
        columns_info.append(info)

    class_counts = df[target_column].value_counts(dropna=False).to_dict()
    majority = max(class_counts.values()) if class_counts else 0
    minority = min(class_counts.values()) if class_counts else 0
    class_ratio = minority / majority if majority else 1.0

    id_like = _detect_id_like_columns(df, target_column)
    has_missing = any(c["missing_pct"] > 0 for c in columns_info if c["name"] != target_column)
    has_categorical = any(c.get("type") == "categorical" for c in columns_info if c["name"] != target_column)

    profile = {
        "rows": len(df),
        "columns": columns_info,
        "target_column": target_column,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "minority_to_majority_ratio": round(class_ratio, 4),
        "imbalance_threshold": IMBALANCE_RATIO,
        "recommendations": {
            "id_like_columns": id_like,
            "apply_imputation": has_missing,
            "suggested_encoding": "one_hot_drop_first",
            "suggested_balance": (
                "smotenc"
                if class_ratio < IMBALANCE_RATIO and has_categorical
                else "smote"
                if class_ratio < IMBALANCE_RATIO
                else "none"
            ),
        },
    }
    ws.write_profile(profile)

    ws.log_step(
        "profile_dataset",
        {"target_column": target_column},
        len(df),
        len(df),
        f"Profiled {len(df)} rows; class ratio {class_ratio:.4f}",
    )
    return profile


def drop_columns(columns: list[str], reason: str, target_column: str) -> dict[str, Any]:
    """Drop specified columns from working dataset."""
    target_column = _require_target(target_column)
    _require_raw_stage()
    df = ws.read_working()

    to_drop = [c for c in columns if c in df.columns and c != target_column]
    rows_before = len(df)
    if to_drop:
        df = df.drop(columns=to_drop)
        state = ws.load_state()
        existing = state.get("dropped_columns", [])
        state["dropped_columns"] = list(dict.fromkeys(existing + to_drop))
        ws.save_state(state)

    ws.write_working(df)
    ws.log_step(
        "drop_columns",
        {"columns": to_drop, "reason": reason},
        rows_before,
        len(df),
        f"Dropped {len(to_drop)} columns: {to_drop}",
    )
    return {"dropped": to_drop, "reason": reason, "remaining_columns": list(df.columns)}


def handle_outliers(
    strategy: str,
    target_column: str,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Handle outliers via IQR rules: iqr_remove, iqr_cap, or skip."""
    target_column = _require_target(target_column)
    _require_raw_stage()
    df = ws.read_working()
    rows_before = len(df)

    if strategy == "skip":
        ws.log_step("handle_outliers", {"strategy": "skip"}, rows_before, rows_before, "Skipped")
        return {"strategy": "skip", "rows_removed": 0, "columns_capped": []}

    numeric_cols = [
        c
        for c in (columns or df.select_dtypes(include=[np.number]).columns)
        if c in df.columns and c != target_column and pd.api.types.is_numeric_dtype(df[c])
    ]

    removals = 0
    capped_cols: list[str] = []
    data = df.copy()

    if strategy == "iqr_remove":
        combined_mask = pd.Series(False, index=data.index)
        for col in numeric_cols:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            mask = (data[col] < lower) | (data[col] > upper)
            if mask.any():
                combined_mask |= mask
        if combined_mask.any():
            removals = int(combined_mask.sum())
            data = data.loc[~combined_mask].reset_index(drop=True)
    elif strategy == "iqr_cap":
        for col in numeric_cols:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            before = data[col].copy()
            data[col] = data[col].clip(lower=lower, upper=upper)
            if not before.equals(data[col]):
                capped_cols.append(col)
    else:
        raise DataPreparationError(f"Unknown outlier strategy: {strategy}")

    ws.write_working(data)
    summary = f"strategy={strategy}, removed={removals}, capped={capped_cols}"
    ws.log_step(
        "handle_outliers",
        {"strategy": strategy, "columns": numeric_cols},
        rows_before,
        len(data),
        summary,
    )
    return {
        "strategy": strategy,
        "rows_removed": removals,
        "columns_capped": capped_cols,
        "rows": len(data),
    }


def impute_missing(
    strategy: str,
    target_column: str,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Impute or drop missing values."""
    target_column = _require_target(target_column)
    _require_raw_stage()
    data = ws.read_working().copy()
    rows_before = len(data)
    actions: list[str] = []

    if strategy == "drop_rows":
        data = data.dropna()
        actions.append(f"Dropped rows with any missing; {rows_before - len(data)} removed")
    else:
        cols = columns or [c for c in data.columns if c != target_column]
        numeric_cols = [c for c in cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
        categorical_cols = [c for c in cols if c in data.columns and _is_categorical(data[c])]

        if strategy in {"median", "mode", "knn"}:
            for col in numeric_cols:
                if not data[col].isna().any():
                    continue
                if strategy == "median":
                    data[col] = data[col].fillna(data[col].median())
                    actions.append(f"median imputed {col}")
                elif strategy == "mode":
                    mode_val = data[col].mode(dropna=True)
                    fill = mode_val.iloc[0] if not mode_val.empty else 0
                    data[col] = data[col].fillna(fill)
                    actions.append(f"mode imputed {col}")
            if strategy == "knn" and numeric_cols:
                knn_cols = [c for c in numeric_cols if data[c].isna().any()]
                if knn_cols:
                    imputer = KNNImputer(n_neighbors=min(5, max(1, len(data) - 1)))
                    data[knn_cols] = imputer.fit_transform(data[knn_cols])
                    actions.append(f"knn imputed {knn_cols}")

        for col in categorical_cols:
            if not data[col].isna().any():
                continue
            mode_val = data[col].mode(dropna=True)
            fill = mode_val.iloc[0] if not mode_val.empty else "unknown"
            data[col] = data[col].fillna(fill)
            actions.append(f"mode imputed categorical {col}")

        if data[target_column].isna().any():
            data = data.dropna(subset=[target_column])
            actions.append("dropped rows with missing target")

    ws.write_working(data.reset_index(drop=True))
    ws.log_step(
        "impute_missing",
        {"strategy": strategy, "columns": columns},
        rows_before,
        len(data),
        "; ".join(actions) or "no missing values",
    )
    return {"strategy": strategy, "actions": actions, "rows": len(data)}


def encode_features(strategy: str, target_column: str) -> dict[str, Any]:
    """One-hot encode categoricals and switch workspace to encoded stage."""
    target_column = _require_target(target_column)
    _require_raw_stage()
    df = ws.read_working()
    rows_before = len(df)

    if strategy not in {"one_hot_drop_first", "one_hot_full"}:
        raise DataPreparationError(f"Unknown encoding strategy: {strategy}")

    state = ws.load_state()
    exclude = _columns_to_exclude(df, target_column, state)
    cols_to_remove = [c for c in exclude if c in df.columns]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
    state["dropped_columns"] = exclude

    ws.write_pre_encode(df)
    encoded_df, dummy_groups = _one_hot_encode(df, target_column, strategy=strategy)
    ws.write_working(encoded_df)

    state["stage"] = "encoded"
    state["dummy_groups"] = dummy_groups
    ws.save_state(state)

    feature_cols = [c for c in encoded_df.columns if c != target_column]
    ws.log_step(
        "encode_features",
        {"strategy": strategy},
        rows_before,
        len(encoded_df),
        f"Encoded to {len(feature_cols)} features",
    )
    return {
        "strategy": strategy,
        "feature_count": len(feature_cols),
        "dummy_groups": dummy_groups,
        "feature_columns": feature_cols,
    }


def filter_by_correlation(
    mode: str,
    target_column: str,
    redundancy_threshold: float | None = None,
    irrelevance_threshold: float | None = None,
) -> dict[str, Any]:
    """Filter features by correlation: redundancy, irrelevance, both, or skip."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") != "encoded":
        raise DataPreparationError("encode_features must run before correlation filtering.")

    df = ws.read_working()
    rows_before = len(df)
    red_thresh = redundancy_threshold if redundancy_threshold is not None else CORR_REDUNDANCY
    irr_thresh = irrelevance_threshold if irrelevance_threshold is not None else CORR_IRRELEVANCE

    redundant: list[str] = []
    irrelevant: list[str] = []

    if mode in {"redundancy", "both"}:
        df, redundant = _drop_redundant_features(df, target_column, red_thresh)
    if mode in {"irrelevance", "both"}:
        df, irrelevant = _drop_irrelevant_features(df, target_column, irr_thresh)
    if mode == "skip":
        ws.log_step("filter_by_correlation", {"mode": "skip"}, rows_before, len(df), "Skipped")
        return {"mode": "skip", "redundant_dropped": [], "irrelevant_dropped": []}

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise DataPreparationError("No features remain after correlation filtering.")

    ws.write_working(df)
    ws.log_step(
        "filter_by_correlation",
        {"mode": mode, "redundancy_threshold": red_thresh, "irrelevance_threshold": irr_thresh},
        rows_before,
        len(df),
        f"redundant={redundant}, irrelevant={irrelevant}",
    )
    return {
        "mode": mode,
        "redundant_dropped": redundant,
        "irrelevant_dropped": irrelevant,
        "feature_count": len(feature_cols),
    }


def balance_classes(
    method: str,
    target_column: str,
    imbalance_ratio: float | None = None,
) -> dict[str, Any]:
    """Balance target classes via smote, smotenc, undersample, or none."""
    target_column = _require_target(target_column)
    state = ws.load_state()
    if state.get("stage") != "encoded":
        raise DataPreparationError("encode_features must run before balance_classes.")

    df = ws.read_working()
    rows_before = len(df)
    ratio_threshold = imbalance_ratio if imbalance_ratio is not None else IMBALANCE_RATIO

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols]
    y_series = df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series.astype(str))

    imbalanced, imbalance_info = _check_imbalance(y_series, ratio_threshold)
    applied_method = "none"
    final_X, final_y = X, y_encoded

    if method != "none" and imbalanced:
        final_X, final_y, applied_method = _apply_balance(X, y_encoded, method)
    elif method != "none" and not imbalanced:
        applied_method = "none"

    final_df = final_X.copy()
    final_df[target_column] = label_encoder.inverse_transform(final_y)
    ws.write_working(final_df)

    state = ws.load_state()
    state["label_classes"] = list(label_encoder.classes_)
    ws.save_state(state)

    ws.log_step(
        "balance_classes",
        {"method": method, "imbalance_ratio": ratio_threshold},
        rows_before,
        len(final_df),
        f"applied={applied_method}, imbalanced={imbalanced}",
    )
    return {
        "method_requested": method,
        "method_applied": applied_method,
        "imbalance_info": imbalance_info,
        "rows_before": rows_before,
        "rows_after": len(final_df),
    }


def finalize_preparation(target_column: str) -> dict[str, Any]:
    """Validate and write cleaned_data.csv, feature_metadata.json, prep_report.md."""
    target_column = _require_target(target_column)
    ensure_output_dirs()
    state = ws.load_state()

    if state.get("stage") != "encoded":
        raise DataPreparationError("Pipeline must reach encoded stage before finalize.")

    df = ws.read_working()
    if target_column not in df.columns:
        raise DataPreparationError(f"Target '{target_column}' missing from working data.")

    pre_encode = ws.read_pre_encode()
    exclude = _columns_to_exclude(pre_encode, target_column, state)

    cols_still_present = [c for c in exclude if c in df.columns]
    if cols_still_present:
        df = df.drop(columns=cols_still_present)

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise DataPreparationError("No feature columns available for finalize.")

    overlap = [c for c in exclude if c in feature_cols]
    if overlap:
        raise DataPreparationError(
            f"Excluded columns must not appear in final features: {overlap}. "
            "Re-run encode_features after dropping columns."
        )

    if len(df) < MIN_PREP_ROWS:
        raise DataPreparationError(f"Need at least {MIN_PREP_ROWS} rows, got {len(df)}.")

    n_classes = df[target_column].nunique(dropna=True)
    if n_classes != 2:
        raise DataPreparationError(
            f"Binary classification requires exactly 2 classes, found {n_classes}."
        )

    dummy_groups = state.get("dummy_groups", {})
    input_schema = build_input_schema(
        pre_encode,
        dummy_groups,
        feature_cols,
        excluded_columns=exclude,
    )

    if not input_schema:
        raise DataPreparationError("feature_metadata schema is empty after finalize.")

    df.to_csv(CLEANED_DATA_PATH, index=False)
    FEATURE_METADATA_PATH.write_text(
        json.dumps(input_schema, indent=2, default=str),
        encoding="utf-8",
    )

    report_lines = [
        "# Data Preparation Report",
        "",
        f"- Target column: `{target_column}`",
        f"- Final rows: {len(df)}",
        f"- Final features: {len(feature_cols)}",
        "",
        "## Steps Applied (from prep_state.json)",
        "",
    ]
    for step in state.get("steps_applied", []):
        report_lines.append(
            f"- **{step['tool']}**: {step['summary']} "
            f"({step['rows_before']} → {step['rows_after']} rows)"
        )

    if state.get("decisions"):
        report_lines.extend(["", "## Agent Decisions", ""])
        for d in state["decisions"]:
            report_lines.append(f"- **{d['issue']}** → {d['choice']}: {d['rationale']}")

    PREP_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    state["dropped_columns"] = exclude
    state["finalized"] = True
    ws.save_state(state)

    ws.log_step(
        "finalize_preparation",
        {"target_column": target_column},
        len(df),
        len(df),
        "Wrote cleaned_data.csv and feature_metadata.json",
    )

    return {
        "cleaned_data_path": str(CLEANED_DATA_PATH),
        "feature_metadata_path": str(FEATURE_METADATA_PATH),
        "prep_report_path": str(PREP_REPORT_PATH),
        "prep_state_path": str(PREP_STATE_PATH),
        "final_rows": len(df),
        "feature_count": len(feature_cols),
        "input_schema": input_schema,
    }
