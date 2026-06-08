"""Data preparation pipeline: load, clean, encode, and balance the dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    CORR_IRRELEVANCE,
    CORR_REDUNDANCY,
    FEATURE_METADATA_PATH,
    IMBALANCE_RATIO,
    MISSING_THRESHOLD,
    PREP_REPORT_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.feature_transform import build_input_schema

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
)


class DataPreparationError(Exception):
    """Raised when data preparation fails."""


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
    if not path.exists():
        raise DataPreparationError(f"Dataset not found: {path}")

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
    """Strip column names and lowercase categorical values."""
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


def _drop_id_like_columns(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    keep_cols: list[str] = []

    for col in df.columns:
        normalized = col.strip().lower()
        if col == target_column:
            keep_cols.append(col)
            continue
        if any(re.search(pattern, normalized) for pattern in ID_COLUMN_PATTERNS):
            dropped.append(col)
        else:
            keep_cols.append(col)

    return df[keep_cols], dropped


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without optional dependencies."""
    headers = [str(col) for col in frame.columns]
    rows: list[list[str]] = [[str(frame.index[i])] + [str(frame.iloc[i, j]) for j in range(len(headers))] for i in range(len(frame))]
    table_headers = [""] + headers
    divider = ["---"] * len(table_headers)
    lines = [
        "| " + " | ".join(table_headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _statistical_description(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])

    lines = [
        "## Statistical Description",
        "",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        f"- Numeric columns: {len(numeric.columns)}",
        f"- Categorical columns: {len(categorical.columns)}",
        "",
    ]

    if not numeric.empty:
        lines.append("### Numeric Summary")
        lines.append("")
        lines.append(_dataframe_to_markdown_table(numeric.describe().round(4)))
        lines.append("")

    if not categorical.empty:
        lines.append("### Categorical Value Counts")
        lines.append("")
        for col in categorical.columns:
            counts = categorical[col].value_counts(dropna=False).head(10).to_frame("count")
            lines.append(f"**{col}**")
            lines.append(_dataframe_to_markdown_table(counts))
            lines.append("")

    return "\n".join(lines)


def _clean_outliers(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Remove domain-impossible values and document each rule."""
    data = df.copy()
    removals: list[str] = []
    initial_rows = len(data)

    rules: list[tuple[str, pd.Series]] = []

    if "age" in data.columns and pd.api.types.is_numeric_dtype(data["age"]):
        rules.append(("age < 0 or age > 150", (data["age"] < 0) | (data["age"] > 150)))

    if "bmi" in data.columns and pd.api.types.is_numeric_dtype(data["bmi"]):
        rules.append(("bmi < 10 or bmi > 80", (data["bmi"] < 10) | (data["bmi"] > 80)))

    glucose_cols = [c for c in data.columns if "glucose" in c.lower()]
    for col in glucose_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            rules.append(
                (f"{col} < 0 or {col} > 600", (data[col] < 0) | (data[col] > 600))
            )

    bp_cols = [c for c in data.columns if "blood_pressure" in c.lower() or c.lower() in {"bp", "trestbps"}]
    for col in bp_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            rules.append(
                (f"{col} < 40 or {col} > 250", (data[col] < 40) | (data[col] > 250))
            )

    for col in data.select_dtypes(include=[np.number]).columns:
        if col == target_column:
            continue
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            mask = (data[col] < lower) | (data[col] > upper)
            if mask.any():
                rules.append((f"{col} outside 3*IQR [{lower:.2f}, {upper:.2f}]", mask))

    combined_mask = pd.Series(False, index=data.index)
    for rule_name, mask in rules:
        count = int(mask.sum())
        if count:
            removals.append(f"- {rule_name}: removed {count} rows")
            combined_mask |= mask

    if combined_mask.any():
        data = data.loc[~combined_mask].reset_index(drop=True)

    removals.insert(0, f"- Total rows removed: {initial_rows - len(data)} of {initial_rows}")
    return data, removals


def _is_categorical(series: pd.Series) -> bool:
    """True for non-numeric feature columns (object, string, category)."""
    return not pd.api.types.is_numeric_dtype(series)


def _handle_missing_values(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    log: list[str] = []
    initial_rows = len(data)

    row_missing_ratio = data.isna().mean(axis=1)
    high_missing_rows = row_missing_ratio > MISSING_THRESHOLD
    if high_missing_rows.any():
        log.append(
            f"- Dropped {int(high_missing_rows.sum())} rows with >"
            f"{MISSING_THRESHOLD:.0%} missing values"
        )
        data = data.loc[~high_missing_rows].reset_index(drop=True)

    cols_to_drop: list[str] = []
    for col in data.columns:
        if col == target_column:
            continue
        missing_ratio = data[col].isna().mean()
        if missing_ratio > MISSING_THRESHOLD:
            cols_to_drop.append(col)
            log.append(
                f"- Dropped column '{col}' ({missing_ratio:.1%} missing, "
                f"threshold {MISSING_THRESHOLD:.0%})"
            )

    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    numeric_cols = [
        c for c in data.columns if c != target_column and pd.api.types.is_numeric_dtype(data[c])
    ]
    categorical_cols = [
        c for c in data.columns if c != target_column and _is_categorical(data[c])
    ]

    for col in numeric_cols:
        if not data[col].isna().any():
            continue
        skewness = data[col].skew(skipna=True)
        if abs(skewness) > 1:
            strategy = "median"
            imputer = SimpleImputer(strategy="median")
        else:
            strategy = "mean"
            imputer = SimpleImputer(strategy="mean")
        data[col] = imputer.fit_transform(data[[col]]).ravel()
        log.append(f"- Imputed numeric '{col}' with {strategy}")

    for col in categorical_cols:
        if not data[col].isna().any():
            continue
        mode_val = data[col].mode(dropna=True)
        fill_value = mode_val.iloc[0] if not mode_val.empty else "unknown"
        data[col] = data[col].fillna(fill_value)
        log.append(f"- Imputed categorical '{col}' with mode ('{fill_value}')")

    remaining_missing = data.isna().sum().sum()
    if remaining_missing:
        knn_cols = [c for c in numeric_cols if data[c].isna().any()]
        if knn_cols:
            imputer = KNNImputer(n_neighbors=min(5, len(data) - 1))
            data[knn_cols] = imputer.fit_transform(data[knn_cols])
            log.append(f"- Applied KNN imputation to columns: {', '.join(knn_cols)}")

        for col in categorical_cols:
            if data[col].isna().any():
                mode_val = data[col].mode(dropna=True)
                fill_value = mode_val.iloc[0] if not mode_val.empty else "unknown"
                data[col] = data[col].fillna(fill_value)
                log.append(f"- Final mode imputation for '{col}'")

    log.insert(0, f"- Rows after missing-value handling: {len(data)} (started with {initial_rows})")
    return data, log


def _encode_target_for_correlation(data: pd.DataFrame, target_column: str) -> np.ndarray:
    """Return a numeric target vector for correlation calculations."""
    if pd.api.types.is_numeric_dtype(data[target_column]):
        return data[target_column].astype(float).values
    return LabelEncoder().fit_transform(data[target_column].astype(str))


def _one_hot_encode(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Encode categoricals: drop_first=True for binary, False for >2 categories."""
    data = df.copy()
    categorical_cols = [c for c in data.columns if c != target_column and _is_categorical(data[c])]
    dummy_groups: dict[str, list[str]] = {}

    for col in categorical_cols:
        n_unique = data[col].nunique(dropna=True)
        drop_first = n_unique == 2
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=drop_first, dtype=int)
        dummy_groups[col] = list(dummies.columns)
        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

    feature_cols = [c for c in data.columns if c != target_column]
    return data[feature_cols + [target_column]], dummy_groups


def _drop_redundant_features(
    df: pd.DataFrame, target_column: str
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
        high_corr = upper[col][upper[col] > CORR_REDUNDANCY].index.tolist()
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
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    dropped: list[str] = []

    encoded_target = _encode_target_for_correlation(data, target_column)

    for col in [c for c in data.columns if c != target_column]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        corr = abs(np.corrcoef(data[col].values, encoded_target)[0, 1])
        if np.isnan(corr) or corr < CORR_IRRELEVANCE:
            dropped.append(col)

    if dropped:
        data = data.drop(columns=dropped)
    return data, dropped


def _check_imbalance(y: pd.Series) -> tuple[bool, dict[str, Any]]:
    counts = y.value_counts()
    if len(counts) < 2:
        return False, {"class_counts": counts.to_dict(), "imbalanced": False}

    majority = int(counts.max())
    minority = int(counts.min())
    ratio = minority / majority if majority else 1.0
    imbalanced = ratio < IMBALANCE_RATIO
    return imbalanced, {
        "class_counts": counts.to_dict(),
        "majority_count": majority,
        "minority_count": minority,
        "minority_to_majority_ratio": round(ratio, 4),
        "imbalanced": imbalanced,
    }


def _apply_smote(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_feature_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    minority_count = int(np.bincount(y).min())
    k_neighbors = max(1, min(5, minority_count - 1))

    if categorical_feature_indices:
        sampler = SMOTENC(
            categorical_features=categorical_feature_indices,
            k_neighbors=k_neighbors,
            random_state=42,
        )
        method = "SMOTENC"
    else:
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
        method = "SMOTE"

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, method


def run_data_preparation(dataset_path: str, target_column: str) -> dict[str, Any]:
    """
    Execute the full 8-step data preparation pipeline.

    Returns a summary dict with output paths, row/column counts, and step logs.
    """
    ensure_output_dirs()
    report_sections: list[str] = ["# Data Preparation Report", ""]
    step_logs: dict[str, Any] = {}

    if not target_column or not str(target_column).strip():
        raise DataPreparationError("target_column must be a non-empty string.")

    target_column = target_column.strip()

    # Step 1: Load and integrate
    raw_df = _load_dataset(dataset_path)
    if target_column not in raw_df.columns:
        raise DataPreparationError(
            f"Target column '{target_column}' not found. Available: {list(raw_df.columns)}"
        )

    df = _integrate_data(raw_df)
    step_logs["step_1_integration"] = {
        "initial_rows": len(raw_df),
        "initial_columns": list(raw_df.columns),
        "integrated_columns": list(df.columns),
    }
    report_sections.extend(["## Step 1: Load and Integration", "", f"- Source: `{dataset_path}`", f"- Rows loaded: {len(df)}", ""])

    # Step 2: Drop ID-like columns
    df, dropped_ids = _drop_id_like_columns(df, target_column)
    step_logs["step_2_id_columns_dropped"] = dropped_ids
    report_sections.extend(["## Step 2: ID-like Column Removal", ""])
    report_sections.extend(dropped_ids or ["- No ID-like columns removed"])
    report_sections.append("")

    # Step 3: Statistical description
    stats_md = _statistical_description(df)
    step_logs["step_3_statistics"] = {"rows": len(df), "columns": len(df.columns)}
    report_sections.append(stats_md)
    report_sections.append("")

    # Step 4: Outlier cleaning
    df, outlier_removals = _clean_outliers(df, target_column)
    step_logs["step_4_outliers"] = outlier_removals
    report_sections.extend(["## Step 4: Outlier Cleaning", ""])
    report_sections.extend(outlier_removals)
    report_sections.append("")

    # Step 5: Missing values
    df, missing_logs = _handle_missing_values(df, target_column)
    step_logs["step_5_missing_values"] = missing_logs
    report_sections.extend(["## Step 5: Missing Value Treatment", ""])
    report_sections.extend(missing_logs)
    report_sections.append("")

    if df.empty:
        raise DataPreparationError("No rows remain after missing-value handling.")

    # Step 6: One-hot encode, then correlation redundancy
    df_before_encode = df.copy()
    encoded_df, dummy_groups = _one_hot_encode(df, target_column)
    encoded_df, redundant_dropped = _drop_redundant_features(encoded_df, target_column)
    step_logs["step_6_redundancy"] = {
        "dummy_groups": dummy_groups,
        "dropped_features": redundant_dropped,
    }
    report_sections.extend(["## Step 6: Correlation Redundancy Removal", ""])
    report_sections.append(f"- Threshold: |r| > {CORR_REDUNDANCY}")
    report_sections.append(f"- Features removed: {redundant_dropped or 'none'}")
    report_sections.append("")

    # Step 7: Correlation irrelevance
    encoded_df, irrelevant_dropped = _drop_irrelevant_features(encoded_df, target_column)
    step_logs["step_7_irrelevance"] = irrelevant_dropped
    report_sections.extend(["## Step 7: Correlation Irrelevance Removal", ""])
    report_sections.append(f"- Threshold: |r| < {CORR_IRRELEVANCE} with target")
    report_sections.append(f"- Features removed: {irrelevant_dropped or 'none'}")
    report_sections.append("")

    feature_cols = [c for c in encoded_df.columns if c != target_column]
    if not feature_cols:
        raise DataPreparationError("No features remain after correlation filtering.")

    input_schema = build_input_schema(df_before_encode, dummy_groups, feature_cols)
    FEATURE_METADATA_PATH.write_text(
        json.dumps(input_schema, indent=2, default=str),
        encoding="utf-8",
    )

    X = encoded_df[feature_cols]
    y_series = encoded_df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series.astype(str))

    # Step 8: SMOTE / SMOTENC
    imbalanced, imbalance_info = _check_imbalance(y_series)
    smote_method = "none"
    final_X, final_y = X, y_encoded

    if imbalanced:
        cat_indices = [
            idx for idx, col in enumerate(X.columns) if set(X[col].unique()) <= {0, 1}
        ]
        has_categorical_dummies = len(cat_indices) > 0 and len(cat_indices) < len(X.columns)

        if has_categorical_dummies:
            final_X, final_y, smote_method = _apply_smote(X, y_encoded, cat_indices)
        else:
            final_X, final_y, smote_method = _apply_smote(X, y_encoded, None)

    final_df = final_X.copy()
    final_df[target_column] = label_encoder.inverse_transform(final_y)

    step_logs["step_8_balancing"] = {
        **imbalance_info,
        "method": smote_method,
        "rows_before": len(X),
        "rows_after": len(final_df),
    }
    report_sections.extend(["## Step 8: Class Balancing", ""])
    report_sections.append(f"- Class distribution before: {imbalance_info['class_counts']}")
    report_sections.append(f"- Imbalanced: {imbalance_info['imbalanced']}")
    report_sections.append(f"- Method applied: {smote_method}")
    report_sections.append(f"- Rows after balancing: {len(final_df)}")
    report_sections.append("")

    final_df.to_csv(CLEANED_DATA_PATH, index=False)
    PREP_REPORT_PATH.write_text("\n".join(report_sections), encoding="utf-8")

    summary = {
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "cleaned_data_path": str(CLEANED_DATA_PATH),
        "prep_report_path": str(PREP_REPORT_PATH),
        "feature_metadata_path": str(FEATURE_METADATA_PATH),
        "initial_rows": len(raw_df),
        "final_rows": len(final_df),
        "final_columns": list(final_df.columns),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "input_schema": input_schema,
        "label_classes": list(label_encoder.classes_),
        "step_logs": step_logs,
        "imbalance_info": imbalance_info,
        "smote_method": smote_method,
    }
    return summary
