"""Feature schema capture and raw-input transformation for inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_categorical_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def build_input_schema(
    df_before_encode: pd.DataFrame,
    dummy_groups: dict[str, list[str]],
    final_feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Build inference schema from pre-encode data and post-filter feature columns."""
    final_set = set(final_feature_columns)
    schema: list[dict[str, Any]] = []

    for col, all_dummies in dummy_groups.items():
        categories = sorted(
            df_before_encode[col].dropna().unique().tolist(),
            key=lambda x: str(x),
        )
        dummy_columns = [d for d in all_dummies if d in final_set]
        if not dummy_columns:
            continue
        schema.append(
            {
                "name": col,
                "type": "categorical",
                "categories": categories,
                "drop_first": len(categories) == 2,
                "dummy_columns": dummy_columns,
            }
        )

    for col in df_before_encode.columns:
        if col in dummy_groups:
            continue
        if col in final_set and pd.api.types.is_numeric_dtype(df_before_encode[col]):
            schema.append({"name": col, "type": "numeric"})

    return schema


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
        for category in categories[1:]:
            if normalized == _normalize_categorical_value(category):
                dummy_name = f"{prefix}_{category}"
                if dummy_name in row_data:
                    row_data[dummy_name] = 1.0
                return
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
