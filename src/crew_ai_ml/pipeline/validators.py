"""Shared dataframe validation helpers for pipeline stages."""

from __future__ import annotations

import pandas as pd


def feature_frame_null_report(df: pd.DataFrame) -> dict[str, int]:
    """Return {column: null_count} for any column in df with missing values."""
    return {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }


def feature_null_report(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    """Return {column: null_count} for feature columns with any missing values."""
    feature_cols = [c for c in df.columns if c != target_column]
    return {
        col: int(df[col].isna().sum())
        for col in feature_cols
        if df[col].isna().any()
    }


def format_null_report(report: dict[str, int]) -> str:
    if not report:
        return ""
    parts = [f"'{col}': {count}" for col, count in sorted(report.items())]
    return "{" + ", ".join(parts) + "}"


def null_report_message(
    report: dict[str, int],
    action_hint: str,
) -> str:
    return (
        f"Feature columns contain missing values: {format_null_report(report)}. "
        f"{action_hint}"
    )
