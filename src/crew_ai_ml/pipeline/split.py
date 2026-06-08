"""Train/test split module."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_dirs,
)


class SplitError(Exception):
    """Raised when data splitting fails."""


def run_split(target_column: str, test_size: float = 0.30, random_state: int = 42) -> dict[str, Any]:
    """
    Perform a stratified 70/30 train-test split on cleaned data.

    Saves train.csv and test.csv and returns a summary with class distributions.
    """
    ensure_output_dirs()

    if not target_column or not str(target_column).strip():
        raise SplitError("target_column must be a non-empty string.")

    target_column = target_column.strip()

    if not CLEANED_DATA_PATH.exists():
        raise SplitError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}. "
            "Run data preparation first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    if target_column not in df.columns:
        raise SplitError(
            f"Target column '{target_column}' not found in cleaned data. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < 10:
        raise SplitError(f"Insufficient rows for stratified split: {len(df)}")

    class_counts = df[target_column].value_counts().to_dict()
    min_class = df[target_column].value_counts().min()
    if min_class < 2:
        raise SplitError(
            f"Cannot stratify: class with only {min_class} sample(s). "
            f"Distribution: {class_counts}"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    train_distribution = train_df[target_column].value_counts(normalize=True).round(4).to_dict()
    test_distribution = test_df[target_column].value_counts(normalize=True).round(4).to_dict()

    return {
        "target_column": target_column,
        "train_path": str(TRAIN_DATA_PATH),
        "test_path": str(TEST_DATA_PATH),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": test_size,
        "overall_class_counts": class_counts,
        "train_class_distribution": train_distribution,
        "test_class_distribution": test_distribution,
        "train_class_counts": train_df[target_column].value_counts().to_dict(),
        "test_class_counts": test_df[target_column].value_counts().to_dict(),
    }
