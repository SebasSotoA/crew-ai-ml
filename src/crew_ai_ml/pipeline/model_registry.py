"""Algorithm registry for model training and hyperparameter search."""

from __future__ import annotations

import json
from functools import reduce
from operator import mul
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

__all__ = [
    "ALGORITHMS",
    "count_grid_combinations",
    "get_estimator",
    "parse_estimator_params",
    "parse_param_grid",
    "validate_algorithm",
]

ALGORITHMS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "estimator_class": RandomForestClassifier,
        "tunable_params": [
            {"name": "n_estimators", "description": "Number of trees in the forest."},
            {"name": "max_depth", "description": "Maximum depth of each tree; None means unlimited."},
            {"name": "min_samples_leaf", "description": "Minimum samples required at a leaf node."},
            {"name": "max_samples", "description": "Fraction of samples drawn for each tree (bootstrap)."},
            {"name": "criterion", "description": "Split quality measure: 'gini' or 'entropy'."},
        ],
        "max_grid_combinations": 200,
    },
    "logistic_regression": {
        "estimator_class": LogisticRegression,
        "tunable_params": [
            {"name": "C", "description": "Inverse regularization strength; smaller values mean stronger regularization."},
            {"name": "penalty", "description": "Regularization norm, e.g. 'l1' or 'l2'."},
            {"name": "solver", "description": "Optimization algorithm, e.g. 'lbfgs' or 'liblinear'."},
        ],
        "max_grid_combinations": 200,
    },
    "gradient_boosting": {
        "estimator_class": GradientBoostingClassifier,
        "tunable_params": [
            {"name": "n_estimators", "description": "Number of boosting stages (trees)."},
            {"name": "learning_rate", "description": "Shrinkage applied to each tree's contribution."},
            {"name": "max_depth", "description": "Maximum depth of each individual tree."},
            {"name": "min_samples_leaf", "description": "Minimum samples required at a leaf node."},
        ],
        "max_grid_combinations": 200,
    },
}


def validate_algorithm(algorithm: str) -> str:
    """Return normalized algorithm name or raise ValueError."""
    key = algorithm.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: {supported}")
    return key


def get_estimator(algorithm: str, params: dict[str, Any]):
    """Instantiate a sklearn estimator using only the provided params."""
    key = validate_algorithm(algorithm)
    if not params:
        raise ValueError("estimator params must be a non-empty dict.")
    entry = ALGORITHMS[key]
    return entry["estimator_class"](**params)


def _normalize_grid_values(obj: Any) -> Any:
    """Recursively convert JSON null to Python None."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_normalize_grid_values(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _normalize_grid_values(v) for k, v in obj.items()}
    return obj


def parse_estimator_params(json_str: str) -> dict[str, Any]:
    """Parse flat JSON estimator kwargs; JSON null becomes None."""
    if not json_str or not str(json_str).strip():
        raise ValueError("estimator params JSON must be a non-empty string.")
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("estimator params JSON must decode to an object.")
    normalized = _normalize_grid_values(parsed)
    if not normalized:
        raise ValueError("estimator params must be a non-empty dict.")
    return normalized


def parse_param_grid(json_str: str) -> dict[str, list[Any]]:
    """Parse a JSON hyperparameter grid string; JSON null becomes None."""
    if not json_str or not str(json_str).strip():
        raise ValueError("param_grid JSON must be a non-empty string.")
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("param_grid JSON must decode to an object.")
    normalized = _normalize_grid_values(parsed)
    for key, value in normalized.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Grid entry '{key}' must be a non-empty list.")
    return normalized


def count_grid_combinations(grid: dict[str, list[Any]]) -> int:
    """Count total combinations in a parameter grid."""
    if not grid:
        return 0
    lengths = [len(values) for values in grid.values()]
    return reduce(mul, lengths, 1)
