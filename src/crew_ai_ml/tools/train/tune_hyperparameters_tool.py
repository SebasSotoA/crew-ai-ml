import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import tune_hyperparameters

logger = logging.getLogger(__name__)


class TuneHyperparametersInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    algorithm: str = Field(
        default="random_forest",
        description="Algorithm key (e.g. random_forest, logistic_regression, gradient_boosting).",
    )
    param_grid_json: str = Field(
        ...,
        description=(
            "Required JSON hyperparameter grid: each key maps to a non-empty list of values. "
            "Use algorithm_catalog tunable param names from profile_train_data."
        ),
    )
    fixed_params_json: str | None = Field(
        default=None,
        description=(
            "Optional JSON object of fixed estimator kwargs held constant during grid search "
            "(e.g. random_state, n_jobs)."
        ),
    )
    cv_folds: int = Field(
        default=5,
        description="Cross-validation folds for GridSearchCV.",
    )
    scoring: str = Field(
        default="f1_weighted",
        description="Scoring metric for GridSearchCV.",
    )


class TuneHyperparametersTool(BaseTool):
    name: str = "tune_hyperparameters"
    description: str = (
        "Run GridSearchCV with an agent-defined param_grid_json and store the best estimator "
        "as a candidate. param_grid_json is required — use algorithm_catalog from profile. "
        "Skip tuning when profile recommendations.tune_recommended is false. "
        "Call after profile_train_data."
    )
    args_schema: Type[BaseModel] = TuneHyperparametersInput

    def _run(
        self,
        target_column: str,
        param_grid_json: str,
        algorithm: str = "random_forest",
        fixed_params_json: str | None = None,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
    ) -> str:
        try:
            result = tune_hyperparameters(
                target_column=target_column,
                algorithm=algorithm,
                param_grid=param_grid_json,
                fixed_params=fixed_params_json,
                cv_folds=cv_folds,
                scoring=scoring,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("tune_hyperparameters failed")
            raise
