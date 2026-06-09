import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import train_baseline

logger = logging.getLogger(__name__)


class TrainBaselineInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    algorithm: str = Field(
        default="random_forest",
        description="Algorithm key (e.g. random_forest, logistic_regression, gradient_boosting).",
    )
    params_json: str = Field(
        ...,
        description=(
            "Required JSON object of estimator kwargs (use algorithm_catalog tunable param "
            "names from profile_train_data). Include random_state when reproducibility matters."
        ),
    )


class TrainBaselineTool(BaseTool):
    name: str = "train_baseline"
    description: str = (
        "Train a baseline model with agent-defined hyperparameters and store as a candidate. "
        "params_json is required — choose param names from profile algorithm_catalog. "
        "Call after profile_train_data."
    )
    args_schema: Type[BaseModel] = TrainBaselineInput

    def _run(
        self,
        target_column: str,
        params_json: str,
        algorithm: str = "random_forest",
    ) -> str:
        try:
            result = train_baseline(
                target_column=target_column,
                algorithm=algorithm,
                params=params_json,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("train_baseline failed")
            raise
