import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import run_training

logger = logging.getLogger(__name__)


class ModelTrainingToolInput(BaseModel):
    """Input schema for ModelTrainingTool."""

    target_column: str = Field(..., description="Name of the target column to predict.")
    param_grid: str | None = Field(
        default=None,
        description="Optional JSON string of hyperparameter grid for model tuning.",
    )


class ModelTrainingTool(BaseTool):
    name: str = "model_training"
    description: str = (
        "Train a machine learning model on the split datasets. "
        "Requires target_column; param_grid is optional JSON."
    )
    args_schema: Type[BaseModel] = ModelTrainingToolInput

    def _run(self, target_column: str, param_grid: str | None = None) -> str:
        parsed_param_grid = None
        if param_grid:
            parsed_param_grid = json.loads(param_grid)

        try:
            summary = run_training(
                target_column=target_column,
                param_grid=parsed_param_grid,
            )
            return json.dumps(summary)
        except Exception:
            logger.exception(
                "Model training failed for target=%s param_grid=%s",
                target_column,
                param_grid,
            )
            raise
