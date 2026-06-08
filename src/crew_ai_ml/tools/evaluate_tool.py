import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import run_evaluation

logger = logging.getLogger(__name__)


class ModelEvaluationToolInput(BaseModel):
    """Input schema for ModelEvaluationTool."""

    target_column: str = Field(..., description="Name of the target column to predict.")


class ModelEvaluationTool(BaseTool):
    name: str = "model_evaluation"
    description: str = (
        "Evaluate the trained model on held-out test data. "
        "Requires target_column."
    )
    args_schema: Type[BaseModel] = ModelEvaluationToolInput

    def _run(self, target_column: str) -> str:
        try:
            summary = run_evaluation(target_column=target_column)
            return json.dumps(summary)
        except Exception:
            logger.exception("Model evaluation failed for target=%s", target_column)
            raise
