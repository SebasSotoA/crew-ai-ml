import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import generate_eval_plots

logger = logging.getLogger(__name__)


class GenerateEvalPlotsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    plots: list[str] = Field(
        ...,
        description="Plots to generate, e.g. ['confusion_matrix', 'roc_curve'].",
    )


class GenerateEvalPlotsTool(BaseTool):
    name: str = "generate_eval_plots"
    description: str = (
        "Generate diagnostic plots (confusion matrix, ROC curve) on the test set. "
        "Call after compute_eval_metrics."
    )
    args_schema: Type[BaseModel] = GenerateEvalPlotsInput

    def _run(self, target_column: str, plots: list[str]) -> str:
        try:
            result = generate_eval_plots(target_column=target_column, plots=plots)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("generate_eval_plots failed")
            raise
