import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import compute_eval_metrics

logger = logging.getLogger(__name__)


class ComputeEvalMetricsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    splits: list[str] = Field(
        ...,
        description="Splits to evaluate, e.g. ['train', 'test'].",
    )
    metrics: list[str] = Field(
        ...,
        description=(
            "Metrics to compute, e.g. "
            "['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']."
        ),
    )


class ComputeEvalMetricsTool(BaseTool):
    name: str = "compute_eval_metrics"
    description: str = (
        "Compute classification metrics for requested train and/or test splits. "
        "Call after profile_eval_context using profile recommendations."
    )
    args_schema: Type[BaseModel] = ComputeEvalMetricsInput

    def _run(self, target_column: str, splits: list[str], metrics: list[str]) -> str:
        try:
            result = compute_eval_metrics(
                target_column=target_column,
                splits=splits,
                metrics=metrics,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("compute_eval_metrics failed")
            raise
