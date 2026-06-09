import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import finalize_evaluation

logger = logging.getLogger(__name__)


class FinalizeEvaluationInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class FinalizeEvaluationTool(BaseTool):
    name: str = "finalize_evaluation"
    description: str = (
        "Write output/eval_report.md, update model.pkl evaluation_metrics, and "
        "mark the evaluation workspace finalized. MANDATORY last tool call."
    )
    args_schema: Type[BaseModel] = FinalizeEvaluationInput

    def _run(self, target_column: str) -> str:
        try:
            result = finalize_evaluation(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("finalize_evaluation failed")
            raise
