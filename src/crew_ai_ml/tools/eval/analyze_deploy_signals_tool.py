import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import analyze_deploy_signals

logger = logging.getLogger(__name__)


class AnalyzeDeploySignalsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class AnalyzeDeploySignalsTool(BaseTool):
    name: str = "analyze_deploy_signals"
    description: str = (
        "Analyze train-test metrics against deployment thresholds and return "
        "threshold-based guidance (suggested verdict). Does not issue the final "
        "verdict — call issue_deploy_verdict after reviewing signals."
    )
    args_schema: Type[BaseModel] = AnalyzeDeploySignalsInput

    def _run(self, target_column: str) -> str:
        try:
            result = analyze_deploy_signals(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("analyze_deploy_signals failed")
            raise
