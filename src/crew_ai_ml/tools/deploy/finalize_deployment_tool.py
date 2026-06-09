import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import finalize_deployment

logger = logging.getLogger(__name__)


class FinalizeDeploymentInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class FinalizeDeploymentTool(BaseTool):
    name: str = "finalize_deployment"
    description: str = (
        "Write output/deployment_report.md and mark deploy workspace finalized. "
        "MANDATORY last tool call on both DEPLOY and DO NOT DEPLOY branches."
    )
    args_schema: Type[BaseModel] = FinalizeDeploymentInput

    def _run(self, target_column: str) -> str:
        try:
            result = finalize_deployment(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("finalize_deployment failed")
            raise
