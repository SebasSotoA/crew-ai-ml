import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import issue_deploy_verdict

logger = logging.getLogger(__name__)


class IssueDeployVerdictInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    verdict: str = Field(
        ...,
        description="Deployment verdict: 'DEPLOY' or 'DO NOT DEPLOY'.",
    )
    rationale: str = Field(
        ...,
        description="Evidence-based rationale citing metrics and threshold guidance.",
    )


class IssueDeployVerdictTool(BaseTool):
    name: str = "issue_deploy_verdict"
    description: str = (
        "Record your agent-owned deployment verdict and rationale after reviewing "
        "analyze_deploy_signals output. Use threshold guidance "
        "(test F1 >= 0.70 and train-test F1 gap <= 0.05 → DEPLOY) but you may "
        "override with documented justification."
    )
    args_schema: Type[BaseModel] = IssueDeployVerdictInput

    def _run(self, target_column: str, verdict: str, rationale: str) -> str:
        try:
            result = issue_deploy_verdict(
                target_column=target_column,
                verdict=verdict,
                rationale=rationale,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("issue_deploy_verdict failed")
            raise
