import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import profile_deploy_context

logger = logging.getLogger(__name__)


class ProfileDeployContextInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ProfileDeployContextTool(BaseTool):
    name: str = "profile_deploy_context"
    description: str = (
        "Profile deployment context from evaluation artifacts and model bundle. "
        "Returns branch-specific workflow recommendations. Call after validate_deploy_inputs."
    )
    args_schema: Type[BaseModel] = ProfileDeployContextInput

    def _run(self, target_column: str) -> str:
        try:
            result = profile_deploy_context(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("profile_deploy_context failed")
            raise
