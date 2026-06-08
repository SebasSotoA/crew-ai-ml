import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel

from crew_ai_ml.pipeline.deploy import run_deployment

logger = logging.getLogger(__name__)


class DeploymentToolInput(BaseModel):
    """Input schema for DeploymentTool. No required arguments."""


class DeploymentTool(BaseTool):
    name: str = "model_deployment"
    description: str = (
        "Deploy the trained and evaluated model to the serving environment."
    )
    args_schema: Type[BaseModel] = DeploymentToolInput

    def _run(self) -> str:
        try:
            summary = run_deployment()
            return json.dumps(summary)
        except Exception:
            logger.exception("Model deployment failed")
            raise
