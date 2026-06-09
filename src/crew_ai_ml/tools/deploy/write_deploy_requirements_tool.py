import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import write_deploy_requirements

logger = logging.getLogger(__name__)


class WriteDeployRequirementsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class WriteDeployRequirementsTool(BaseTool):
    name: str = "write_deploy_requirements"
    description: str = (
        "Write output/requirements.txt with Streamlit runtime dependencies. "
        "DEPLOY branch only — call after generate_streamlit_app."
    )
    args_schema: Type[BaseModel] = WriteDeployRequirementsInput

    def _run(self, target_column: str) -> str:
        try:
            result = write_deploy_requirements(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("write_deploy_requirements failed")
            raise
