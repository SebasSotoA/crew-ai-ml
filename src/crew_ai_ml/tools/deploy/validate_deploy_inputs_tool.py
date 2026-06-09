import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import validate_deploy_inputs

logger = logging.getLogger(__name__)


class ValidateDeployInputsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ValidateDeployInputsTool(BaseTool):
    name: str = "validate_deploy_inputs"
    description: str = (
        "Validate evaluation report, finalized eval state, and model.pkl are ready "
        "for deployment; reset deploy workspace. Always call this first."
    )
    args_schema: Type[BaseModel] = ValidateDeployInputsInput

    def _run(self, target_column: str) -> str:
        try:
            result = validate_deploy_inputs(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("validate_deploy_inputs failed")
            raise
