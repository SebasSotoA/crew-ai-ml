import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import validate_eval_inputs

logger = logging.getLogger(__name__)


class ValidateEvalInputsInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ValidateEvalInputsTool(BaseTool):
    name: str = "validate_eval_inputs"
    description: str = (
        "Validate model.pkl and train/test CSVs are ready for evaluation and reset "
        "the evaluation workspace. Always call this first."
    )
    args_schema: Type[BaseModel] = ValidateEvalInputsInput

    def _run(self, target_column: str) -> str:
        try:
            result = validate_eval_inputs(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("validate_eval_inputs failed")
            raise
