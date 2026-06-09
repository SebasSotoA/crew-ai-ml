import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import validate_cleaned_data

logger = logging.getLogger(__name__)


class ValidateCleanedDataInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ValidateCleanedDataTool(BaseTool):
    name: str = "validate_cleaned_data"
    description: str = (
        "Validate output/cleaned_data.csv is ready for splitting and reset split workspace. "
        "Always call this first."
    )
    args_schema: Type[BaseModel] = ValidateCleanedDataInput

    def _run(self, target_column: str) -> str:
        try:
            result = validate_cleaned_data(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("validate_cleaned_data failed")
            raise
