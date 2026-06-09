import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import validate_train_data

logger = logging.getLogger(__name__)


class ValidateTrainDataInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ValidateTrainDataTool(BaseTool):
    name: str = "validate_train_data"
    description: str = (
        "Validate output/train.csv is ready for modeling and reset training workspace. "
        "Always call this first."
    )
    args_schema: Type[BaseModel] = ValidateTrainDataInput

    def _run(self, target_column: str) -> str:
        try:
            result = validate_train_data(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("validate_train_data failed")
            raise
