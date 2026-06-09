import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import validate_split

logger = logging.getLogger(__name__)


class ValidateSplitInput(BaseModel):
    target_column: str = Field(..., description="Target column name.")
    max_class_drift: float = Field(
        default=0.05,
        description="Maximum allowed absolute class proportion drift between train and test.",
    )


class ValidateSplitTool(BaseTool):
    name: str = "validate_split"
    description: str = (
        "Validate train/test class proportions are within drift tolerance. "
        "Call after split_train_test."
    )
    args_schema: Type[BaseModel] = ValidateSplitInput

    def _run(self, target_column: str, max_class_drift: float = 0.05) -> str:
        try:
            result = validate_split(
                target_column=target_column,
                max_class_drift=max_class_drift,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("validate_split failed")
            raise
