import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import finalize_split

logger = logging.getLogger(__name__)


class FinalizeSplitInput(BaseModel):
    target_column: str = Field(
        ...,
        description="Target column name. Mandatory final step before model training.",
    )


class FinalizeSplitTool(BaseTool):
    name: str = "finalize_split"
    description: str = (
        "Write output/train.csv, output/test.csv, and split report. "
        "MANDATORY last step of data splitting."
    )
    args_schema: Type[BaseModel] = FinalizeSplitInput

    def _run(self, target_column: str) -> str:
        try:
            result = finalize_split(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("finalize_split failed")
            raise
