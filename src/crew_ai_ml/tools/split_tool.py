import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import run_split

logger = logging.getLogger(__name__)


class DataSplitToolInput(BaseModel):
    """Input schema for DataSplitTool."""

    target_column: str = Field(..., description="Name of the target column to predict.")


class DataSplitTool(BaseTool):
    name: str = "data_split"
    description: str = (
        "Split prepared data into train, validation, and test sets. "
        "Requires target_column."
    )
    args_schema: Type[BaseModel] = DataSplitToolInput

    def _run(self, target_column: str) -> str:
        try:
            summary = run_split(target_column=target_column)
            return json.dumps(summary)
        except Exception:
            logger.exception("Data split failed for target=%s", target_column)
            raise
