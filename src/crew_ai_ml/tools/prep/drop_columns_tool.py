import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import drop_columns
from crew_ai_ml.pipeline.prep_workspace import prep_tool_guard

logger = logging.getLogger(__name__)


class DropColumnsInput(BaseModel):
    columns: list[str] = Field(..., description="Column names to drop from working data.")
    reason: str = Field(..., description="Why these columns are being removed.")
    target_column: str = Field(..., description="Target column (never dropped).")


class DropColumnsTool(BaseTool):
    name: str = "drop_columns"
    description: str = "Drop specified columns (e.g. ID-like or zero-variance features)."
    args_schema: Type[BaseModel] = DropColumnsInput

    def _run(self, columns: list[str], reason: str, target_column: str) -> str:
        try:
            with prep_tool_guard():
                result = drop_columns(columns=columns, reason=reason, target_column=target_column)
            return json.dumps(result)
        except Exception:
            logger.exception("drop_columns failed")
            raise
