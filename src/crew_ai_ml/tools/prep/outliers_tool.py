import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import handle_outliers
from crew_ai_ml.pipeline.prep_workspace import prep_tool_guard

logger = logging.getLogger(__name__)


class HandleOutliersInput(BaseModel):
    strategy: str = Field(
        ...,
        description="Outlier strategy: iqr_remove, iqr_cap, or skip.",
    )
    target_column: str = Field(..., description="Target column name.")
    columns: list[str] | None = Field(
        default=None,
        description="Optional numeric columns to process; default all numeric features.",
    )


class HandleOutliersTool(BaseTool):
    name: str = "handle_outliers"
    description: str = (
        "Handle numeric outliers using IQR rules. Strategies: iqr_remove, iqr_cap, skip."
    )
    args_schema: Type[BaseModel] = HandleOutliersInput

    def _run(
        self,
        strategy: str,
        target_column: str,
        columns: list[str] | None = None,
    ) -> str:
        try:
            with prep_tool_guard():
                result = handle_outliers(
                    strategy=strategy,
                    target_column=target_column,
                    columns=columns,
                )
            return json.dumps(result)
        except Exception:
            logger.exception("handle_outliers failed")
            raise
