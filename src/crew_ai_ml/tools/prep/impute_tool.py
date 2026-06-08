import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import impute_missing

logger = logging.getLogger(__name__)


class ImputeMissingInput(BaseModel):
    strategy: str = Field(
        ...,
        description="Imputation strategy: median, mode, knn, or drop_rows.",
    )
    target_column: str = Field(..., description="Target column name.")
    columns: list[str] | None = Field(
        default=None,
        description="Optional columns to impute; default all feature columns.",
    )


class ImputeMissingTool(BaseTool):
    name: str = "impute_missing"
    description: str = (
        "Handle missing values. Strategies: median, mode, knn, drop_rows."
    )
    args_schema: Type[BaseModel] = ImputeMissingInput

    def _run(
        self,
        strategy: str,
        target_column: str,
        columns: list[str] | None = None,
    ) -> str:
        try:
            result = impute_missing(
                strategy=strategy,
                target_column=target_column,
                columns=columns,
            )
            return json.dumps(result)
        except Exception:
            logger.exception("impute_missing failed")
            raise
