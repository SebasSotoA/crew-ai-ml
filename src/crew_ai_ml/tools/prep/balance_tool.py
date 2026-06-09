import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import balance_classes
from crew_ai_ml.pipeline.prep_workspace import prep_tool_guard

logger = logging.getLogger(__name__)


class BalanceClassesInput(BaseModel):
    method: str = Field(
        ...,
        description="Balancing method: smotenc, smote, undersample, or none.",
    )
    target_column: str = Field(..., description="Target column name.")
    imbalance_ratio: float | None = Field(
        default=None,
        description="Minority/majority ratio threshold to trigger balancing (default 0.4).",
    )


class BalanceClassesTool(BaseTool):
    name: str = "balance_classes"
    description: str = (
        "Balance binary target classes. Methods: smotenc, smote, undersample, none. "
        "Use smotenc when one-hot dummy columns exist."
    )
    args_schema: Type[BaseModel] = BalanceClassesInput

    def _run(
        self,
        method: str,
        target_column: str,
        imbalance_ratio: float | None = None,
    ) -> str:
        try:
            with prep_tool_guard():
                result = balance_classes(
                    method=method,
                    target_column=target_column,
                    imbalance_ratio=imbalance_ratio,
                )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("balance_classes failed")
            raise
