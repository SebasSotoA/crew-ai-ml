import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import finalize_preparation
from crew_ai_ml.pipeline.prep_workspace import prep_tool_guard

logger = logging.getLogger(__name__)


class FinalizePreparationInput(BaseModel):
    target_column: str = Field(
        ...,
        description="Target column name. Mandatory final step before split stage.",
    )


class FinalizePreparationTool(BaseTool):
    name: str = "finalize_preparation"
    description: str = (
        "Validate and write output/cleaned_data.csv and output/feature_metadata.json. "
        "MANDATORY last step of data preparation."
    )
    args_schema: Type[BaseModel] = FinalizePreparationInput

    def _run(self, target_column: str) -> str:
        try:
            with prep_tool_guard():
                result = finalize_preparation(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("finalize_preparation failed")
            raise
