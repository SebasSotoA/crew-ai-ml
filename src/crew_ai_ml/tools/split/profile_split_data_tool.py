import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import profile_split_data

logger = logging.getLogger(__name__)


class ProfileSplitDataInput(BaseModel):
    target_column: str = Field(..., description="Target column to profile for split recommendations.")


class ProfileSplitDataTool(BaseTool):
    name: str = "profile_split_data"
    description: str = (
        "Profile cleaned data and recommend test_size, stratify, and random_state. "
        "Call after validate_cleaned_data before splitting."
    )
    args_schema: Type[BaseModel] = ProfileSplitDataInput

    def _run(self, target_column: str) -> str:
        try:
            result = profile_split_data(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("profile_split_data failed")
            raise
