import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import profile_dataset

logger = logging.getLogger(__name__)


class ProfileDatasetInput(BaseModel):
    target_column: str = Field(..., description="Target column to profile class balance and dtypes.")


class ProfileDatasetTool(BaseTool):
    name: str = "profile_dataset"
    description: str = (
        "Profile the working dataset: missing %, dtypes, class counts, imbalance ratio, "
        "and recommendations. Call after load_dataset before transforming."
    )
    args_schema: Type[BaseModel] = ProfileDatasetInput

    def _run(self, target_column: str) -> str:
        try:
            result = profile_dataset(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("profile_dataset failed")
            raise
