import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import profile_train_data

logger = logging.getLogger(__name__)


class ProfileTrainDataInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ProfileTrainDataTool(BaseTool):
    name: str = "profile_train_data"
    description: str = (
        "Profile training data and recommend algorithm and tuning strategy. "
        "Call after validate_train_data."
    )
    args_schema: Type[BaseModel] = ProfileTrainDataInput

    def _run(self, target_column: str) -> str:
        try:
            result = profile_train_data(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("profile_train_data failed")
            raise
