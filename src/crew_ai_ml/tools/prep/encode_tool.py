import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import encode_features

logger = logging.getLogger(__name__)


class EncodeFeaturesInput(BaseModel):
    strategy: str = Field(
        ...,
        description="Encoding strategy: one_hot_drop_first or one_hot_full.",
    )
    target_column: str = Field(..., description="Target column name.")


class EncodeFeaturesTool(BaseTool):
    name: str = "encode_features"
    description: str = (
        "One-hot encode categorical features. Required before correlation filter and balancing."
    )
    args_schema: Type[BaseModel] = EncodeFeaturesInput

    def _run(self, strategy: str, target_column: str) -> str:
        try:
            result = encode_features(strategy=strategy, target_column=target_column)
            return json.dumps(result)
        except Exception:
            logger.exception("encode_features failed")
            raise
