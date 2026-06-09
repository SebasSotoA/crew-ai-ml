import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.split import split_train_test

logger = logging.getLogger(__name__)


class SplitTrainTestInput(BaseModel):
    target_column: str = Field(..., description="Target column name.")
    test_size: float = Field(..., description="Fraction of rows held out for test (e.g. 0.2).")
    stratify: bool = Field(..., description="Whether to stratify split on target column.")
    random_state: int = Field(default=42, description="Random seed for reproducibility.")


class SplitTrainTestTool(BaseTool):
    name: str = "split_train_test"
    description: str = (
        "Split cleaned data into train and test holds using chosen parameters. "
        "Call after profile_split_data."
    )
    args_schema: Type[BaseModel] = SplitTrainTestInput

    def _run(
        self,
        target_column: str,
        test_size: float,
        stratify: bool,
        random_state: int = 42,
    ) -> str:
        try:
            result = split_train_test(
                target_column=target_column,
                test_size=test_size,
                stratify=stratify,
                random_state=random_state,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("split_train_test failed")
            raise
