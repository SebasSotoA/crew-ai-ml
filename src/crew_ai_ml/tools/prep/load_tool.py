import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import load_dataset

logger = logging.getLogger(__name__)


class LoadDatasetInput(BaseModel):
    dataset_path: str = Field(
        ...,
        description=(
            "Path to the raw CSV or Excel dataset. Use the exact kickoff path "
            "(e.g. data/passengers_satisfaction.csv), not just the filename."
        ),
    )
    target_column: str = Field(..., description="Binary classification target column name.")


class LoadDatasetTool(BaseTool):
    name: str = "load_dataset"
    description: str = (
        "Load a dataset into the prep workspace. Always call this first. "
        "Resets prior prep state."
    )
    args_schema: Type[BaseModel] = LoadDatasetInput

    def _run(self, dataset_path: str, target_column: str) -> str:
        try:
            result = load_dataset(dataset_path=dataset_path, target_column=target_column)
            return json.dumps(result)
        except Exception:
            logger.exception("load_dataset failed")
            raise
