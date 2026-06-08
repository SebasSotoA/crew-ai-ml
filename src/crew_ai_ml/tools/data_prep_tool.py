import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import run_data_preparation

logger = logging.getLogger(__name__)


class DataPreparationToolInput(BaseModel):
    """Input schema for DataPreparationTool."""

    dataset_path: str = Field(..., description="Path to the raw dataset file.")
    target_column: str = Field(..., description="Name of the target column to predict.")


class DataPreparationTool(BaseTool):
    name: str = "data_preparation"
    description: str = (
        "Prepare and clean the dataset for modeling. "
        "Requires dataset_path and target_column."
    )
    args_schema: Type[BaseModel] = DataPreparationToolInput

    def _run(self, dataset_path: str, target_column: str) -> str:
        try:
            summary = run_data_preparation(
                dataset_path=dataset_path,
                target_column=target_column,
            )
            return json.dumps(summary)
        except Exception:
            logger.exception(
                "Data preparation failed for dataset=%s target=%s",
                dataset_path,
                target_column,
            )
            raise
