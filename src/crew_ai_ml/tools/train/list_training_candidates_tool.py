import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import list_training_candidates

logger = logging.getLogger(__name__)


class ListTrainingCandidatesInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ListTrainingCandidatesTool(BaseTool):
    name: str = "list_training_candidates"
    description: str = (
        "List stored training candidates and the current best selection. "
        "Call after training at least one candidate."
    )
    args_schema: Type[BaseModel] = ListTrainingCandidatesInput

    def _run(self, target_column: str) -> str:
        try:
            result = list_training_candidates(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("list_training_candidates failed")
            raise
