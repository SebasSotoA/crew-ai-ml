import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import finalize_training

logger = logging.getLogger(__name__)


class FinalizeTrainingInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    candidate_id: str | None = Field(
        default=None,
        description="Optional candidate id to promote; defaults to best validation F1.",
    )


class FinalizeTrainingTool(BaseTool):
    name: str = "finalize_training"
    description: str = (
        "Promote the best candidate to output/model.pkl and write training_log.md. "
        "MANDATORY last tool call."
    )
    args_schema: Type[BaseModel] = FinalizeTrainingInput

    def _run(self, target_column: str, candidate_id: str | None = None) -> str:
        try:
            result = finalize_training(
                target_column=target_column,
                candidate_id=candidate_id,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("finalize_training failed")
            raise
