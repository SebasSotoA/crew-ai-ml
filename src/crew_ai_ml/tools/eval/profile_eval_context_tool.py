import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.evaluate import profile_eval_context

logger = logging.getLogger(__name__)


class ProfileEvalContextInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class ProfileEvalContextTool(BaseTool):
    name: str = "profile_eval_context"
    description: str = (
        "Profile model and datasets; recommend splits, metrics, plots, and deployment "
        "threshold guidance. Call after validate_eval_inputs before computing metrics."
    )
    args_schema: Type[BaseModel] = ProfileEvalContextInput

    def _run(self, target_column: str) -> str:
        try:
            result = profile_eval_context(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("profile_eval_context failed")
            raise
