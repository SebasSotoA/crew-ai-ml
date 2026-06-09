import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.train import log_training_decision

logger = logging.getLogger(__name__)


class LogTrainingDecisionInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    issue: str = Field(
        ...,
        description='Decision topic (e.g. "algorithm_selection", "candidate_selection").',
    )
    options_considered: str = Field(
        ...,
        description=(
            'JSON array string of options considered (e.g. '
            '["gradient_boosting", "random_forest"]).'
        ),
    )
    choice: str = Field(..., description="Selected option (e.g. algorithm name).")
    rationale: str = Field(
        ...,
        description=(
            "Evidence-based rationale (min 20 chars). Reference profile recommendations "
            "when deviating from recommendations.algorithm."
        ),
    )


class LogTrainingDecisionTool(BaseTool):
    name: str = "log_training_decision"
    description: str = (
        "Record a structured training decision with rationale in train_state.json. "
        "Call after profile_train_data and before train_baseline. "
        "Required for algorithm choice; log additional decisions as needed."
    )
    args_schema: Type[BaseModel] = LogTrainingDecisionInput

    def _run(
        self,
        target_column: str,
        issue: str,
        options_considered: str,
        choice: str,
        rationale: str,
    ) -> str:
        try:
            result = log_training_decision(
                target_column=target_column,
                issue=issue,
                options_considered=options_considered,
                choice=choice,
                rationale=rationale,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("log_training_decision failed")
            raise
