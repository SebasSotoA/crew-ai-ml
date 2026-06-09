import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import document_deploy_failure

logger = logging.getLogger(__name__)


class DocumentDeployFailureInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    rationale: str = Field(
        ...,
        description="Evidence-based rationale for blocking deployment (min 20 chars).",
    )
    remediation_steps_json: str | None = Field(
        default=None,
        description=(
            "JSON array of remediation step strings, or newline-separated steps. "
            "Omit for default remediation guidance."
        ),
    )


class DocumentDeployFailureTool(BaseTool):
    name: str = "document_deploy_failure"
    description: str = (
        "Write output/failure_report.md explaining why deployment was blocked. "
        "DO NOT DEPLOY branch only — call after profile_deploy_context."
    )
    args_schema: Type[BaseModel] = DocumentDeployFailureInput

    def _run(
        self,
        target_column: str,
        rationale: str,
        remediation_steps_json: str | None = None,
    ) -> str:
        try:
            remediation_steps = remediation_steps_json or ""
            result = document_deploy_failure(
                target_column=target_column,
                remediation_steps=remediation_steps,
                rationale=rationale,
            )
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("document_deploy_failure failed")
            raise
