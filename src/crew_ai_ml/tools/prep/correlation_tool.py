import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.data_preparation import filter_by_correlation

logger = logging.getLogger(__name__)


class FilterByCorrelationInput(BaseModel):
    mode: str = Field(
        ...,
        description="Filter mode: redundancy, irrelevance, both, or skip.",
    )
    target_column: str = Field(..., description="Target column name.")
    redundancy_threshold: float | None = Field(
        default=None,
        description="Optional |r| threshold for redundant feature pairs (default 0.8).",
    )
    irrelevance_threshold: float | None = Field(
        default=None,
        description="Optional |r| threshold vs target for irrelevant features (default 0.05).",
    )


class FilterByCorrelationTool(BaseTool):
    name: str = "filter_by_correlation"
    description: str = (
        "Remove redundant or irrelevant numeric features by correlation. "
        "Modes: redundancy, irrelevance, both, skip."
    )
    args_schema: Type[BaseModel] = FilterByCorrelationInput

    def _run(
        self,
        mode: str,
        target_column: str,
        redundancy_threshold: float | None = None,
        irrelevance_threshold: float | None = None,
    ) -> str:
        try:
            result = filter_by_correlation(
                mode=mode,
                target_column=target_column,
                redundancy_threshold=redundancy_threshold,
                irrelevance_threshold=irrelevance_threshold,
            )
            return json.dumps(result)
        except Exception:
            logger.exception("filter_by_correlation failed")
            raise
