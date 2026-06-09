import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import generate_streamlit_app

logger = logging.getLogger(__name__)


class GenerateStreamlitAppInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")


class GenerateStreamlitAppTool(BaseTool):
    name: str = "generate_streamlit_app"
    description: str = (
        "Generate output/app.py Streamlit inference app with manual and CSV tabs. "
        "DEPLOY branch only — call after configure_app_ui."
    )
    args_schema: Type[BaseModel] = GenerateStreamlitAppInput

    def _run(self, target_column: str) -> str:
        try:
            result = generate_streamlit_app(target_column=target_column)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("generate_streamlit_app failed")
            raise
