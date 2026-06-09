import json
import logging
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_ml.pipeline.deploy import configure_app_ui

logger = logging.getLogger(__name__)


class ConfigureAppUiInput(BaseModel):
    target_column: str = Field(..., description="Binary classification target column name.")
    ui_config_json: str | None = Field(
        default=None,
        description=(
            "JSON object with Streamlit UI settings (page_title, page_icon, "
            "confidence_threshold, tab labels, sidebar_metrics, etc.). "
            "Omit to use profile recommendations."
        ),
    )


class ConfigureAppUiTool(BaseTool):
    name: str = "configure_app_ui"
    description: str = (
        "Configure Streamlit UI settings via ui_config_json (e.g. page_title). "
        "DEPLOY branch only — do NOT call when verdict is DO NOT DEPLOY."
    )
    args_schema: Type[BaseModel] = ConfigureAppUiInput

    def _run(self, target_column: str, ui_config_json: str | None = None) -> str:
        try:
            ui_config: dict[str, Any] | None = None
            if ui_config_json:
                ui_config = json.loads(ui_config_json)
            result = configure_app_ui(target_column=target_column, ui_config=ui_config)
            return json.dumps(result, default=str)
        except Exception:
            logger.exception("configure_app_ui failed")
            raise
