from crew_ai_ml.tools.deploy.configure_app_ui_tool import ConfigureAppUiTool
from crew_ai_ml.tools.deploy.document_deploy_failure_tool import DocumentDeployFailureTool
from crew_ai_ml.tools.deploy.finalize_deployment_tool import FinalizeDeploymentTool
from crew_ai_ml.tools.deploy.generate_streamlit_app_tool import GenerateStreamlitAppTool
from crew_ai_ml.tools.deploy.profile_deploy_context_tool import ProfileDeployContextTool
from crew_ai_ml.tools.deploy.validate_deploy_inputs_tool import ValidateDeployInputsTool
from crew_ai_ml.tools.deploy.write_deploy_requirements_tool import WriteDeployRequirementsTool

DEPLOY_TOOLS = [
    ValidateDeployInputsTool(),
    ProfileDeployContextTool(),
    ConfigureAppUiTool(),
    GenerateStreamlitAppTool(),
    WriteDeployRequirementsTool(),
    DocumentDeployFailureTool(),
    FinalizeDeploymentTool(),
]

__all__ = [
    "DEPLOY_TOOLS",
    "ValidateDeployInputsTool",
    "ProfileDeployContextTool",
    "ConfigureAppUiTool",
    "GenerateStreamlitAppTool",
    "WriteDeployRequirementsTool",
    "DocumentDeployFailureTool",
    "FinalizeDeploymentTool",
]
