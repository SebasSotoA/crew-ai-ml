"""ML pipeline modules for data preparation through deployment."""

from crew_ai_ml.pipeline.config import (
    CLEANED_DATA_PATH,
    DEFAULT_PARAM_GRID,
    DEPLOY_MAX_F1_GAP,
    DEPLOY_MIN_F1,
    EVALUATION_REPORT_PATH,
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_dirs,
)
from crew_ai_ml.pipeline.data_preparation import DataPreparationError, run_data_preparation
from crew_ai_ml.pipeline.feature_transform import (
    build_input_schema,
    transform_raw_dataframe,
    transform_raw_input,
)
from crew_ai_ml.pipeline.deploy import DeploymentError, run_deployment
from crew_ai_ml.pipeline.evaluate import EvaluationError, run_evaluation
from crew_ai_ml.pipeline.split import SplitError, run_split
from crew_ai_ml.pipeline.train import TrainingError, run_training

__all__ = [
    "CLEANED_DATA_PATH",
    "DEFAULT_PARAM_GRID",
    "DEPLOY_MAX_F1_GAP",
    "DEPLOY_MIN_F1",
    "DataPreparationError",
    "DeploymentError",
    "EVALUATION_REPORT_PATH",
    "EvaluationError",
    "FEATURE_METADATA_PATH",
    "MODEL_PATH",
    "build_input_schema",
    "transform_raw_dataframe",
    "transform_raw_input",
    "OUTPUT_DIR",
    "PLOTS_DIR",
    "PROJECT_ROOT",
    "SplitError",
    "TEST_DATA_PATH",
    "TRAIN_DATA_PATH",
    "TrainingError",
    "ensure_output_dirs",
    "run_data_preparation",
    "run_deployment",
    "run_evaluation",
    "run_split",
    "run_training",
]
