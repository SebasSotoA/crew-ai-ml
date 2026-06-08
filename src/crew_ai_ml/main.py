#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crew_ai_ml.crew import CrewAiMl

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

DEFAULT_DATASET_PATH = "data/passengers_satisfaction.csv"


def _resolve_target_column() -> str:
    target = os.getenv("TARGET_COLUMN", "").strip()
    if target:
        return target
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    target = input("Enter the target column name for binary classification: ").strip()
    if not target:
        raise ValueError("Target column is required.")
    return target


def _validate_dataset_exists(dataset_path: str) -> None:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            "Place your CSV at data/dataset.csv or update dataset_path."
        )


def run() -> None:
    dataset_path = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)
    _validate_dataset_exists(dataset_path)
    target_column = _resolve_target_column()

    inputs = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "current_year": str(datetime.now().year),
    }

    print(f"Starting ML pipeline: dataset={dataset_path}, target={target_column}")

    try:
        result = CrewAiMl().crew().kickoff(inputs=inputs)
        print("\nPipeline completed successfully.")
        print(result.raw[:500] if hasattr(result, "raw") else result)
    except Exception as e:
        print(f"Pipeline halted: {e}")
        return


def train() -> None:
    inputs = {
        "dataset_path": os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH),
        "target_column": os.getenv("TARGET_COLUMN", "target"),
        "current_year": str(datetime.now().year),
    }
    CrewAiMl().crew().train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs=inputs,
    )


def replay() -> None:
    CrewAiMl().crew().replay(task_id=sys.argv[1])


def test() -> None:
    inputs = {
        "dataset_path": os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH),
        "target_column": os.getenv("TARGET_COLUMN", "target"),
        "current_year": str(datetime.now().year),
    }
    CrewAiMl().crew().test(
        n_iterations=int(sys.argv[1]),
        eval_llm=sys.argv[2],
        inputs=inputs,
    )


def run_with_trigger() -> None:
    import json

    if len(sys.argv) < 2:
        raise ValueError("No trigger payload provided. Pass JSON as the first argument.")

    trigger_payload = json.loads(sys.argv[1])
    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "dataset_path": trigger_payload.get("dataset_path", DEFAULT_DATASET_PATH),
        "target_column": trigger_payload.get("target_column", os.getenv("TARGET_COLUMN", "")),
        "current_year": str(datetime.now().year),
    }
    return CrewAiMl().crew().kickoff(inputs=inputs)
