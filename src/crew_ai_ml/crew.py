import os

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, before_kickoff, crew, task

from crew_ai_ml.pipeline.deploy_workspace import validate_deployment_task_output
from crew_ai_ml.pipeline.eval_workspace import validate_evaluation_task_output
from crew_ai_ml.pipeline.prep_workspace import set_kickoff_inputs, validate_prep_task_output
from crew_ai_ml.pipeline.split_workspace import validate_split_task_output
from crew_ai_ml.pipeline.train_workspace import validate_training_task_output
from crew_ai_ml.tools import (
    DEPLOY_TOOLS,
    EVAL_TOOLS,
    PREP_TOOLS,
    SPLIT_TOOLS,
    TRAIN_TOOLS,
)


def _get_llm() -> str:
    model = os.getenv("MODEL", "gpt-4o-mini")
    if "/" not in model:
        return f"openai/{model}"
    return model


@CrewBase
class CrewAiMl:
    """Sequential ML pipeline crew for binary classification."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    @before_kickoff
    def store_kickoff_inputs(self, inputs: dict) -> dict:
        set_kickoff_inputs(
            inputs.get("dataset_path"),
            inputs.get("target_column"),
        )
        return inputs

    @agent
    def data_preparation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["data_preparation_agent"],  # type: ignore[index]
            tools=PREP_TOOLS,
            llm=_get_llm(),
            reasoning=True,
            verbose=True,
        )

    @agent
    def split_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["split_agent"],  # type: ignore[index]
            tools=SPLIT_TOOLS,
            llm=_get_llm(),
            reasoning=True,
            verbose=True,
        )

    @agent
    def model_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["model_agent"],  # type: ignore[index]
            tools=TRAIN_TOOLS,
            llm=_get_llm(),
            reasoning=True,
            verbose=True,
        )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluation_agent"],  # type: ignore[index]
            tools=EVAL_TOOLS,
            llm=_get_llm(),
            reasoning=True,
            verbose=True,
        )

    @agent
    def deployment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["deployment_agent"],  # type: ignore[index]
            tools=DEPLOY_TOOLS,
            llm=_get_llm(),
            reasoning=True,
            verbose=True,
        )

    @task
    def data_preparation_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_preparation_task"],  # type: ignore[index]
            guardrail=validate_prep_task_output,
        )

    @task
    def split_task(self) -> Task:
        return Task(
            config=self.tasks_config["split_task"],  # type: ignore[index]
            context=[self.data_preparation_task()],
            guardrail=validate_split_task_output,
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config["model_training_task"],  # type: ignore[index]
            context=[self.split_task()],
            guardrail=validate_training_task_output,
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluation_task"],  # type: ignore[index]
            context=[self.model_training_task()],
            guardrail=validate_evaluation_task_output,
        )

    @task
    def deployment_task(self) -> Task:
        return Task(
            config=self.tasks_config["deployment_task"],  # type: ignore[index]
            context=[self.evaluation_task()],
            guardrail=validate_deployment_task_output,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.data_preparation_agent(),
                self.split_agent(),
                self.model_agent(),
                self.evaluation_agent(),
                self.deployment_agent(),
            ],
            tasks=[
                self.data_preparation_task(),
                self.split_task(),
                self.model_training_task(),
                self.evaluation_task(),
                self.deployment_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )
