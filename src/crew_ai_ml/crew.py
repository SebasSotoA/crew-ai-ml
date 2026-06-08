import os

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from crew_ai_ml.tools import (
    DataPreparationTool,
    DataSplitTool,
    DeploymentTool,
    ModelEvaluationTool,
    ModelTrainingTool,
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

    @agent
    def data_preparation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["data_preparation_agent"],  # type: ignore[index]
            tools=[DataPreparationTool()],
            llm=_get_llm(),
            verbose=True,
        )

    @agent
    def split_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["split_agent"],  # type: ignore[index]
            tools=[DataSplitTool()],
            llm=_get_llm(),
            verbose=True,
        )

    @agent
    def model_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["model_agent"],  # type: ignore[index]
            tools=[ModelTrainingTool()],
            llm=_get_llm(),
            verbose=True,
        )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluation_agent"],  # type: ignore[index]
            tools=[ModelEvaluationTool()],
            llm=_get_llm(),
            verbose=True,
        )

    @agent
    def deployment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["deployment_agent"],  # type: ignore[index]
            tools=[DeploymentTool()],
            llm=_get_llm(),
            verbose=True,
        )

    @task
    def data_preparation_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_preparation_task"],  # type: ignore[index]
        )

    @task
    def split_task(self) -> Task:
        return Task(
            config=self.tasks_config["split_task"],  # type: ignore[index]
            context=[self.data_preparation_task()],
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config["model_training_task"],  # type: ignore[index]
            context=[self.split_task()],
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluation_task"],  # type: ignore[index]
            context=[self.model_training_task()],
        )

    @task
    def deployment_task(self) -> Task:
        return Task(
            config=self.tasks_config["deployment_task"],  # type: ignore[index]
            context=[self.evaluation_task()],
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
