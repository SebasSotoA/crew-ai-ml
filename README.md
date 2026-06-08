# crew-ai-ml

Multi-agent ML pipeline powered by [CrewAI](https://crewai.com) for end-to-end binary classification: data preparation, train/test split, model training, evaluation, and Streamlit deployment.

## Setup

Requires Python >=3.10, <3.14 and [uv](https://docs.astral.sh/uv/).

```powershell
pip install uv
crewai install
Copy-Item .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## Run

```powershell
crewai run
```

By default the pipeline uses `data/titanic.csv`. Set `TARGET_COLUMN` in `.env` or enter it when prompted.

## Project layout

- `src/crew_ai_ml/config/` — agent and task definitions (YAML)
- `src/crew_ai_ml/crew.py` — crew orchestration
- `src/crew_ai_ml/pipeline/` — deterministic ML pipeline steps
- `src/crew_ai_ml/tools/` — CrewAI tools wrapping pipeline steps
- `output/` — generated artifacts (models, reports, plots; gitignored)

## Docs

- [CrewAI documentation](https://docs.crewai.com)
