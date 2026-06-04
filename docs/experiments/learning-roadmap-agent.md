# Learning Roadmap Agent Experiment

This experiment evaluates agents that create structured learning roadmaps. It
uses the same EEA trace format as the controlled benchmark, but calls real agent
frameworks when their optional dependencies and API keys are available.

## Install

LangChain:

```bash
pip install "entropy-agent-eval[langchain]"
export OPENAI_API_KEY="..."
```

Google ADK:

```bash
pip install "entropy-agent-eval[google-adk]"
export GOOGLE_API_KEY="..."
```

The runner also loads a local `.env` file automatically. For Google ADK, use
one of these names:

```bash
GOOGLE_API_KEY="..."
```

or:

```bash
GEMINI_API_KEY="..."
```

## Run

LangChain only:

```bash
poetry run python scripts/run_learning_roadmap_experiment.py --provider langchain
```

Google ADK only:

```bash
poetry run python scripts/run_learning_roadmap_experiment.py --provider google-adk
```

The default ADK model is `gemini-2.5-flash`.

Both:

```bash
poetry run python scripts/run_learning_roadmap_experiment.py --provider both
```

Outputs are written to `experiments/learning_roadmap_results/`:

- `runs.json`
- `summary.json`
- `summary.csv`

## What Is Measured

Each run records four planning-tool events:

- learner profile assessment
- module selection
- weekly schedule construction
- assessment checkpoint design

It then records the framework model call and grades the final roadmap with a
simple rubric based on required sections and expected domain terms. The grading function is intentionally transparent so you can replace it with human review or a stricter evaluator later.
