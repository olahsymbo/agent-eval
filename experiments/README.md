# Controlled Benchmark

This directory contains a reproducible benchmark for comparing agent behavior
under a shared task set and trace schema. The included reference agents run
locally, require no API keys, and exercise the same `AgentRun` format used by
framework integrations.

## Run

```bash
poetry run python scripts/run_experiment.py
```

Outputs are written to `experiments/results/`:

- `runs.json`: normalized `AgentRun` records
- `summary.json`: full per-agent EEA reports
- `summary.csv`: compact table 

## Design

The default experiment compares four reference agent patterns:

- `direct-llm`
- `react-search`
- `react-search-code`
- `planner-executor`

Each agent is run on each task for three repetitions.

The task file is [tasks.json](tasks.json). It includes factual QA, multi-hop
reasoning, and coding/debugging tasks.

## Using Your Own Agents

Replace the reference agents in `scripts/run_experiment.py` with your own
wrappers. Each wrapper only needs a `name` and a `run(task, repetition)` method
returning an `AgentRun` or compatible dictionary.

```python
class MyLangChainAgent:
    name = "my-langchain-agent"

    def run(self, task, repetition):
        handler = EntropyCallbackHandler(task_id=task.id)
        result = chain.invoke(
            {"input": task.prompt},
            config={"callbacks": [handler]},
        )
        return handler.to_run(
            success=grade(result, task.expected),
            cost=estimate_cost(result),
            outcome="correct" if grade(result, task.expected) else "incorrect",
        )
```

Use the same task set and repetition count across agents.
