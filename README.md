# Entropy-Based Evaluation of AI Agents

`entropy-agent-eval` implements **EEA**, a toolkit for measuring agent behavior with entropy metrics:

- action entropy for action-selection uncertainty
- trajectory entropy for strategy diversity
- tool entropy for tool-use specialization
- information gain for uncertainty reduction
- entropy curves for temporal behavior
- robustness summaries across repeated runs
- a configurable Entropic Agent Score 

Any agent library can integrate by converting its trace events into `AgentRun`
records.

## Who This Is For

Use EEA when you want to compare agent behavior beyond success rate:

- framework authors who want behavioral diagnostics
- application teams evaluating agent changes before deployment
- researchers comparing ReAct, planner, tool-using, or multi-agent systems
- observability teams turning traces into evaluation metrics

## Install

Requires Python 3.12 or newer.

From GitHub:

```bash
pip install git+https://github.com/olahsymbo/entropy-agent-eval.git
```

For local development:

```bash
poetry install --with dev
```

Optional plotting support:

```bash
pip install "entropy-agent-eval[plots]"
```

Build source and wheel distributions:

```bash
poetry build
```

Install a local wheel:

```bash
pip install dist/entropy_agent_eval-v0.1.1-py3-none-any.whl
```

## Release

Package builds are handled by Poetry. To cut a release:

```bash
poetry version patch
git tag v0.1.1
git push origin main --tags
```


## Quick Start

```python
from entropy_agent_eval import AgentRun, EntropyEvaluator

runs = [
    AgentRun.from_mapping(
        {
            "task": "Write sorting algorithm",
            "success": True,
            "cost": 0.12,
            "trajectory": ["search", "python", "test", "answer"],
            "before": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
            "after": {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02},
        }
    )
]

report = EntropyEvaluator().evaluate(runs)
print(report.as_dict())
```

## CLI

```bash
eea examples/runs.json
eea examples/runs.json --per-run
```

The CLI accepts JSON objects with a top-level `runs` list, raw JSON lists, or
JSONL files.

## Integration Model

You do not have to export JSON logs. JSON is only one supported path.

EEA needs one thing: normalized traces as `AgentRun` objects. Those traces can
come from live callbacks, custom wrappers, databases, observability systems,
JSON/JSONL files, or benchmark harnesses.

```text
LangChain / Google ADK / custom agent / stored trace
        ↓
AgentRun
        ↓
EntropyEvaluator
        ↓
entropy metrics + Entropic Agent Score
```

## Data Contract

The central integration type is `AgentRun`:

```json
{
  "task": "qa-001",
  "success": true,
  "cost": 0.08,
  "trajectory": ["search", "read", "answer"],
  "before": {"correct": 0.45, "distractor": 0.55},
  "after": {"correct": 0.92, "distractor": 0.08}
}
```

For richer logs, use explicit events:

```json
{
  "task_id": "coding-42",
  "events": [
    {"kind": "tool", "name": "search"},
    {"kind": "tool", "name": "python"},
    {"kind": "action", "name": "answer"}
  ],
  "success": true
}
```

### Cost

`cost` is user or framework supplied. It can mean USD, total tokens,
token-normalized cost, tool-call cost, compute cost, or any other numeric
penalty you want to apply consistently across compared runs.

The evaluator reports it as `mean_cost` and subtracts it inside
`EntropicAgentScore`. If cost is unknown or irrelevant, omit it or leave it as
`0.0`.

Full guide: [docs/concepts/cost.md](docs/concepts/cost.md)

## Custom Agent Integration

```python
from entropy_agent_eval import EntropyEvaluator
from entropy_agent_eval.adapters import EventRecorder

recorder = EventRecorder(task_id="task-123")
recorder.tool("search")
recorder.tool("python")
recorder.action("answer")

run = recorder.to_run(success=True, cost=0.04)
print(EntropyEvaluator().evaluate([run]).as_dict())
```

Full guide: [docs/integrations/custom-agents.md](docs/integrations/custom-agents.md)

## LangChain Integration

```python
from entropy_agent_eval.adapters.langchain import EntropyCallbackHandler

handler = EntropyCallbackHandler(task_id="lc-001")

# Pass `handler` in your LangChain config/callbacks.
# result = chain.invoke(inputs, config={"callbacks": [handler]})

run = handler.to_run(success=True, cost=0.10)
```

Full guide: [docs/integrations/langchain.md](docs/integrations/langchain.md)

## Google ADK-Style Event Integration

```python
from entropy_agent_eval.adapters.google_adk import runs_from_adk_events

run = runs_from_adk_events(
    "adk-001",
    [
        {"event_type": "tool", "tool_name": "Search"},
        {"event_type": "model", "model": "gemini"},
    ],
    success=True,
)
```

Full guide: [docs/integrations/google-adk.md](docs/integrations/google-adk.md)

## Stored Trace Integration

If your traces are already in a database, warehouse, or observability platform,
export or query them into `AgentRun`-compatible dictionaries and evaluate them
offline.

Full guide: [docs/integrations/observability.md](docs/integrations/observability.md)

## Metric Notes

High entropy is not automatically good. EEA treats entropy as a behavioral
signature:

- low action entropy can mean focus or brittle determinism
- medium entropy can indicate adaptive branching
- high entropy can indicate exploration or chaos
- successful agents should often reduce state entropy over time
- robust agents can have moderate trajectory entropy with low outcome entropy

`EntropicAgentScore` is configurable:

```python
from entropy_agent_eval import EntropicAgentScore, EntropyEvaluator

evaluator = EntropyEvaluator(
    EntropicAgentScore(
        success_weight=2.0,
        information_gain_weight=1.0,
        exploration_efficiency_weight=0.5,
        cost_weight=1.5,
    )
)
```

## Benchmark

Any callable that accepts a `BenchmarkTask` and returns an `AgentRun` or
compatible dictionary can be benchmarked:

```python
from entropy_agent_eval.benchmarks import QA_TASKS, run_benchmark

def agent(task):
    return {
        "task_id": task.id,
        "trajectory": ["think", "answer"],
        "success": True,
    }

runs = run_benchmark(QA_TASKS, agent)
```

## Controlled Benchmark

The [experiments](experiments/) directory contains a controlled benchmark that
compares reference agent patterns across factual QA, multi-hop, and coding
tasks.

```bash
poetry run python scripts/run_experiment.py
```

The script writes normalized runs and per-agent summaries to
`experiments/results/`.

## Learning Roadmap Agent Experiment

The project also includes a framework-backed experiment for a Learning Roadmap
Agent. It can run with LangChain, Google ADK, or both when the optional
dependencies and API keys are installed.

```bash
pip install "entropy-agent-eval[langchain]"
export OPENAI_API_KEY="..."
poetry run python scripts/run_learning_roadmap_experiment.py --provider langchain
```

```bash
pip install "entropy-agent-eval[google-adk]"
export GOOGLE_API_KEY="..."
poetry run python scripts/run_learning_roadmap_experiment.py --provider google-adk
```

The roadmap experiment runner also reads `.env` automatically. For Google ADK,
set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

Full guide: [docs/experiments/learning-roadmap-agent.md](docs/experiments/learning-roadmap-agent.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New adapters are welcome, especially for
frameworks that can expose tool calls, model calls, actions, costs, outcomes,
and uncertainty states.

## License

MIT. See [LICENSE](LICENSE).
