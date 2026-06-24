# Entropy-Based Observability for AI Agent Behavior

`entropy-agent-eval` implements **EOA**, a lightweight toolkit for turning agent traces into entropy-based observability telemetry. It does not score or rank agents. It helps you inspect how agent behavior changes across actions, tools, trajectories, uncertainty states, and outcomes.

The package computes descriptive signals such as:

- action entropy for action-selection dispersion
- trajectory entropy for behavioral path diversity
- tool entropy for tool-use concentration
- information gain for optional before/after uncertainty states
- entropy curves for temporal behavior
- robustness summaries across repeated runs

Any agent library can integrate by converting its trace events into `AgentRun` records.

Reference: https://arxiv.org/pdf/2606.05872

## Who This Is For

Use EOA when you want trace-level behavioral visibility beyond final success rate:

- framework authors inspecting runtime behavior
- application teams monitoring prompt, tool, or model changes
- researchers studying ReAct, planner, tool-using, or multi-agent traces
- observability teams turning stored traces into diagnostic telemetry

## Install

Requires Python 3.12 or newer.

```bash
pip install entropy-agent-eval
pip install "entropy-agent-eval[langchain]"
pip install "entropy-agent-eval[google-adk]"
pip install "entropy-agent-eval[plots]"
```

For local development:

```bash
poetry install --with dev
```

## Quick Start

```python
from entropy_agent_eval import AgentRun, EntropyObserver

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

report = EntropyObserver().observe(runs)
print(report.as_dict())
```

## CLI

```bash
eea examples/runs.json
eea examples/runs.json --per-run
```

The CLI accepts JSON objects with a top-level `runs` list, raw JSON lists, or JSONL files. It emits observability telemetry as JSON.

## Integration Model

EOA needs one thing: normalized traces as `AgentRun` objects. Those traces can come from live callbacks, custom wrappers, databases, observability systems, JSON/JSONL files, or workload harnesses.

```text
LangChain / Google ADK / custom agent / stored trace
        ↓
AgentRun
        ↓
EntropyObserver
        ↓
entropy-based observability signals
```

## Data Contract

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

## Custom Agent Integration

```python
from entropy_agent_eval import EntropyObserver
from entropy_agent_eval.adapters import EventRecorder

recorder = EventRecorder(task_id="task-123")
recorder.tool("search")
recorder.tool("python")
recorder.action("answer")

run = recorder.to_run(success=True, cost=0.04)
print(EntropyObserver().observe([run]).as_dict())
```

## Reading Signals

High entropy is not automatically good. Low entropy is not automatically bad. Entropy values are descriptive telemetry. They point to behavioral patterns that deserve trace inspection.

- Low action entropy can indicate focus or rigidity.
- High action entropy can indicate exploration or instability.
- Low tool entropy can indicate specialization or overreliance.
- High tool entropy can indicate broad search or tool thrashing.
- Low outcome entropy across repeated runs can indicate stable outcomes.

Use these signals alongside success, cost, latency, outcome quality, and qualitative trace review.

## Documentation

- [Concept overview](docs/concepts/overview.md)
- [Observability signals](docs/concepts/signals.md)
- [Cost](docs/concepts/cost.md)
- [Custom agents](docs/integrations/custom-agents.md)
- [LangChain](docs/integrations/langchain.md)
- [Google ADK](docs/integrations/google-adk.md)
- [Stored traces](docs/integrations/observability.md)
