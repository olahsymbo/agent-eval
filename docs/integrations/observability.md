# Observability and Stored Traces

Many teams already store agent traces in observability systems, warehouses, or
JSONL files. You do not need to run evaluation inside the agent process.

Normalize each trace into this shape:

```json
{
  "task_id": "trace-001",
  "events": [
    {"kind": "tool", "name": "search"},
    {"kind": "tool", "name": "database"},
    {"kind": "action", "name": "answer"}
  ],
  "success": true,
  "cost": 0.07,
  "outcome": "correct"
}
```

Then evaluate from Python:

```python
from entropy_agent_eval import EntropyEvaluator
from entropy_agent_eval.io import load_runs

runs = load_runs("runs.jsonl")
report = EntropyEvaluator().evaluate(runs)
```

Or from the CLI:

```bash
eea runs.jsonl
```

`cost` should come from the same place as the trace metadata: provider token
usage, billing records, tool-call prices, runtime measurements, or your own
normalized cost unit. Keep the unit consistent across runs being compared. See
[Cost](../concepts/cost.md).
