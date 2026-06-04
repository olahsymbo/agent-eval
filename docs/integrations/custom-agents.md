# Custom Agents

`entropy-agent-eval` only needs a normalized trace. For custom agents, the
smallest integration is `EventRecorder`.

```python
from entropy_agent_eval import EntropyEvaluator
from entropy_agent_eval.adapters import EventRecorder

recorder = EventRecorder(task_id="custom-001")

recorder.tool("search", query="entropy metrics")
recorder.llm("planning-model")
recorder.action("answer")

run = recorder.to_run(success=True, cost=0.03, outcome="correct")
report = EntropyEvaluator().evaluate([run])
print(report.as_dict())
```

You can also construct records directly:

```python
from entropy_agent_eval import AgentRun

run = AgentRun.from_mapping(
    {
        "task_id": "custom-002",
        "events": [
            {"kind": "tool", "name": "database"},
            {"kind": "action", "name": "summarize"},
        ],
        "success": True,
    }
)
```

Use `kind="tool"` for external tool calls, `kind="llm"` for model calls, and
`kind="action"` for meaningful agent decisions.
