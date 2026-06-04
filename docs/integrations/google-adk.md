# Google ADK

The Google ADK adapter accepts ADK-style event dictionaries and normalizes them
into one `AgentRun`.

```python
from entropy_agent_eval import EntropyEvaluator
from entropy_agent_eval.adapters.google_adk import runs_from_adk_events

events = [
    {"event_type": "tool", "tool_name": "Search"},
    {"event_type": "model", "model": "gemini"},
    {"event_type": "action", "name": "answer"},
]

run = runs_from_adk_events(
    "adk-task-001",
    events,
    success=True,
    cost=0.09,
    outcome="correct",
)

report = EntropyEvaluator().evaluate([run])
print(report.as_dict())
```

The adapter normalizes events, but `cost` is still supplied by your application
or trace layer. It may come from model token usage, tool costs, runtime cost, or
another consistent numeric unit. See [Cost](../concepts/cost.md).

If your ADK application emits a different event shape, pass dictionaries with
one of these fields where possible:

- event kind: `kind`, `event_type`, or `type`
- event name: `name`, `tool_name`, `agent_name`, or `model`
- timestamp: `timestamp` or `time`

Unknown fields are kept in event metadata.
