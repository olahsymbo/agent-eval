from __future__ import annotations

from typing import Any, Iterable, Mapping

from entropy_agent_eval.models import AgentEvent, AgentRun


def runs_from_adk_events(
    task_id: str,
    events: Iterable[Mapping[str, Any]],
    *,
    success: bool | None = None,
    cost: float = 0.0,
    outcome: str | None = None,
) -> AgentRun:
    normalized = []
    for event in events:
        kind = str(event.get("kind") or event.get("event_type") or event.get("type") or "event")
        name = str(
            event.get("name")
            or event.get("tool_name")
            or event.get("agent_name")
            or event.get("model")
            or kind
        )
        timestamp = event.get("timestamp") or event.get("time")
        metadata = {
            key: value
            for key, value in event.items()
            if key not in {"kind", "event_type", "type", "name", "tool_name", "agent_name", "model", "timestamp", "time"}
        }
        normalized.append(AgentEvent(kind=kind, name=name, timestamp=timestamp, metadata=metadata))
    return AgentRun(task_id=task_id, events=normalized, success=success, cost=cost, outcome=outcome)
