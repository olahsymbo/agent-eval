from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from entropy_agent_eval.models import AgentEvent, AgentRun, InformationState


def normalize_events(events: Iterable[Mapping[str, Any]]) -> List[AgentEvent]:
    """Normalize dictionaries from arbitrary agent runtimes into AgentEvent objects."""

    return [AgentEvent.from_mapping(event) for event in events]


class EventRecorder:
    """Small framework-agnostic recorder for custom agent loops."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.events: list[AgentEvent] = []

    def record(self, kind: str, name: str, **metadata: Any) -> None:
        timestamp = metadata.pop("timestamp", None)
        self.events.append(AgentEvent(kind=kind, name=name, timestamp=timestamp, metadata=metadata))

    def action(self, name: str, **metadata: Any) -> None:
        self.record("action", name, **metadata)

    def tool(self, name: str, **metadata: Any) -> None:
        self.record("tool", name, **metadata)

    def llm(self, name: str = "llm", **metadata: Any) -> None:
        self.record("llm", name, **metadata)

    def to_run(
        self,
        *,
        success: Optional[bool] = None,
        cost: float = 0.0,
        before_state: Optional[InformationState] = None,
        after_state: Optional[InformationState] = None,
        outcome: Optional[str] = None,
        **metadata: Any,
    ) -> AgentRun:
        return AgentRun(
            task_id=self.task_id,
            events=list(self.events),
            success=success,
            cost=cost,
            before_state=before_state,
            after_state=after_state,
            outcome=outcome,
            metadata=metadata,
        )
