from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class InformationState:
    """A probability distribution over hypotheses at a point in an agent run."""

    probabilities: Mapping[str, float]

    @classmethod
    def from_sequence(cls, probabilities: Sequence[float]) -> "InformationState":
        return cls({str(index): value for index, value in enumerate(probabilities)})

    def values(self) -> List[float]:
        return list(self.probabilities.values())


@dataclass
class AgentEvent:
    """One normalized event emitted by an agent runtime."""

    kind: str
    name: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentEvent":
        kind = str(data.get("kind") or data.get("type") or "event")
        name = str(data.get("name") or data.get("action") or data.get("tool") or kind)
        timestamp = data.get("timestamp")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"kind", "type", "name", "action", "tool", "timestamp"}
        }
        return cls(kind=kind, name=name, timestamp=timestamp, metadata=metadata)


@dataclass
class AgentRun:
    """A single task execution by an agent.

    This is the integration boundary for EOA. Adapters for LangChain, Google ADK,
    OpenAI Agents SDK, custom ReAct loops, or stored JSON logs should produce this
    type and then pass it to :class:`EntropyObserver`.
    """

    task_id: str
    events: List[AgentEvent] = field(default_factory=list)
    success: Optional[bool] = None
    reward: Optional[float] = None
    cost: float = 0.0
    latency_ms: Optional[float] = None
    before_state: Optional[InformationState] = None
    after_state: Optional[InformationState] = None
    outcome: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentRun":
        events = data.get("events")
        if events is None and "trajectory" in data:
            events = [{"kind": "action", "name": name} for name in data["trajectory"]]
        normalized_events = [AgentEvent.from_mapping(event) for event in (events or [])]

        before = data.get("before_state") or data.get("before")
        after = data.get("after_state") or data.get("after")
        explicit_metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        metadata = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "task_id",
                "task",
                "id",
                "events",
                "trajectory",
                "success",
                "reward",
                "cost",
                "latency_ms",
                "before_state",
                "before",
                "after_state",
                "after",
                "outcome",
                "metadata",
            }
        }
        metadata.update(explicit_metadata)
        return cls(
            task_id=str(data.get("task_id") or data.get("task") or data.get("id") or "unknown"),
            events=normalized_events,
            success=data.get("success"),
            reward=data.get("reward"),
            cost=float(data.get("cost") or 0.0),
            latency_ms=data.get("latency_ms"),
            before_state=_coerce_information_state(before),
            after_state=_coerce_information_state(after),
            outcome=data.get("outcome"),
            metadata=metadata,
        )

    @property
    def actions(self) -> List[str]:
        return [event.name for event in self.events if event.kind in {"action", "tool", "llm"}]

    @property
    def tools(self) -> List[str]:
        return [event.name for event in self.events if event.kind == "tool"]

    @property
    def trajectory(self) -> str:
        return " -> ".join(event.name for event in self.events)

    def trajectory_tuple(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.events)


def runs_from_records(records: Iterable[Mapping[str, Any]]) -> List[AgentRun]:
    return [AgentRun.from_mapping(record) for record in records]


def _coerce_information_state(value: Any) -> Optional[InformationState]:
    if value is None:
        return None
    if isinstance(value, InformationState):
        return value
    if isinstance(value, Mapping):
        return InformationState(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return InformationState.from_sequence(value)
    raise TypeError("information state must be a mapping or sequence of probabilities")
