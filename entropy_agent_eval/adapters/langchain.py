from __future__ import annotations

from typing import Any, Optional

from entropy_agent_eval.adapters.generic import EventRecorder
from entropy_agent_eval.models import AgentRun


try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:
    BaseCallbackHandler = object  # type: ignore[misc,assignment]


class EntropyCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    def __init__(self, task_id: str) -> None:
        self.recorder = EventRecorder(task_id)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        self.recorder.tool(str(name), input=input_str, **_compact(kwargs))

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "llm"
        self.recorder.llm(str(name), prompt_count=len(prompts), **_compact(kwargs))

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "chain"
        self.recorder.action(str(name), input_keys=sorted(inputs.keys()), **_compact(kwargs))

    def to_run(
        self,
        *,
        success: Optional[bool] = None,
        cost: float = 0.0,
        outcome: Optional[str] = None,
        **metadata: Any,
    ) -> AgentRun:
        return self.recorder.to_run(success=success, cost=cost, outcome=outcome, **metadata)


def _compact(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in {"run_id", "parent_run_id", "tags"}}
