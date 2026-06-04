from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Protocol

from entropy_agent_eval.models import AgentRun


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    prompt: str
    expected: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentCallable(Protocol):
    def __call__(self, task: BenchmarkTask) -> AgentRun | Mapping[str, Any]:
        """Execute a task and return an AgentRun or AgentRun-compatible mapping."""


def run_benchmark(tasks: Iterable[BenchmarkTask], agent: AgentCallable) -> List[AgentRun]:
    """Run tasks through any callable agent and normalize outputs."""

    runs: list[AgentRun] = []
    for task in tasks:
        started = time.perf_counter()
        result = agent(task)
        elapsed_ms = (time.perf_counter() - started) * 1000
        run = result if isinstance(result, AgentRun) else AgentRun.from_mapping(result)
        run.task_id = run.task_id if run.task_id != "unknown" else task.id
        run.latency_ms = run.latency_ms if run.latency_ms is not None else elapsed_ms
        run.metadata.setdefault("benchmark_prompt", task.prompt)
        if task.expected is not None:
            run.metadata.setdefault("expected", task.expected)
        runs.append(run)
    return runs
