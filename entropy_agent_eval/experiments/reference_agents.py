from __future__ import annotations

import random
from dataclasses import dataclass

from entropy_agent_eval.benchmarks import BenchmarkTask
from entropy_agent_eval.models import AgentRun, InformationState


@dataclass
class ReferenceAgent:
    name: str
    base_success: float
    base_cost: float
    latency_ms: float
    seed: int = 7

    def run(self, task: BenchmarkTask, repetition: int) -> AgentRun:
        rng = random.Random(f"{self.seed}:{self.name}:{task.id}:{repetition}")
        difficulty = float(task.metadata.get("difficulty_score", 0.5))
        category = str(task.metadata.get("category", "general"))
        success_probability = max(0.05, min(0.98, self.base_success - (difficulty * 0.22)))
        success = rng.random() < success_probability
        trajectory = self._trajectory(category, success, rng)
        cost = round(self.base_cost + 0.015 * len(trajectory) + difficulty * 0.04, 4)
        before, after = self._states(success, difficulty)
        return AgentRun.from_mapping(
            {
                "task_id": task.id,
                "events": [_event_for_step(step) for step in trajectory],
                "success": success,
                "cost": cost,
                "latency_ms": self.latency_ms + 100 * len(trajectory) + difficulty * 500,
                "before": before.probabilities,
                "after": after.probabilities,
                "outcome": "correct" if success else "incorrect",
                "agent_name": self.name,
            }
        )

    def _trajectory(self, category: str, success: bool, rng: random.Random) -> list[str]:
        raise NotImplementedError

    def _states(self, success: bool, difficulty: float) -> tuple[InformationState, InformationState]:
        before = InformationState({"correct": 0.35, "partial": 0.25, "wrong": 0.40})
        if success:
            confidence = min(0.92, 0.65 + (1 - difficulty) * 0.2)
            after = InformationState(
                {"correct": confidence, "partial": 1 - confidence - 0.05, "wrong": 0.05}
            )
        else:
            after = InformationState({"correct": 0.34, "partial": 0.31, "wrong": 0.35})
        return before, after


def _event_for_step(step: str) -> dict[str, str]:
    if step in {"search", "code", "database", "memory"}:
        return {"kind": "tool", "name": step}
    if step in {"think", "plan", "synthesize", "revise"}:
        return {"kind": "llm", "name": step}
    return {"kind": "action", "name": step}


class DirectAgent(ReferenceAgent):
    def __init__(self, seed: int = 7) -> None:
        super().__init__("direct-llm", base_success=0.72, base_cost=0.01, latency_ms=900, seed=seed)

    def _trajectory(self, category: str, success: bool, rng: random.Random) -> list[str]:
        if category == "coding" and not success:
            return ["think", "answer"]
        return ["think", "answer"]


class SearchOnlyAgent(ReferenceAgent):
    def __init__(self, seed: int = 7) -> None:
        super().__init__("react-search", base_success=0.82, base_cost=0.025, latency_ms=1500, seed=seed)

    def _trajectory(self, category: str, success: bool, rng: random.Random) -> list[str]:
        if category == "coding":
            return ["think", "search", "read", "answer"] if success else ["search", "search", "answer"]
        if rng.random() < 0.25:
            return ["search", "read", "search", "answer"]
        return ["search", "read", "answer"]


class SearchCodeAgent(ReferenceAgent):
    def __init__(self, seed: int = 7) -> None:
        super().__init__(
            "react-search-code", base_success=0.87, base_cost=0.04, latency_ms=1900, seed=seed
        )

    def _trajectory(self, category: str, success: bool, rng: random.Random) -> list[str]:
        if category == "coding":
            return (
                ["search", "code", "test", "answer"]
                if success
                else ["search", "code", "test", "debug", "answer"]
            )
        if category == "multi_hop":
            return ["search", "read", "search", "read", "answer"]
        return ["search", "read", "answer"]


class PlannerAgent(ReferenceAgent):
    def __init__(self, seed: int = 7) -> None:
        super().__init__("planner-executor", base_success=0.84, base_cost=0.05, latency_ms=2200, seed=seed)

    def _trajectory(self, category: str, success: bool, rng: random.Random) -> list[str]:
        if category == "coding":
            return ["plan", "code", "test", "revise", "answer"] if success else ["plan", "code", "answer"]
        if category == "multi_hop":
            return ["plan", "search", "read", "search", "synthesize", "answer"]
        return ["plan", "search", "synthesize", "answer"]
