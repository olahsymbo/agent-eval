from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional

from entropy_agent_eval.metrics.core import (
    entropy_reduction,
    normalized_entropy,
    shannon_entropy,
)
from entropy_agent_eval.metrics.robustness import robustness_summary
from entropy_agent_eval.metrics.temporal import entropy_curve, rolling_entropy_curve
from entropy_agent_eval.models import AgentRun


@dataclass
class ObservabilityReport:
    """Trace-derived observability telemetry for a collection of agent runs."""

    runs: int
    action_entropy: float
    action_entropy_normalized: float
    tool_entropy: float
    tool_entropy_normalized: float
    trajectory_entropy: float
    trajectory_entropy_normalized: float
    success_rate: Optional[float]
    information_gain: float
    mean_cost: float
    robustness: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "runs": self.runs,
            "action_entropy": self.action_entropy,
            "action_entropy_normalized": self.action_entropy_normalized,
            "tool_entropy": self.tool_entropy,
            "tool_entropy_normalized": self.tool_entropy_normalized,
            "trajectory_entropy": self.trajectory_entropy,
            "trajectory_entropy_normalized": self.trajectory_entropy_normalized,
            "success_rate": self.success_rate,
            "information_gain": self.information_gain,
            "mean_cost": self.mean_cost,
            "robustness": self.robustness,
        }


class EntropyObserver:
    """Compute entropy-based observability signals for agent traces."""

    @staticmethod
    def action_entropy(actions: Iterable[str]) -> float:
        return shannon_entropy(list(actions))

    @staticmethod
    def tool_entropy(tools: Iterable[str]) -> float:
        return shannon_entropy(list(tools))

    @staticmethod
    def trajectory_entropy(trajectories: Iterable[Iterable[str] | str]) -> float:
        normalized = [
            tuple(trajectory) if not isinstance(trajectory, str) else trajectory
            for trajectory in trajectories
        ]
        return shannon_entropy(normalized)

    @staticmethod
    def information_gain(before: Iterable[float], after: Iterable[float]) -> float:
        return entropy_reduction(list(before), list(after))

    @staticmethod
    def entropy_curve(symbols: Iterable[str]) -> list[float]:
        return entropy_curve(symbols)

    @staticmethod
    def rolling_entropy_curve(symbols: Iterable[str], window_size: int) -> list[float]:
        return rolling_entropy_curve(symbols, window_size)

    def observe_run(self, run: AgentRun) -> Mapping[str, object]:
        actions = run.actions
        tools = run.tools
        gain = (
            entropy_reduction(run.before_state.values(), run.after_state.values())
            if run.before_state and run.after_state
            else 0.0
        )
        return {
            "task_id": run.task_id,
            "action_entropy": shannon_entropy(actions),
            "action_entropy_normalized": normalized_entropy(actions),
            "tool_entropy": shannon_entropy(tools),
            "tool_entropy_normalized": normalized_entropy(tools),
            "trajectory": list(run.trajectory_tuple()),
            "trajectory_length": len(run.events),
            "entropy_curve": entropy_curve(actions),
            "information_gain": gain,
            "success": run.success,
            "cost": run.cost,
            "outcome": run.outcome,
        }

    def observe(self, runs: Iterable[AgentRun]) -> ObservabilityReport:
        materialized = list(runs)
        actions = [action for run in materialized for action in run.actions]
        tools = [tool for run in materialized for tool in run.tools]
        trajectories = [run.trajectory_tuple() for run in materialized]
        successes = [run.success for run in materialized if run.success is not None]
        success_rate = mean([1.0 if success else 0.0 for success in successes]) if successes else None
        gains = [
            entropy_reduction(run.before_state.values(), run.after_state.values())
            for run in materialized
            if run.before_state and run.after_state
        ]
        information_gain = mean(gains) if gains else 0.0
        mean_cost = mean([run.cost for run in materialized]) if materialized else 0.0
        action_h = shannon_entropy(actions)

        return ObservabilityReport(
            runs=len(materialized),
            action_entropy=action_h,
            action_entropy_normalized=normalized_entropy(actions),
            tool_entropy=shannon_entropy(tools),
            tool_entropy_normalized=normalized_entropy(tools),
            trajectory_entropy=shannon_entropy(trajectories),
            trajectory_entropy_normalized=normalized_entropy(trajectories),
            success_rate=success_rate,
            information_gain=information_gain,
            mean_cost=mean_cost,
            robustness=robustness_summary(materialized),
        )
