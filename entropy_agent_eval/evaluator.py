from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional

from entropy_agent_eval.metrics.core import (
    entropy_reduction,
    exploration_efficiency,
    normalized_entropy,
    shannon_entropy,
)
from entropy_agent_eval.metrics.robustness import robustness_summary
from entropy_agent_eval.metrics.temporal import entropy_curve, rolling_entropy_curve
from entropy_agent_eval.models import AgentRun


@dataclass(frozen=True)
class EntropicAgentScore:
    """Weighted composite score.

    Defaults reward success, information gain, and exploration efficiency while
    penalizing monetary or token-normalized cost.
    """

    success_weight: float = 1.0
    information_gain_weight: float = 1.0
    exploration_efficiency_weight: float = 1.0
    cost_weight: float = 1.0

    def compute(
        self,
        success_rate: float,
        information_gain: float,
        exploration_efficiency_value: float,
        cost: float,
    ) -> float:
        return (
            self.success_weight * success_rate
            + self.information_gain_weight * information_gain
            + self.exploration_efficiency_weight * exploration_efficiency_value
            - self.cost_weight * cost
        )


@dataclass
class EvaluationReport:
    runs: int
    action_entropy: float
    action_entropy_normalized: float
    tool_entropy: float
    tool_entropy_normalized: float
    trajectory_entropy: float
    trajectory_entropy_normalized: float
    success_rate: Optional[float]
    information_gain: float
    exploration_efficiency: Optional[float]
    mean_cost: float
    entropic_agent_score: Optional[float]
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
            "exploration_efficiency": self.exploration_efficiency,
            "mean_cost": self.mean_cost,
            "entropic_agent_score": self.entropic_agent_score,
            "robustness": self.robustness,
        }


class EntropyEvaluator:
    """Compute entropy metrics for one run or a corpus of runs."""

    def __init__(self, score: Optional[EntropicAgentScore] = None) -> None:
        self.score = score or EntropicAgentScore()

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

    def evaluate_run(self, run: AgentRun) -> Mapping[str, object]:
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
        }

    def evaluate(self, runs: Iterable[AgentRun]) -> EvaluationReport:
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
        efficiency = (
            exploration_efficiency(success_rate, action_h) if success_rate is not None else None
        )
        score = (
            self.score.compute(success_rate, information_gain, efficiency, mean_cost)
            if success_rate is not None and efficiency is not None
            else None
        )

        return EvaluationReport(
            runs=len(materialized),
            action_entropy=action_h,
            action_entropy_normalized=normalized_entropy(actions),
            tool_entropy=shannon_entropy(tools),
            tool_entropy_normalized=normalized_entropy(tools),
            trajectory_entropy=shannon_entropy(trajectories),
            trajectory_entropy_normalized=normalized_entropy(trajectories),
            success_rate=success_rate,
            information_gain=information_gain,
            exploration_efficiency=efficiency,
            mean_cost=mean_cost,
            entropic_agent_score=score,
            robustness=robustness_summary(materialized),
        )
