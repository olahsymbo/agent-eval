from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, Iterable

from entropy_agent_eval.metrics.core import normalized_entropy, shannon_entropy
from entropy_agent_eval.models import AgentRun


def robustness_summary(runs: Iterable[AgentRun]) -> Dict[str, Any]:
    """Summarize stability across repeated task attempts."""

    materialized = list(runs)
    if not materialized:
        return {
            "runs": 0,
            "trajectory_entropy": 0.0,
            "outcome_entropy": 0.0,
            "success_rate": None,
            "cost_mean": 0.0,
            "cost_std": 0.0,
        }

    trajectories = [run.trajectory_tuple() for run in materialized]
    outcomes = [
        run.outcome
        if run.outcome is not None
        else ("success" if run.success is True else "failure" if run.success is False else "unknown")
        for run in materialized
    ]
    successes = [run.success for run in materialized if run.success is not None]
    costs = [run.cost for run in materialized]
    return {
        "runs": len(materialized),
        "trajectory_entropy": shannon_entropy(trajectories),
        "trajectory_entropy_normalized": normalized_entropy(trajectories),
        "outcome_entropy": shannon_entropy(outcomes),
        "outcome_entropy_normalized": normalized_entropy(outcomes),
        "success_rate": mean([1.0 if success else 0.0 for success in successes]) if successes else None,
        "cost_mean": mean(costs),
        "cost_std": pstdev(costs) if len(costs) > 1 else 0.0,
    }
