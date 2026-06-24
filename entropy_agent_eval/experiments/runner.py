from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from entropy_agent_eval.benchmarks import BenchmarkTask
from entropy_agent_eval.observer import EntropyObserver
from entropy_agent_eval.io import dump_runs
from entropy_agent_eval.models import AgentRun


class ExperimentAgent(Protocol):
    name: str

    def run(self, task: BenchmarkTask, repetition: int) -> AgentRun | Mapping[str, object]:
        """Run one benchmark task and return an AgentRun or compatible mapping."""


@dataclass(frozen=True)
class ExperimentConfig:
    repetitions: int = 3
    output_dir: Path = Path("experiments/results")
    seed: int = 7


@dataclass
class ExperimentResult:
    runs: list[AgentRun]
    summaries: dict[str, dict[str, object]]


def run_experiment(
    tasks: Iterable[BenchmarkTask],
    agents: Iterable[ExperimentAgent],
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run every agent on every task for N repetitions."""

    materialized_tasks = list(tasks)
    materialized_agents = list(agents)
    runs: list[AgentRun] = []

    for agent in materialized_agents:
        for task in materialized_tasks:
            for repetition in range(1, config.repetitions + 1):
                result = agent.run(task, repetition)
                run = result if isinstance(result, AgentRun) else AgentRun.from_mapping(result)
                run.metadata.setdefault("agent_name", agent.name)
                run.metadata.setdefault("task_category", task.metadata.get("category"))
                run.metadata.setdefault("difficulty", task.metadata.get("difficulty"))
                run.metadata.setdefault("repetition", repetition)
                run.metadata.setdefault("benchmark_prompt", task.prompt)
                if task.expected is not None:
                    run.metadata.setdefault("expected", task.expected)
                runs.append(run)

    summaries = summarize_by_agent(runs)
    write_experiment_outputs(config.output_dir, runs, summaries)
    return ExperimentResult(runs=runs, summaries=summaries)


def summarize_by_agent(runs: Iterable[AgentRun]) -> dict[str, dict[str, object]]:
    """Compute one observability report per agent."""

    grouped: dict[str, list[AgentRun]] = {}
    for run in runs:
        agent_name = str(run.metadata.get("agent_name", "unknown"))
        grouped.setdefault(agent_name, []).append(run)

    observer = EntropyObserver()
    summaries = {}
    for agent_name, agent_runs in sorted(grouped.items()):
        report = observer.observe(agent_runs).as_dict()
        report["mean_latency_ms"] = _mean(
            run.latency_ms for run in agent_runs if run.latency_ms is not None
        )
        report["mean_trajectory_length"] = _mean(len(run.events) for run in agent_runs)
        summaries[agent_name] = report
    return summaries


def write_experiment_outputs(
    output_dir: Path,
    runs: list[AgentRun],
    summaries: dict[str, dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_runs(runs, output_dir / "runs.json")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    write_summary_csv(output_dir / "summary.csv", summaries)


def write_summary_csv(path: Path, summaries: dict[str, dict[str, object]]) -> None:
    fields = [
        "agent",
        "runs",
        "success_rate",
        "mean_cost",
        "mean_latency_ms",
        "mean_trajectory_length",
        "action_entropy",
        "trajectory_entropy",
        "tool_entropy",
        "information_gain",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for agent_name, report in summaries.items():
            row = {"agent": agent_name}
            row.update({field: report.get(field) for field in fields if field != "agent"})
            writer.writerow(row)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
