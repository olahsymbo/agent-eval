from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entropy_agent_eval.benchmarks import BenchmarkTask  # noqa: E402
from entropy_agent_eval.experiments import (  # noqa: E402
    DirectAgent,
    ExperimentConfig,
    PlannerAgent,
    SearchCodeAgent,
    SearchOnlyAgent,
)
from entropy_agent_eval.experiments.runner import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled observability workload.")
    parser.add_argument("--tasks", default=str(ROOT / "experiments" / "tasks.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    agents = [
        DirectAgent(seed=args.seed),
        SearchOnlyAgent(seed=args.seed),
        SearchCodeAgent(seed=args.seed),
        PlannerAgent(seed=args.seed),
    ]
    result = run_experiment(
        tasks,
        agents,
        ExperimentConfig(
            repetitions=args.repetitions,
            output_dir=Path(args.output_dir),
            seed=args.seed,
        ),
    )

    print(f"Wrote {len(result.runs)} runs to {Path(args.output_dir)}")
    for agent_name, summary in result.summaries.items():
        print(
            f"{agent_name}: success={summary['success_rate']:.3f}, "
            f"action_entropy={summary['action_entropy']:.3f}, "
            f"trajectory_entropy={summary['trajectory_entropy']:.3f}"
        )
    return 0


def load_tasks(path: Path) -> list[BenchmarkTask]:
    records = json.loads(path.read_text())
    tasks = []
    for record in records:
        metadata = {
            "category": record["category"],
            "difficulty": record["difficulty"],
            "difficulty_score": record["difficulty_score"],
        }
        tasks.append(
            BenchmarkTask(
                id=record["id"],
                prompt=record["prompt"],
                expected=record.get("expected"),
                metadata=metadata,
            )
        )
    return tasks


if __name__ == "__main__":
    raise SystemExit(main())
