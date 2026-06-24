from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entropy_agent_eval.experiments import ExperimentConfig  # noqa: E402
from entropy_agent_eval.experiments.google_adk_roadmap import (  # noqa: E402
    GoogleADKLearningRoadmapAgent,
)
from entropy_agent_eval.experiments.langchain_roadmap import (  # noqa: E402
    LangChainLearningRoadmapAgent,
)
from entropy_agent_eval.experiments.learning_roadmap import load_learning_roadmap_tasks  # noqa: E402
from entropy_agent_eval.experiments.runner import run_experiment  # noqa: E402


def main() -> int:
    load_env_file(ROOT / ".env")
    normalize_google_api_key()

    parser = argparse.ArgumentParser(description="Run the Learning Roadmap Agent observability study.")
    parser.add_argument("--provider", choices=["langchain", "google-adk", "both"], default="both")
    parser.add_argument("--tasks", default=str(ROOT / "experiments" / "learning_roadmap_tasks.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "learning_roadmap_results"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--langchain-model", default="gpt-4o-mini")
    parser.add_argument("--adk-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    agents = []
    try:
        if args.provider in {"langchain", "both"}:
            agents.append(LangChainLearningRoadmapAgent(model=args.langchain_model))
        if args.provider in {"google-adk", "both"}:
            agents.append(GoogleADKLearningRoadmapAgent(model=args.adk_model))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not agents:
        raise SystemExit("No agents selected.")

    tasks = load_learning_roadmap_tasks(args.tasks)
    result = run_experiment(
        tasks,
        agents,
        ExperimentConfig(repetitions=args.repetitions, output_dir=Path(args.output_dir)),
    )
    print(f"Wrote {len(result.runs)} runs to {args.output_dir}")
    for agent_name, summary in result.summaries.items():
        print(
            f"{agent_name}: success={summary['success_rate']:.3f}, "
            f"action_entropy={summary['action_entropy']:.3f}, "
            f"tool_entropy={summary['tool_entropy']:.3f}"
        )
    return 0


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_google_api_key() -> None:
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


if __name__ == "__main__":
    raise SystemExit(main())
