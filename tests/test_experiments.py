from pathlib import Path

from entropy_agent_eval.benchmarks import BenchmarkTask
from entropy_agent_eval.experiments import DirectAgent, ExperimentConfig, SearchOnlyAgent
from entropy_agent_eval.experiments.runner import run_experiment, summarize_by_agent


def test_experiment_runner_writes_outputs(tmp_path: Path):
    tasks = [
        BenchmarkTask(
            id="demo",
            prompt="Demo task",
            expected="Demo answer",
            metadata={"category": "factual", "difficulty": "easy", "difficulty_score": 0.2},
        )
    ]
    result = run_experiment(
        tasks,
        [DirectAgent(seed=1), SearchOnlyAgent(seed=1)],
        ExperimentConfig(repetitions=2, output_dir=tmp_path),
    )

    assert len(result.runs) == 4
    assert set(result.summaries) == {"direct-llm", "react-search"}
    assert (tmp_path / "runs.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()


def test_summarize_by_agent_groups_runs():
    agent = DirectAgent(seed=2)
    task = BenchmarkTask(
        id="demo",
        prompt="Demo task",
        metadata={"category": "factual", "difficulty": "easy", "difficulty_score": 0.2},
    )
    run = agent.run(task, repetition=1)
    run.metadata["agent_name"] = agent.name

    summary = summarize_by_agent([run])

    assert summary["direct-llm"]["runs"] == 1
    assert "mean_trajectory_length" in summary["direct-llm"]
