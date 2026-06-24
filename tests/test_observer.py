import pytest

from entropy_agent_eval import AgentRun, EntropyObserver
from entropy_agent_eval.adapters import EventRecorder


def test_evaluate_corpus():
    runs = [
        AgentRun.from_mapping(
            {
                "task": "qa",
                "trajectory": ["search", "read", "answer"],
                "success": True,
                "cost": 0.1,
                "before": {"A": 0.5, "B": 0.5},
                "after": {"A": 0.9, "B": 0.1},
            }
        ),
        AgentRun.from_mapping(
            {
                "task": "qa",
                "trajectory": ["search", "answer"],
                "success": False,
                "cost": 0.2,
                "before": {"A": 0.5, "B": 0.5},
                "after": {"A": 0.6, "B": 0.4},
            }
        ),
    ]

    report = EntropyObserver().observe(runs)

    assert report.runs == 2
    assert report.success_rate == pytest.approx(0.5)
    assert report.action_entropy > 0
    assert report.trajectory_entropy == pytest.approx(1.0)
    assert report.information_gain > 0
    assert not any("score" in key for key in report.as_dict())


def test_event_recorder_integration_boundary():
    recorder = EventRecorder("custom")
    recorder.tool("search")
    recorder.llm("gpt")
    recorder.action("answer")

    run = recorder.to_run(success=True, cost=0.03)
    telemetry = EntropyObserver().observe_run(run)

    assert run.task_id == "custom"
    assert telemetry["trajectory"] == ["search", "gpt", "answer"]
    assert telemetry["success"] is True


def test_agent_run_from_mapping_restores_metadata_field():
    run = AgentRun.from_mapping(
        {
            "task_id": "saved",
            "trajectory": ["think", "answer"],
            "metadata": {"agent_name": "saved-agent"},
        }
    )

    assert run.metadata["agent_name"] == "saved-agent"
