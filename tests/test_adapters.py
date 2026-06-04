from entropy_agent_eval.adapters.google_adk import runs_from_adk_events


def test_google_adk_adapter_normalizes_common_event_fields():
    run = runs_from_adk_events(
        "adk-demo",
        [
            {"event_type": "tool", "tool_name": "Search", "query": "agent entropy"},
            {"event_type": "model", "model": "gemini", "tokens": 120},
            {"type": "action", "name": "answer"},
        ],
        success=True,
        cost=0.11,
        outcome="correct",
    )

    assert run.task_id == "adk-demo"
    assert run.success is True
    assert run.cost == 0.11
    assert run.outcome == "correct"
    assert [event.kind for event in run.events] == ["tool", "model", "action"]
    assert [event.name for event in run.events] == ["Search", "gemini", "answer"]
    assert run.events[0].metadata["query"] == "agent entropy"
