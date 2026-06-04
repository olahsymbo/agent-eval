from pathlib import Path

from entropy_agent_eval.experiments.learning_roadmap import (
    build_roadmap_context,
    build_roadmap_prompt,
    grade_roadmap_response,
    load_learning_roadmap_tasks,
)


def test_learning_roadmap_task_loader():
    tasks = load_learning_roadmap_tasks(Path("experiments/learning_roadmap_tasks.json"))

    assert len(tasks) == 3
    assert tasks[0].metadata["category"] == "learning_roadmap"
    assert "expected_terms" in tasks[0].metadata


def test_learning_roadmap_context_and_prompt():
    task = load_learning_roadmap_tasks(Path("experiments/learning_roadmap_tasks.json"))[0]
    context = build_roadmap_context(task)
    prompt = build_roadmap_prompt(task, context)

    assert context["modules"]
    assert context["schedule"]
    assert "Weekly Plan" in prompt


def test_learning_roadmap_grader():
    response = """
    Overview
    Weekly plan
    Project: build an agent with tool use
    Resources
    Assessment through evaluation and deployment capstone
    """

    assert grade_roadmap_response(
        response,
        ["tool", "evaluation", "deployment", "capstone"],
    )
