from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from entropy_agent_eval.benchmarks import BenchmarkTask
from entropy_agent_eval.models import InformationState


ROADMAP_SYSTEM_PROMPT = """You are a Learning Roadmap Agent.
Create practical learning roadmaps with clear milestones, projects, resources,
time estimates, assessment checkpoints, and prerequisite notes. Be specific,
structured, and realistic."""


def load_learning_roadmap_tasks(path: str | Path) -> list[BenchmarkTask]:
    records = json.loads(Path(path).read_text())
    tasks = []
    for record in records:
        metadata = {
            "category": "learning_roadmap",
            "difficulty": record.get("difficulty", "medium"),
            "difficulty_score": record.get("difficulty_score", 0.5),
            "learner": record.get("learner", {}),
            "goal": record.get("goal", ""),
            "constraints": record.get("constraints", {}),
            "expected_terms": record.get("expected_terms", []),
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


def assess_learner_profile(learner: dict[str, Any], goal: str) -> dict[str, Any]:
    experience = str(learner.get("experience", "beginner")).lower()
    available_hours = learner.get("hours_per_week", 5)
    if "advanced" in experience or "senior" in experience:
        level = "advanced"
    elif "intermediate" in experience or "some" in experience:
        level = "intermediate"
    else:
        level = "beginner"
    return {
        "level": level,
        "hours_per_week": available_hours,
        "goal": goal,
        "risk": "scope creep" if available_hours < 5 else "pace management",
    }


def select_learning_modules(goal: str, level: str) -> list[str]:
    goal_text = goal.lower()
    modules = ["foundations", "guided practice", "capstone project"]
    if "agent" in goal_text or "llm" in goal_text:
        modules.extend(["prompting and evaluation", "tool use", "agent orchestration"])
    if "data" in goal_text or "ml" in goal_text:
        modules.extend(["data preparation", "model evaluation", "deployment basics"])
    if "web" in goal_text or "frontend" in goal_text:
        modules.extend(["interface fundamentals", "state management", "deployment"])
    if level == "beginner":
        modules.insert(1, "prerequisite refresh")
    return modules


def build_weekly_schedule(modules: list[str], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    weeks = int(constraints.get("weeks", 8))
    schedule = []
    for index, module in enumerate(modules[:weeks], start=1):
        schedule.append(
            {
                "week": index,
                "focus": module,
                "deliverable": f"Complete a short artifact demonstrating {module}.",
            }
        )
    if len(schedule) < weeks:
        schedule.append(
            {
                "week": weeks,
                "focus": "review and portfolio polish",
                "deliverable": "Publish the capstone and write a reflection.",
            }
        )
    return schedule


def design_assessment_checkpoints(modules: list[str]) -> list[str]:
    checkpoints = ["diagnostic self-assessment"]
    checkpoints.extend(f"checkpoint: {module}" for module in modules[:4])
    checkpoints.append("final capstone review")
    return checkpoints


def build_roadmap_context(task: BenchmarkTask) -> dict[str, Any]:
    learner = dict(task.metadata.get("learner", {}))
    goal = str(task.metadata.get("goal") or task.prompt)
    constraints = dict(task.metadata.get("constraints", {}))
    profile = assess_learner_profile(learner, goal)
    modules = select_learning_modules(goal, profile["level"])
    schedule = build_weekly_schedule(modules, constraints)
    checkpoints = design_assessment_checkpoints(modules)
    return {
        "learner_profile": profile,
        "modules": modules,
        "schedule": schedule,
        "checkpoints": checkpoints,
    }


def build_roadmap_prompt(task: BenchmarkTask, context: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Task: {task.prompt}",
            f"Goal: {task.metadata.get('goal')}",
            f"Learner: {json.dumps(task.metadata.get('learner', {}), sort_keys=True)}",
            f"Constraints: {json.dumps(task.metadata.get('constraints', {}), sort_keys=True)}",
            f"Planning context: {json.dumps(context, indent=2, sort_keys=True)}",
            "Return a roadmap with sections: Overview, Prerequisites, Weekly Plan, Projects, Resources, Assessment.",
        ]
    )


def grade_roadmap_response(response: str, expected_terms: list[str]) -> bool:
    text = response.lower()
    required_sections = ["overview", "weekly", "project", "resource", "assessment"]
    section_score = sum(1 for section in required_sections if section in text)
    term_score = sum(1 for term in expected_terms if term.lower() in text)
    return section_score >= 4 and term_score >= max(1, len(expected_terms) // 2)


def roadmap_information_states(success: bool, expected_terms: list[str], response: str) -> tuple[InformationState, InformationState]:
    before = InformationState({"complete": 0.34, "partial": 0.33, "weak": 0.33})
    matched = sum(1 for term in expected_terms if term.lower() in response.lower())
    coverage = matched / max(1, len(expected_terms))
    if success:
        complete = min(0.92, 0.7 + coverage * 0.2)
        after = InformationState({"complete": complete, "partial": 1 - complete - 0.05, "weak": 0.05})
    else:
        partial = min(0.55, 0.25 + coverage * 0.3)
        after = InformationState({"complete": 0.25, "partial": partial, "weak": 1 - 0.25 - partial})
    return before, after
