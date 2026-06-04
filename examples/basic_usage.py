from entropy_agent_eval import AgentRun, EntropyEvaluator, InformationState


runs = [
    AgentRun.from_mapping(
        {
            "task": "Write sorting algorithm",
            "trajectory": ["search", "python", "test", "answer"],
            "success": True,
            "cost": 0.12,
            "before": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
            "after": {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02},
        }
    ),
    AgentRun.from_mapping(
        {
            "task": "Write sorting algorithm",
            "trajectory": ["python", "test", "debug", "test", "answer"],
            "success": True,
            "cost": 0.18,
            "before": InformationState.from_sequence([0.25, 0.25, 0.25, 0.25]).probabilities,
            "after": [0.8, 0.1, 0.06, 0.04],
        }
    ),
]

report = EntropyEvaluator().evaluate(runs)
print(report.as_dict())
