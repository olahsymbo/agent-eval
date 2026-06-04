from entropy_agent_eval.benchmarks.base import BenchmarkTask


QA_TASKS = [
    BenchmarkTask(
        id="qa-capital-france",
        prompt="What is the capital of France?",
        expected="Paris",
        metadata={"difficulty": "easy"},
    ),
    BenchmarkTask(
        id="qa-entropy-definition",
        prompt="In one sentence, define Shannon entropy.",
        metadata={"difficulty": "medium"},
    ),
]
