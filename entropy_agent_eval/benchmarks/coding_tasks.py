from entropy_agent_eval.benchmarks.base import BenchmarkTask


CODING_TASKS = [
    BenchmarkTask(
        id="code-sort",
        prompt="Write a Python function that returns a sorted copy of a list.",
        expected="Function should not mutate the input and should return ascending order.",
        metadata={"difficulty": "easy"},
    ),
    BenchmarkTask(
        id="code-dedupe-stable",
        prompt="Write a Python function that removes duplicates while preserving order.",
        expected="Function should preserve the first occurrence of each item.",
        metadata={"difficulty": "medium"},
    ),
]
