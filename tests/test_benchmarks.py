from entropy_agent_eval.benchmarks import QA_TASKS, run_benchmark


def test_run_benchmark_normalizes_mapping_results():
    def agent(task):
        return {"task_id": task.id, "trajectory": ["think", "answer"], "success": True}

    runs = run_benchmark(QA_TASKS[:1], agent)

    assert len(runs) == 1
    assert runs[0].task_id == "qa-capital-france"
    assert runs[0].latency_ms is not None
    assert runs[0].metadata["benchmark_prompt"]
