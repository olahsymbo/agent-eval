from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from entropy_agent_eval.models import AgentRun


def load_runs(path: str | Path) -> List[AgentRun]:
    """Load AgentRun records from a JSON or JSONL file."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text())
        records = payload["runs"] if isinstance(payload, dict) and "runs" in payload else payload
    if not isinstance(records, list):
        raise ValueError("expected a list of run records or an object with a 'runs' list")
    return [AgentRun.from_mapping(record) for record in records]


def dump_runs(runs: Iterable[AgentRun], path: str | Path) -> None:
    """Persist normalized runs as JSON."""

    payload = []
    for run in runs:
        payload.append(
            {
                "task_id": run.task_id,
                "events": [
                    {
                        "kind": event.kind,
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "metadata": event.metadata,
                    }
                    for event in run.events
                ],
                "success": run.success,
                "reward": run.reward,
                "cost": run.cost,
                "latency_ms": run.latency_ms,
                "before_state": run.before_state.probabilities if run.before_state else None,
                "after_state": run.after_state.probabilities if run.after_state else None,
                "outcome": run.outcome,
                "metadata": run.metadata,
            }
        )
    Path(path).write_text(json.dumps({"runs": payload}, indent=2, sort_keys=True))
