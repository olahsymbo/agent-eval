from __future__ import annotations

import argparse
import json
from typing import Sequence

from entropy_agent_eval.observer import EntropyObserver
from entropy_agent_eval.io import load_runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eea", description="Inspect agent traces with entropy-based observability signals.")
    parser.add_argument("path", help="JSON or JSONL run log path")
    parser.add_argument("--per-run", action="store_true", help="Emit per-run telemetry instead of corpus telemetry")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args(argv)

    runs = load_runs(args.path)
    observer = EntropyObserver()
    payload = (
        [dict(observer.observe_run(run)) for run in runs]
        if args.per_run
        else observer.observe(runs).as_dict()
    )
    print(json.dumps(payload, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
