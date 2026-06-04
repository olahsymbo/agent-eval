from __future__ import annotations

import argparse
import json
from typing import Sequence

from entropy_agent_eval.evaluator import EntropyEvaluator
from entropy_agent_eval.io import load_runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eea", description="Evaluate agent logs with entropy metrics.")
    parser.add_argument("path", help="JSON or JSONL run log path")
    parser.add_argument("--per-run", action="store_true", help="Emit per-run metrics instead of corpus metrics")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args(argv)

    runs = load_runs(args.path)
    evaluator = EntropyEvaluator()
    payload = (
        [dict(evaluator.evaluate_run(run)) for run in runs]
        if args.per_run
        else evaluator.evaluate(runs).as_dict()
    )
    print(json.dumps(payload, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
