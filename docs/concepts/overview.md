# Concept Overview

Entropy-based Evaluation of AI Agents, or EEA, is a way to evaluate how an
agent behaves while it solves a task. Traditional evaluation often asks only
whether the answer was correct. EEA also asks how much uncertainty, branching,
tool use, and outcome stability appeared on the path to that answer.

![EEA trace-to-metrics flow](../assets/eea-trace-flow.svg)

The central idea is simple:

1. Capture an agent trace.
2. Normalize that trace into an `AgentRun`.
3. Compute entropy metrics over actions, tools, trajectories, state changes,
   outcomes, and cost.
4. Compare those metrics across agents, prompts, tools, models, or releases.

## Why Entropy

Entropy measures how spread out a distribution is. In EEA, that distribution
can be the set of actions an agent chooses, the tools it calls, the full
trajectories it follows across repeated runs, or the hypotheses it assigns
probability to before and after solving a task.

This makes entropy useful for agent evaluation because agent quality is not only
about correctness. Two agents can both answer correctly while behaving very
differently:

- one may repeatedly follow a short, focused path
- one may branch when the task demands it
- one may wander through many tools before converging
- one may be accurate but too expensive to run
- one may succeed once but behave inconsistently across repeats

EEA turns those behavioral differences into comparable numbers.

## The Evaluation Boundary

`AgentRun` is the integration boundary. Framework-specific details stay outside
the evaluator. LangChain callbacks, Google ADK events, custom ReAct loops,
database traces, or JSON logs all become the same shape:

```json
{
  "task_id": "qa-001",
  "events": [
    {"kind": "tool", "name": "search"},
    {"kind": "tool", "name": "read"},
    {"kind": "action", "name": "answer"}
  ],
  "success": true,
  "cost": 0.08,
  "before": {"correct": 0.45, "distractor": 0.55},
  "after": {"correct": 0.92, "distractor": 0.08}
}
```

The evaluator can then work with a single run or a corpus of runs without
knowing which agent framework produced them.

## Behavioral Signals

![Interpreting entropy in agent behavior](../assets/entropy-interpretation.svg)

Entropy values are signals, not grades by themselves. A high value is not
automatically good, and a low value is not automatically bad.

Low entropy can mean the agent is focused, predictable, and efficient. It can
also mean the agent has become brittle and does not adapt when the task changes.

Medium entropy can be healthy when the agent branches across useful strategies
while still producing stable outcomes.

High entropy can indicate broad exploration, but it can also reveal noisy tool
use, indecision, or unnecessary cost.

The strongest interpretation comes from reading entropy together with success
rate, information gain, cost, and robustness.

## What EEA Helps You Compare

Use EEA when you want to compare behavior across:

- two agent frameworks
- two prompt versions
- two model choices
- agents with different tools enabled
- repeated runs of the same task
- a controlled benchmark and a production trace sample

The result is a behavioral profile. It helps explain whether a change made the
agent more capable, merely more random, more stable, more expensive, or more
efficient.

## Common Reading Patterns

An agent with high success, positive information gain, moderate action entropy,
and low outcome entropy is often behaving well. It explores enough to solve the
task while converging on stable results.

An agent with high action entropy, high trajectory entropy, low information
gain, and high cost may be wandering. It is doing many things, but those things
are not reducing uncertainty or improving outcomes.

An agent with low action entropy and high success may be a strong specialist.
If it fails on shifted tasks, that same low entropy may point to brittle
determinism.

An agent with varied trajectories but low outcome entropy may be robust. It can
take different paths while still landing on the same kind of result.
