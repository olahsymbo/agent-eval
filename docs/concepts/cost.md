# Cost

`cost` is supplied by your agent framework, application, trace exporter, or
benchmark harness. The evaluator does not calculate provider billing by itself.

Use `cost` as the numeric penalty for one agent run. It can represent:

- USD, such as `0.08` for eight cents
- total tokens
- token-normalized cost
- tool-call cost
- compute or runtime cost
- an internal unit used by your evaluation system

The most important rule is consistency. If you compare agents or versions, use
the same unit for every run in that comparison.

```python
run = AgentRun.from_mapping(
    {
        "task": "qa-001",
        "trajectory": ["search", "read", "answer"],
        "success": True,
        "cost": 0.08,
    }
)
```

`EntropyEvaluator` reports the average as `mean_cost`. `EntropicAgentScore`
subtracts cost as a penalty:

```text
score = success_reward + information_gain_reward + exploration_efficiency_reward - cost_penalty
```

If cost is unknown or irrelevant for your evaluation, omit it or set it to
`0.0`.

## Example: Provider Token Cost

If your framework exposes token usage and you know provider prices, compute cost
before creating the run:

```python
input_cost = prompt_tokens * input_price_per_token
output_cost = completion_tokens * output_price_per_token
cost = input_cost + output_cost
```

Then pass that value to `AgentRun`, `EventRecorder.to_run`, or a framework
adapter.
