# Observability Signals

EOA signals describe the behavior of an agent run or a group of runs. They are most useful when read together and inspected against the underlying trace.

## Shannon Entropy

Most signals use Shannon entropy:

```text
H(X) = -sum(p(x) * log2(p(x)))
```

The observer accepts either symbols, such as action names, or probability vectors, such as an information state. Repeated symbols produce an empirical distribution. More concentration means lower entropy. More spread means higher entropy.

`normalized_entropy` divides the entropy by the maximum possible entropy for the observed support size. This gives a value from `0.0` to `1.0`, which can be easier to compare across different numbers of observed actions, tools, or trajectories.

## Action Entropy

Action entropy describes how diverse the agent's meaningful decisions are. In the current observer, actions include events with `kind` equal to `action`, `tool`, or `llm`.

Use it to spot repetitive loops, broad action selection, behavior changes after prompt/model/tool updates, and differences between focused and exploratory successful runs.

## Tool Entropy

Tool entropy is computed only from events where `kind` is `tool`. Low tool entropy means one or a few tools dominate. High tool entropy means tool use is spread across more tools. Neither value is inherently good or bad. The task and tool environment determine the interpretation.

## Trajectory Entropy

Trajectory entropy treats each full run path as a symbol. Across repeated runs, low trajectory entropy means the agent tends to follow the same path. High trajectory entropy means it reaches for different paths.

This is useful for robustness analysis. An agent can have diverse trajectories but stable outcomes, which suggests flexible problem solving. It can also have diverse trajectories and unstable outcomes, which suggests uncontrolled behavior.

## Information Gain

Information gain is the entropy reduction between the `before` and `after` information states:

```text
information_gain = H(before) - H(after)
```

Positive information gain means the supplied state became more concentrated. Negative information gain means the after state is more uncertain than the before state. This signal should be omitted or treated as simulated instrumentation when an agent system does not expose meaningful before/after hypothesis distributions.

## Entropy Curves

An entropy curve shows cumulative action entropy after each observed action. Rolling entropy uses a fixed-size window to show local behavior changes.

Use entropy curves to inspect temporal behavior such as repeated behavior, exploration, late strategy shifts, or local tool bursts.

## Robustness Summary

The robustness summary aggregates repeated attempts:

- `trajectory_entropy`: diversity of full paths
- `outcome_entropy`: diversity of final outcomes
- `success_rate`: share of successful runs
- `cost_mean`: average supplied cost
- `cost_std`: cost variability

Low outcome entropy with high success can indicate stable outcomes. High outcome entropy means repeated runs produce different result types and need closer inspection.

## Derived Views and Alerts

Derived dashboard views may combine entropy with success, cost, latency, or outcome labels. Treat those combinations as alerting or triage rules, not objective rankings. Entropy helps localize behavioral change. The trace, task, tool environment, and outcome determine what the change means.
