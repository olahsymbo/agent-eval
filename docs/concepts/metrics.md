# Metrics

EEA metrics describe the behavior of an agent run or a group of runs. They are
most useful when read together instead of as isolated numbers.

## Shannon Entropy

Most metrics use Shannon entropy:

```text
H(X) = -sum(p(x) * log2(p(x)))
```

The evaluator accepts either symbols, such as action names, or probability
vectors, such as an information state. Repeated symbols produce an empirical
distribution. More concentration means lower entropy; more spread means higher
entropy.

`normalized_entropy` divides the entropy by the maximum possible entropy for the
observed support size. This gives a value from `0.0` to `1.0`, which can be
easier to compare across different numbers of observed actions, tools, or
trajectories.

## Action Entropy

Action entropy measures how diverse the agent's meaningful decisions are. In
the current evaluator, actions include events with `kind` equal to `action`,
`tool`, or `llm`.

Low action entropy means the agent repeats a small set of actions. High action
entropy means the agent uses a wider variety of actions.

Use it to spot:

- repetitive loops
- overly broad action selection
- changes in behavior after prompt, model, or tool updates
- whether successful runs are focused or exploratory

## Tool Entropy

Tool entropy is computed only from events where `kind` is `tool`.

Low tool entropy means one or a few tools dominate. This can be good when a task
has a clear best tool, but it can also reveal overreliance.

High tool entropy means tool use is spread across more tools. This can be useful
for multi-step research or planning tasks, but it can also indicate tool
thrashing when success does not improve.

## Trajectory Entropy

Trajectory entropy treats each full run path as a symbol. For example:

```text
("search", "read", "answer")
("search", "calculate", "answer")
("think", "answer")
```

Across repeated runs, low trajectory entropy means the agent tends to follow the
same path. High trajectory entropy means it reaches for different paths.

This is especially useful for robustness analysis. An agent can have diverse
trajectories but stable outcomes, which suggests flexible problem solving. It
can also have diverse trajectories and unstable outcomes, which suggests
uncontrolled behavior.

## Information Gain

Information gain is the entropy reduction between the `before` and `after`
information states:

```text
information_gain = H(before) - H(after)
```

Positive information gain means the agent reduced uncertainty. For example, a
state that starts spread across several hypotheses and ends concentrated on the
correct hypothesis has positive gain.

Zero information gain means no before/after state was supplied, or the state
entropy did not change.

Negative information gain means the after state is more uncertain than the
before state. That can happen when the agent discovers ambiguity, but it can
also signal confusion.

## Entropy Curves

An entropy curve shows cumulative action entropy after each observed action.
Rolling entropy uses a fixed-size window to show local behavior changes.

Use entropy curves to inspect temporal behavior:

- a flat curve can indicate repeated behavior
- a rising curve can indicate exploration
- a late spike can reveal a change in strategy
- a rolling spike can reveal a local loop or tool burst

## Exploration Efficiency

Exploration efficiency is success per bit of action entropy:

```text
exploration_efficiency = success_rate / action_entropy
```

It rewards agents that succeed without unnecessary behavioral spread. Very low
action entropy uses a small epsilon to avoid division by zero, so compare this
metric alongside the raw entropy and success rate.

## Robustness Summary

The robustness summary aggregates repeated attempts:

- `trajectory_entropy`: diversity of full paths
- `outcome_entropy`: diversity of final outcomes
- `success_rate`: share of successful runs
- `cost_mean`: average supplied cost
- `cost_std`: cost variability

Low outcome entropy with high success is usually a good sign. It means the
agent's results are stable. High outcome entropy means repeated runs produce
different result types and need closer inspection.

## Entropic Agent Score

`EntropicAgentScore` combines success, information gain, exploration efficiency,
and cost:

```text
score = success_reward + information_gain_reward + exploration_efficiency_reward - cost_penalty
```

The weights are configurable:

```python
from entropy_agent_eval import EntropicAgentScore, EntropyEvaluator

evaluator = EntropyEvaluator(
    EntropicAgentScore(
        success_weight=2.0,
        information_gain_weight=1.0,
        exploration_efficiency_weight=0.5,
        cost_weight=1.5,
    )
)
```

Use the score as a comparison helper, not as a replacement for the individual
metrics. The individual metrics explain why the score moved.
