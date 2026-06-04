# LangChain

The LangChain adapter records chain, tool, and LLM callback events without
making LangChain a required dependency of the core package.

```python
from entropy_agent_eval import EntropyEvaluator
from entropy_agent_eval.adapters.langchain import EntropyCallbackHandler

handler = EntropyCallbackHandler(task_id="support-ticket-001")

result = chain.invoke(
    {"question": "Where is my order?"},
    config={"callbacks": [handler]},
)

run = handler.to_run(success=True, cost=0.12, outcome="resolved")
report = EntropyEvaluator().evaluate([run])
print(report.as_dict())
```

The callback records behavior, but `cost` is still supplied by your application.
For LangChain apps, that value often comes from token usage callbacks, provider
metadata, or your own pricing calculation. See [Cost](../concepts/cost.md).

For repeated evaluation, store each `run` in memory, a database, or JSONL and
evaluate batches:

```python
report = EntropyEvaluator().evaluate(runs)
```

The adapter preserves common callback fields such as `run_id`, `parent_run_id`,
and `tags` in event metadata.
