# Contributing

Thanks for helping improve `entropy-agent-eval`.

## Development Setup

```bash
poetry install --with dev --extras plots
```

Run tests:

```bash
poetry run pytest -q
```

Run linting:

```bash
poetry run ruff check .
```

## Integration Contributions

Adapters should stay optional. Do not add hard runtime dependencies on agent
frameworks such as LangChain, Google ADK, AutoGen, CrewAI, or OpenAI Agents SDK
unless the dependency is behind an optional extra.

Preferred adapter shape:

- accept framework events, callbacks, or trace objects
- normalize them into `AgentEvent` and `AgentRun`
- preserve framework-specific fields in `metadata`
- add tests using simple dictionaries or lightweight fakes

## Metric Contributions

Metrics should be deterministic, documented, and covered by tests. Prefer
standard-library implementations unless a dependency provides a clear benefit.

## Pull Request Checklist

- Tests pass with `python -m pytest -q`.
- Public APIs are documented in `README.md` or `docs/`.
- New adapters do not force extra dependencies for core users.
- New files use clear names and avoid generated artifacts.
