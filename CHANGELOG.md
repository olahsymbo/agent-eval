# Changelog

All notable changes to this project will be documented here.

## 0.1.0 - Unreleased

- Add core entropy metrics for actions, tools, trajectories, uncertainty reduction, and temporal curves.
- Add `AgentRun`, `AgentEvent`, and `InformationState` as the framework-neutral data contract.
- Add `EntropyEvaluator`, `EvaluationReport`, and configurable `EntropicAgentScore`.
- Add generic, LangChain, and Google ADK-style adapters.
- Add JSON/JSONL loading and the `eea` CLI.
- Add a minimal benchmark harness with sample QA and coding tasks.
- Add optional matplotlib entropy curve plotting.
