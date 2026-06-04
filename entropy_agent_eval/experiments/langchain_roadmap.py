from __future__ import annotations

from entropy_agent_eval.adapters import EventRecorder
from entropy_agent_eval.benchmarks import BenchmarkTask
from entropy_agent_eval.experiments.learning_roadmap import (
    ROADMAP_SYSTEM_PROMPT,
    build_roadmap_context,
    build_roadmap_prompt,
    grade_roadmap_response,
    roadmap_information_states,
)
from entropy_agent_eval.models import AgentRun


class LangChainLearningRoadmapAgent:
    name = "langchain-learning-roadmap"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install LangChain support with: pip install 'entropy-agent-eval[langchain]'"
            ) from exc
        self.model_name = model
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.human_message = HumanMessage
        self.system_message = SystemMessage

    def run(self, task: BenchmarkTask, repetition: int) -> AgentRun:
        recorder = EventRecorder(task_id=task.id)
        context = self._prepare_context(task, recorder)
        prompt = build_roadmap_prompt(task, context)
        recorder.llm(self.model_name, provider="langchain")
        response = self.llm.invoke(
            [
                self.system_message(content=ROADMAP_SYSTEM_PROMPT),
                self.human_message(content=prompt),
            ]
        )
        text = str(getattr(response, "content", response))
        expected_terms = list(task.metadata.get("expected_terms", []))
        success = grade_roadmap_response(text, expected_terms)
        before, after = roadmap_information_states(success, expected_terms, text)
        usage = getattr(response, "usage_metadata", None) or {}
        cost = _cost_from_usage(usage)
        return recorder.to_run(
            success=success,
            cost=cost,
            before_state=before,
            after_state=after,
            outcome="usable_roadmap" if success else "incomplete_roadmap",
            agent_name=self.name,
            provider="langchain",
            model=self.model_name,
            repetition=repetition,
            response=text,
            usage=usage,
        )

    def _prepare_context(self, task: BenchmarkTask, recorder: EventRecorder) -> dict[str, object]:
        recorder.tool("assess_learner_profile")
        recorder.tool("select_learning_modules")
        recorder.tool("build_weekly_schedule")
        recorder.tool("design_assessment_checkpoints")
        return build_roadmap_context(task)


def _cost_from_usage(usage: dict[str, object]) -> float:
    total_tokens = usage.get("total_tokens") or usage.get("total_token_count") or 0
    try:
        return round(float(total_tokens) * 0.000001, 6)
    except (TypeError, ValueError):
        return 0.0
