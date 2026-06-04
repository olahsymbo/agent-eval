from __future__ import annotations

import asyncio

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


class GoogleADKLearningRoadmapAgent:
    name = "google-adk-learning-roadmap"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Install Google ADK support with: pip install 'entropy-agent-eval[google-adk]'"
            ) from exc
        self.model_name = model
        self.agent_class = LlmAgent
        self.runner_class = Runner
        self.session_service_class = InMemorySessionService
        self.types = types

    def run(self, task: BenchmarkTask, repetition: int) -> AgentRun:
        return asyncio.run(self._run_async(task, repetition))

    async def _run_async(self, task: BenchmarkTask, repetition: int) -> AgentRun:
        recorder = EventRecorder(task_id=task.id)
        context = self._prepare_context(task, recorder)
        prompt = build_roadmap_prompt(task, context)
        recorder.llm(self.model_name, provider="google-adk")
        response_text = await self._call_adk(task, prompt, repetition)
        expected_terms = list(task.metadata.get("expected_terms", []))
        success = grade_roadmap_response(response_text, expected_terms)
        before, after = roadmap_information_states(success, expected_terms, response_text)
        return recorder.to_run(
            success=success,
            cost=0.0,
            before_state=before,
            after_state=after,
            outcome="usable_roadmap" if success else "incomplete_roadmap",
            agent_name=self.name,
            provider="google-adk",
            model=self.model_name,
            repetition=repetition,
            response=response_text,
        )

    def _prepare_context(self, task: BenchmarkTask, recorder: EventRecorder) -> dict[str, object]:
        recorder.tool("assess_learner_profile")
        recorder.tool("select_learning_modules")
        recorder.tool("build_weekly_schedule")
        recorder.tool("design_assessment_checkpoints")
        return build_roadmap_context(task)

    async def _call_adk(self, task: BenchmarkTask, prompt: str, repetition: int) -> str:
        app_name = "learning_roadmap_experiment"
        user_id = "eea_user"
        session_id = f"{task.id}_{repetition}"
        agent = self.agent_class(
            name="learning_roadmap_agent",
            model=self.model_name,
            instruction=ROADMAP_SYSTEM_PROMPT,
            description="Creates structured learning roadmaps.",
        )
        session_service = self.session_service_class()
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        runner = self.runner_class(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )
        content = self.types.Content(role="user", parts=[self.types.Part(text=prompt)])
        final_response = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text or ""
                break
        return final_response
