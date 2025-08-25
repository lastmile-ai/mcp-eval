"""Simple LLM client for judge evaluations."""

from typing import Optional, TypeVar
from pydantic import BaseModel
from mcp_agent.workflows.factory import create_llm
from mcp_eval.config import get_settings

T = TypeVar("T", bound=BaseModel)


class JudgeLLMClient:
    """Simple LLM client wrapper for judge evaluations.

    This wraps an AugmentedLLM instance configured specifically for judging.
    """

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.model = model
        self.provider = provider
        self._llm = None
        self._actual_model = None
        self._actual_provider = None

    async def _get_llm(self):
        """Lazy initialization of the AugmentedLLM instance."""
        if not self._llm:
            settings = get_settings()

            # Determine provider: explicit > judge config > global config > infer from model
            provider = self.provider
            if not provider:
                provider = settings.judge.provider
            if not provider:
                provider = settings.provider

            # Determine model: explicit > judge config > global config > ModelSelector
            model = self.model
            if not model:
                model = settings.judge.model
            if not model:
                model = settings.model

            # If still no model, let mcp-agent's ModelSelector pick one
            if not model:
                from mcp.types import ModelPreferences

                # For judging, prioritize intelligence and cost-effectiveness
                model = ModelPreferences(
                    costPriority=0.4, speedPriority=0.2, intelligencePriority=0.4
                )

            # Create an AugmentedLLM with minimal agent for judging
            self._llm = create_llm(
                agent_name="judge",
                instruction="You are an evaluation judge that provides objective assessments.",
                provider=provider,
                model=model,
            )

            # Store the actual configuration used
            self._actual_provider = provider
            self._actual_model = model if isinstance(model, str) else "model-selector"

        return self._llm

    def get_config(self) -> dict:
        """Get the actual configuration used by this judge."""
        return {
            "provider": self._actual_provider,
            "model": self._actual_model,
        }

    async def generate_str(self, prompt: str) -> str:
        """Generate a string response."""
        llm = await self._get_llm()
        return await llm.generate_str(prompt)

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate a structured response using Pydantic model."""
        llm = await self._get_llm()
        # Use the underlying LLM's structured generation
        response = await llm.generate_structured(prompt, response_model=response_model)
        return response

    async def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM call for demo purposes."""
        # In real implementation, this would call the actual LLM
        # For now, return a mock JSON response for judge evaluations
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return '{"score": 0.85, "reasoning": "Task appears to be successfully completed based on trace analysis", "passed": true, "confidence": 0.9}'
        return "The response meets the specified criteria."


def get_judge_client(
    model: Optional[str] = None, provider: Optional[str] = None
) -> JudgeLLMClient:
    """Get a judge LLM client.

    Uses the provided model/provider or falls back to config settings.
    If no model is configured, mcp-agent will use its model selection.
    """
    return JudgeLLMClient(model=model, provider=provider)
