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

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model
        self._llm = None

    async def _get_llm(self):
        """Lazy initialization of the AugmentedLLM instance."""
        if not self._llm:
            settings = get_settings()
            provider = settings.provider or (
                "anthropic" if "claude" in (self.model or "") else "openai"
            )
            # Create an AugmentedLLM with minimal agent for judging
            self._llm = create_llm(
                agent_name="judge",
                instruction="You are an evaluation judge that provides objective assessments.",
                provider=provider,
                model=self.model,
            )
        return self._llm

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
        # For now, return a mock score
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return "0.85"
        return "The response meets the specified criteria."


def get_judge_client(model: Optional[str] = None) -> JudgeLLMClient:
    """Get a judge LLM client."""
    return JudgeLLMClient(model or "claude-3-5-haiku-20241022")
