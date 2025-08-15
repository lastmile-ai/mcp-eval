"""Simple LLM client for judge evaluations."""

from typing import Optional, TypeVar
from pydantic import BaseModel
from mcp_agent.workflows.factory import _llm_factory
from mcp_eval.config import get_settings

T = TypeVar("T", bound=BaseModel)


class JudgeLLMClient:
    """Simple LLM client for judge evaluations."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model
        self._client = None

    async def generate_str(self, prompt: str) -> str:
        """Generate a string response."""
        if not self._client:
            settings = get_settings()
            provider = settings.provider or ("anthropic" if "claude" in (self.model or "") else "openai")
            factory = _llm_factory(provider=provider, model=self.model, context=None)
            # Judge client is LLM-only; we build a client without an Agent
            self._client = factory  # use factory with callable-style interface

        return await self._client.generate_str(prompt)

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate a structured response using Pydantic model."""
        if not self._client:
            settings = get_settings()
            provider = settings.provider or ("anthropic" if "claude" in (self.model or "") else "openai")
            factory = _llm_factory(provider=provider, model=self.model, context=None)
            self._client = factory

        # Use the underlying LLM client's structured generation
        response = await self._client.generate_structured(
            prompt, response_model=response_model
        )
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
