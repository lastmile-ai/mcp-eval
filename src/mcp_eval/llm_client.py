"""Simple LLM client for judge evaluations."""

from typing import Optional
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class JudgeLLMClient:
    """Simple LLM client for judge evaluations."""

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self._client = None

    async def generate_str(self, prompt: str) -> str:
        """Generate a string response."""
        if not self._client:
            if "claude" in self.model:
                self._client = AnthropicAugmentedLLM()
            elif "gpt" in self.model:
                self._client = OpenAIAugmentedLLM()
            else:
                self._client = AnthropicAugmentedLLM()  # Default

        # For judge evaluations, we create a simple mock agent
        # In practice, this would use the proper LLM client
        return await self._mock_llm_call(prompt)

    async def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM call for demo purposes."""
        # In real implementation, this would call the actual LLM
        # For now, return a mock score
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return "0.85"
        return "The response meets the specified criteria."


def get_judge_client(model: Optional[str] = None) -> JudgeLLMClient:
    """Get a judge LLM client."""
    return JudgeLLMClient(model or "claude-3-haiku-20240307")
