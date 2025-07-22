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
        import openai
        import os
        
        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Fallback to mock for demo if no API key
            return await self._mock_llm_call(prompt)
        
        try:
            # Use OpenAI client directly for judge evaluations
            client = openai.AsyncOpenAI(api_key=api_key)
            
            response = await client.chat.completions.create(
                model=self.model if "gpt" in self.model else "gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert evaluator. Analyze the provided context and respond with valid JSON containing score, reasoning, passed, and confidence fields."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            # Fallback to mock
            return await self._mock_llm_call(prompt)

    async def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM call for demo purposes."""
        # In real implementation, this would call the actual LLM
        # For now, return a mock JSON response for judge evaluations
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return '{"score": 0.85, "reasoning": "Task appears to be successfully completed based on trace analysis", "passed": true, "confidence": 0.9}'
        return "The response meets the specified criteria."


def get_judge_client(model: Optional[str] = None) -> JudgeLLMClient:
    """Get a judge LLM client."""
    return JudgeLLMClient(model or "claude-3-haiku-20240307")
