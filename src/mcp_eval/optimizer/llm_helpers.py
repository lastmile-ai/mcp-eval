"""LLM helper classes for various optimization tasks."""

from openai import OpenAI


class SummarizerLLM:
    """LLM-based message summarizer using OpenAI GPT-4o-mini."""
    
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o-mini"
    
    def summarize_message(self, message: str) -> str:
        """Summarize a long message using the LLM.
        
        Args:
            message: The message to summarize
            
        Returns:
            Summarized version of the message
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes messages concisely while preserving key information. Keep summaries under 200 characters."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this message: {message}"
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to truncation if LLM call fails
            return message[:200] + "... [truncated due to error]"