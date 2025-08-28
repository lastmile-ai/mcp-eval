import pytest

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.llm_judge import LLMJudge
from mcp_eval.config import MCPEvalSettings, JudgeConfig, set_settings


@pytest.mark.asyncio
async def test_llm_judge_passes_judge_config_to_client(monkeypatch):
    # Configure judge-specific provider/model
    settings = MCPEvalSettings(
        provider="anthropic",
        model="claude-sonnet-4-0",
        judge=JudgeConfig(provider="openai", model="gpt-4o-mini"),
    )
    set_settings(settings)

    # Patch judge client to capture config
    captured = {}

    class _StubClient:
        async def _get_llm(self):
            return self

        def get_config(self):
            return captured

        async def generate_str(self, prompt: str):
            # Return valid JSON for judge
            return (
                '{"score": 0.9, "reasoning": "ok", "passed": true, "confidence": 0.9}'
            )

    def _get_client(model=None, provider=None):
        captured["provider"] = provider or settings.judge.provider or settings.provider
        captured["model"] = model or settings.judge.model or settings.model
        return _StubClient()

    monkeypatch.setattr("mcp_eval.llm_client.get_judge_client", _get_client)
    monkeypatch.setattr(
        "mcp_eval.evaluators.llm_judge.get_judge_client", _get_client, raising=False
    )

    ctx = EvaluatorContext(
        inputs="i",
        output="o",
        expected_output=None,
        metadata={},
        metrics=None,
        span_tree=None,
    )

    judge = LLMJudge(rubric="r", min_score=0.1)
    res = await judge.evaluate(ctx)
    assert res.passed is True
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
