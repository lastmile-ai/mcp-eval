import pytest

from mcp_eval.config import MCPEvalSettings, JudgeConfig, set_settings
from mcp_eval.llm_client import JudgeLLMClient


class _FakeLLM:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

    async def generate_str(self, prompt: str) -> str:
        return "ok"

    async def generate_structured(self, prompt: str, response_model):
        return response_model()


@pytest.mark.asyncio
async def test_judge_client_uses_judge_provider_over_global(monkeypatch):
    settings = MCPEvalSettings(
        provider="anthropic",
        model="claude-sonnet-4-0",
        judge=JudgeConfig(provider="openai", model="gpt-4o-mini"),
    )
    set_settings(settings)

    def _fake_create_llm(agent_name: str, instruction: str, provider: str, model):
        # provider/model should reflect judge-level overrides
        return _FakeLLM(provider, model)

    monkeypatch.setattr("mcp_eval.llm_client.create_llm", _fake_create_llm)

    client = JudgeLLMClient()
    await client._get_llm()
    cfg = client.get_config()
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_judge_client_falls_back_to_global_provider(monkeypatch):
    settings = MCPEvalSettings(provider="openai", model="gpt-4o")
    set_settings(settings)

    def _fake_create_llm(agent_name: str, instruction: str, provider: str, model):
        return _FakeLLM(provider, model)

    monkeypatch.setattr("mcp_eval.llm_client.create_llm", _fake_create_llm)

    client = JudgeLLMClient()
    await client._get_llm()
    cfg = client.get_config()
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_judge_client_raises_clear_error_when_unconfigured(monkeypatch):
    # No provider/model configured anywhere
    settings = MCPEvalSettings(provider=None, model=None, judge=JudgeConfig())
    set_settings(settings)

    def _raise_create_llm(agent_name: str, instruction: str, provider: str, model):
        raise Exception("API key missing")

    monkeypatch.setattr("mcp_eval.llm_client.create_llm", _raise_create_llm)

    client = JudgeLLMClient()
    with pytest.raises(RuntimeError) as exc:
        await client._get_llm()
    assert "Failed to initialize LLM judge" in str(exc.value)
    assert "Ensure judge.provider" in str(exc.value)


