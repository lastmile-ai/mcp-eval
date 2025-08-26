import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate environment for tests and set temp report/trace dirs.

    - Use a temp working directory
    - Configure mcp-eval settings to avoid external LLM calls
    - Point OTEL traces and reporting to temp locations
    """
    # Work in a temp directory by default
    monkeypatch.chdir(tmp_path)

    # Minimal mcpeval.yaml so loader finds something predictable
    (tmp_path / "mcpeval.yaml").write_text(
        """
name: "Test Suite"
description: "Local tests"
judge:
  model: "dummy-judge"
  min_score: 0.5
reporting:
  formats: ["json"]
  output_dir: "./test-reports"
mcp:
  servers: {}
agents:
  definitions: []
        """.strip()
    )

    # Secrets file to satisfy CLI validate checks
    (tmp_path / "mcpeval.secrets.yaml").write_text(
        """
anthropic:
  api_key: "test_key"
        """.strip()
    )

    # Ensure we don't try to auto-load unrelated external pytest plugins from env
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


@pytest.fixture(autouse=True)
def _configure_settings(monkeypatch, tmp_path):
    """Programmatically set MCPEval settings for tests to avoid LLM usage."""
    from mcp_eval.config import MCPEvalSettings, set_settings

    reports_dir = tmp_path / "test-reports"
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    settings = MCPEvalSettings(
        name="Test Suite",
        description="Local tests",
        # Disable default provider/model so sessions don't attach LLMs
        provider=None,
        model=None,
    )
    # Configure OTEL to write to temp file and stay file-only
    settings.otel.enabled = True
    settings.otel.exporters = ["file"]
    settings.otel.path = str(traces_dir / "trace.jsonl")
    # Quiet logger
    settings.logger.transports = ["none"]
    # Reporting outputs to tmp
    settings.reporting.output_dir = str(reports_dir)
    settings.reporting.formats = ["json"]

    set_settings(settings)


@pytest.fixture()
def stub_judge_client(monkeypatch):
    """Stub out the judge client to avoid real LLM calls."""

    class _StubClient:
        def __init__(self, model=None, provider=None):
            self._model = model or "stub-model"
            self._provider = provider or "stub-provider"

        async def _get_llm(self):
            return self

        def get_config(self):
            return {"provider": self._provider, "model": self._model}

        async def generate_str(self, prompt: str) -> str:
            # Return a valid JSON payload for LLMJudge
            return (
                '{"score": 0.9, "reasoning": "Good", "passed": true, "confidence": 0.88}'
            )

        async def generate_structured(self, prompt: str, response_model):
            # Return a reasonable default instance for MultiCriteriaJudge
            return response_model(score=0.85, explanation="Solid", confidence=0.9)

    def _get_client(model=None, provider=None):
        return _StubClient(model=model, provider=provider)

    # Patch both the factory and the imports used inside evaluators
    monkeypatch.setattr("mcp_eval.llm_client.get_judge_client", _get_client)
    monkeypatch.setattr("mcp_eval.evaluators.llm_judge.get_judge_client", _get_client, raising=False)
    monkeypatch.setattr("mcp_eval.evaluators.multi_criteria_judge.get_judge_client", _get_client, raising=False)
    return _get_client


