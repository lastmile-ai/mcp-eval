from pathlib import Path
import pytest

from mcp_eval.config import load_config, get_current_config, set_settings, MCPEvalSettings, use_agent, ProgrammaticDefaults


def test_load_config_from_temp(tmp_path, monkeypatch):
    cfg = tmp_path / "mcpeval.yaml"
    cfg.write_text(
        """
name: T
reporting:
  output_dir: ./test-reports
judge:
  model: dummy
        """.strip()
    )

    settings = load_config(cfg)
    assert settings.judge.model == "dummy"
    flat = get_current_config()
    assert "reporting" in flat and flat["reporting"]["output_dir"] == "./test-reports"


def test_programmatic_set_settings_and_use_agent(monkeypatch):
    s = MCPEvalSettings(name="X")
    s.provider = None
    s.model = None
    set_settings(s)

    # Using use_agent with an AgentSpec name should not raise; we just store it
    use_agent("default")
    # Programmatic default agent object path: ensure setter stores without error
    assert ProgrammaticDefaults.get_default_agent() is None


