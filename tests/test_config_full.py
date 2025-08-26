"""Comprehensive tests for config.py to achieve >80% coverage."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mcp_eval.config import (
    MCPEvalSettings,
    AgentConfig,
    OTELSettings,
    LoggerSettings,
    ReportingSettings,
    JudgeSettings,
    ProgrammaticDefaults,
    load_config,
    get_settings,
    set_settings,
    get_current_config,
    _merge_configs,
    load_secrets,
    register_agent_definition,
    get_agent_definition,
    list_agent_definitions,
)


def test_otel_settings_defaults():
    """Test OTELSettings default values."""
    otel = OTELSettings()
    assert otel.enabled is True
    assert otel.exporters == ["file"]
    assert otel.path == "./traces/{test_name}_trace.jsonl"
    assert otel.service_name == "mcp-eval"
    assert otel.console_output is False


def test_otel_settings_custom():
    """Test OTELSettings with custom values."""
    otel = OTELSettings(
        enabled=False,
        exporters=["jaeger", "file"],
        path="/custom/path.jsonl",
        service_name="custom-service",
        console_output=True,
        jaeger_endpoint="http://localhost:6831",
    )
    assert otel.enabled is False
    assert "jaeger" in otel.exporters
    assert otel.jaeger_endpoint == "http://localhost:6831"


def test_logger_settings_defaults():
    """Test LoggerSettings default values."""
    logger = LoggerSettings()
    assert logger.level == "INFO"
    assert logger.transports == ["file", "console"]
    assert logger.file_path == "./logs/{test_name}.log"


def test_logger_settings_custom():
    """Test LoggerSettings with custom values."""
    logger = LoggerSettings(
        level="DEBUG", transports=["console"], file_path="/custom/log.txt"
    )
    assert logger.level == "DEBUG"
    assert logger.transports == ["console"]
    assert logger.file_path == "/custom/log.txt"


def test_reporting_settings_defaults():
    """Test ReportingSettings default values."""
    reporting = ReportingSettings()
    assert reporting.formats == ["json", "html"]
    assert reporting.output_dir == "./test-reports"


def test_reporting_settings_custom():
    """Test ReportingSettings with custom values."""
    reporting = ReportingSettings(formats=["json"], output_dir="/custom/reports")
    assert reporting.formats == ["json"]
    assert reporting.output_dir == "/custom/reports"


def test_judge_settings_defaults():
    """Test JudgeSettings default values."""
    judge = JudgeSettings()
    assert judge.model == "claude-3-5-sonnet-20241022"
    assert judge.min_score == 0.5


def test_judge_settings_custom():
    """Test JudgeSettings with custom values."""
    judge = JudgeSettings(model="gpt-4", min_score=0.8)
    assert judge.model == "gpt-4"
    assert judge.min_score == 0.8


def test_agent_config():
    """Test AgentConfig dataclass."""
    agent = AgentConfig(
        name="test_agent",
        description="Test agent",
        tools=["tool1", "tool2"],
        system_prompt="Test prompt",
    )
    assert agent.name == "test_agent"
    assert agent.description == "Test agent"
    assert agent.tools == ["tool1", "tool2"]
    assert agent.system_prompt == "Test prompt"


def test_mcp_eval_settings_defaults():
    """Test MCPEvalSettings default values."""
    settings = MCPEvalSettings(name="Test", description="Test suite")
    assert settings.name == "Test"
    assert settings.description == "Test suite"
    assert settings.provider is None
    assert settings.model is None
    assert settings.judge is not None
    assert settings.otel is not None
    assert settings.logger is not None
    assert settings.reporting is not None


def test_mcp_eval_settings_custom():
    """Test MCPEvalSettings with custom values."""
    settings = MCPEvalSettings(
        name="Custom",
        description="Custom test",
        provider="anthropic",
        model="claude-3-sonnet",
        timeout=60,
        judge=JudgeSettings(model="gpt-4"),
        otel=OTELSettings(enabled=False),
    )
    assert settings.name == "Custom"
    assert settings.provider == "anthropic"
    assert settings.model == "claude-3-sonnet"
    assert settings.timeout == 60
    assert settings.judge.model == "gpt-4"
    assert settings.otel.enabled is False


def test_programmatic_defaults():
    """Test ProgrammaticDefaults singleton."""
    defaults1 = ProgrammaticDefaults()
    defaults2 = ProgrammaticDefaults()
    assert defaults1 is defaults2  # Same instance


def test_programmatic_defaults_set_get():
    """Test setting and getting programmatic defaults."""
    defaults = ProgrammaticDefaults()
    defaults.set("test_key", "test_value")
    assert defaults.get("test_key") == "test_value"
    assert defaults.get("nonexistent") is None


def test_programmatic_defaults_clear():
    """Test clearing programmatic defaults."""
    defaults = ProgrammaticDefaults()
    defaults.set("key1", "value1")
    defaults.set("key2", "value2")
    defaults.clear()
    assert defaults.get("key1") is None
    assert defaults.get("key2") is None


def test_get_set_settings():
    """Test getting and setting global settings."""
    original = get_settings()

    new_settings = MCPEvalSettings(name="New", description="New settings")
    set_settings(new_settings)

    retrieved = get_settings()
    assert retrieved.name == "New"
    assert retrieved.description == "New settings"

    # Restore original
    set_settings(original)


def test_merge_configs():
    """Test _merge_configs function."""
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}

    result = _merge_configs(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 2
    assert result["b"]["d"] == 3
    assert result["e"] == 4


def test_merge_configs_deep_nesting():
    """Test _merge_configs with deep nesting."""
    base = {"level1": {"level2": {"level3": {"value": 1}}}}
    override = {"level1": {"level2": {"level3": {"value": 2, "new": 3}}}}

    result = _merge_configs(base, override)
    assert result["level1"]["level2"]["level3"]["value"] == 2
    assert result["level1"]["level2"]["level3"]["new"] == 3


def test_load_secrets_file():
    """Test loading secrets from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        secrets = {
            "anthropic": {"api_key": "test_key"},
            "openai": {"api_key": "openai_key"},
        }
        yaml.dump(secrets, f)
        secrets_file = f.name

    try:
        result = load_secrets(secrets_file)
        assert result["anthropic"]["api_key"] == "test_key"
        assert result["openai"]["api_key"] == "openai_key"
    finally:
        Path(secrets_file).unlink()


def test_load_secrets_nonexistent():
    """Test loading secrets from nonexistent file."""
    result = load_secrets("/nonexistent/file.yaml")
    assert result == {}


def test_load_secrets_from_env():
    """Test loading secrets from environment variables."""
    with patch.dict(
        os.environ, {"ANTHROPIC_API_KEY": "env_key", "OPENAI_API_KEY": "env_openai_key"}
    ):
        result = load_secrets(None)
        assert result["anthropic"]["api_key"] == "env_key"
        assert result["openai"]["api_key"] == "env_openai_key"


def test_load_config_basic():
    """Test basic config loading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = """
name: Test Config
description: Test description
judge:
  model: gpt-4
  min_score: 0.7
        """
        f.write(config)
        config_file = f.name

    try:
        settings = load_config(config_file)
        assert settings.name == "Test Config"
        assert settings.description == "Test description"
        assert settings.judge.model == "gpt-4"
        assert settings.judge.min_score == 0.7
    finally:
        Path(config_file).unlink()


def test_load_config_with_servers():
    """Test config loading with MCP servers."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = """
name: Server Config
description: Test with servers
mcp:
  servers:
    test_server:
      command: python
      args: [server.py]
        """
        f.write(config)
        config_file = f.name

    try:
        settings = load_config(config_file)
        assert settings.name == "Server Config"
        assert "test_server" in settings.mcp["servers"]
        assert settings.mcp["servers"]["test_server"]["command"] == "python"
    finally:
        Path(config_file).unlink()


def test_load_config_with_agents():
    """Test config loading with agent definitions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = """
name: Agent Config
description: Test with agents
agents:
  definitions:
    - name: test_agent
      description: Test agent
      tools: [tool1, tool2]
        """
        f.write(config)
        config_file = f.name

    try:
        settings = load_config(config_file)
        assert settings.name == "Agent Config"
        assert len(settings.agents["definitions"]) == 1
        assert settings.agents["definitions"][0]["name"] == "test_agent"
    finally:
        Path(config_file).unlink()


def test_load_config_with_secrets():
    """Test config loading with secrets file."""
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    config_file.write("name: Test\ndescription: Test")
    config_file.close()

    secrets_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    secrets_file.write("anthropic:\n  api_key: secret_key")
    secrets_file.close()

    try:
        settings = load_config(config_file.name, secrets_path=secrets_file.name)
        # Secrets are loaded but not directly accessible in settings
        assert settings.name == "Test"
    finally:
        Path(config_file.name).unlink()
        Path(secrets_file.name).unlink()


def test_get_current_config():
    """Test get_current_config function."""
    settings = MCPEvalSettings(name="Current", description="Current config")
    set_settings(settings)

    config = get_current_config()
    assert config["name"] == "Current"
    assert config["description"] == "Current config"
    assert "judge" in config
    assert "otel" in config


def test_register_and_get_agent_definition():
    """Test registering and retrieving agent definitions."""
    from mcp_agent.agents.agent_spec import AgentSpec

    agent_spec = AgentSpec(
        name="registered_agent", description="Registered test agent", tools=[]
    )

    register_agent_definition("test_agent", agent_spec)

    retrieved = get_agent_definition("test_agent")
    assert retrieved == agent_spec
    assert retrieved.name == "registered_agent"


def test_get_agent_definition_not_found():
    """Test getting non-existent agent definition."""
    result = get_agent_definition("nonexistent_agent")
    assert result is None


def test_list_agent_definitions():
    """Test listing all agent definitions."""
    from mcp_agent.agents.agent_spec import AgentSpec

    # Clear existing definitions first
    import mcp_eval.config as config_module

    config_module._agent_definitions.clear()

    agent1 = AgentSpec(name="agent1", description="Agent 1", tools=[])
    agent2 = AgentSpec(name="agent2", description="Agent 2", tools=[])

    register_agent_definition("first", agent1)
    register_agent_definition("second", agent2)

    definitions = list_agent_definitions()
    assert "first" in definitions
    assert "second" in definitions
    assert definitions["first"] == agent1
    assert definitions["second"] == agent2


def test_load_config_invalid_yaml():
    """Test loading invalid YAML config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        invalid_file = f.name

    try:
        with pytest.raises(yaml.YAMLError):
            load_config(invalid_file)
    finally:
        Path(invalid_file).unlink()


def test_load_config_nonexistent():
    """Test loading nonexistent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_mcp_eval_settings_with_mcp():
    """Test MCPEvalSettings with MCP configuration."""
    mcp_config = {
        "servers": {"server1": {"command": "cmd1"}, "server2": {"command": "cmd2"}}
    }
    settings = MCPEvalSettings(
        name="MCP Test", description="Test with MCP", mcp=mcp_config
    )
    assert settings.mcp == mcp_config
    assert len(settings.mcp["servers"]) == 2


def test_mcp_eval_settings_with_agents():
    """Test MCPEvalSettings with agents configuration."""
    agents_config = {
        "definitions": [
            {"name": "agent1", "tools": ["tool1"]},
            {"name": "agent2", "tools": ["tool2"]},
        ]
    }
    settings = MCPEvalSettings(
        name="Agents Test", description="Test with agents", agents=agents_config
    )
    assert settings.agents == agents_config
    assert len(settings.agents["definitions"]) == 2


def test_load_config_with_all_settings():
    """Test loading config with all settings sections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = """
name: Complete Config
description: Test all settings
provider: anthropic
model: claude-3
timeout: 120
judge:
  model: gpt-4
  min_score: 0.8
otel:
  enabled: false
  exporters: [jaeger]
logger:
  level: DEBUG
  transports: [console]
reporting:
  formats: [json]
  output_dir: /custom/reports
mcp:
  servers:
    test: {command: test}
agents:
  definitions:
    - name: test
      tools: [tool1]
        """
        f.write(config)
        config_file = f.name

    try:
        settings = load_config(config_file)
        assert settings.name == "Complete Config"
        assert settings.provider == "anthropic"
        assert settings.model == "claude-3"
        assert settings.timeout == 120
        assert settings.judge.model == "gpt-4"
        assert settings.otel.enabled is False
        assert settings.logger.level == "DEBUG"
        assert settings.reporting.output_dir == "/custom/reports"
        assert "test" in settings.mcp["servers"]
        assert len(settings.agents["definitions"]) == 1
    finally:
        Path(config_file).unlink()
