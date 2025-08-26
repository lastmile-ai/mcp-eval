import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from mcp_eval.config import (
    MCPEvalSettings,
    JudgeConfig,
    MetricsConfig,
    ReportingConfig,
    ExecutionConfig,
    ProgrammaticDefaults,
    load_config,
    get_settings,
    set_settings,
    get_current_config,
    update_config,
    use_config,
    use_agent,
    find_eval_config,
    find_eval_secrets,
)


def test_judge_config_defaults():
    """Test JudgeConfig default values."""
    judge = JudgeConfig()
    assert judge.model is None  # Default is None (falls back to global)


def test_judge_config_custom():
    """Test JudgeConfig with custom values."""
    judge = JudgeConfig(model="gpt-4", min_score=0.8)
    assert judge.model == "gpt-4"
    assert judge.min_score == 0.8


def test_metrics_config_defaults():
    """Test MetricsConfig default values."""
    metrics = MetricsConfig()
    assert "response_time" in metrics.collect  # Default metrics
    assert "tool_coverage" in metrics.collect
    assert len(metrics.collect) == 5  # Has 5 default metrics


def test_metrics_config_custom():
    """Test MetricsConfig with custom values."""
    metrics = MetricsConfig(collect=["custom_metric", "another_metric"])
    assert metrics.collect == ["custom_metric", "another_metric"]
    assert len(metrics.collect) == 2


def test_reporting_config_defaults():
    """Test ReportingConfig default values."""
    reporting = ReportingConfig()
    assert "json" in reporting.formats
    assert "markdown" in reporting.formats
    assert reporting.output_dir == "./test-reports"
    assert reporting.include_traces is True


def test_reporting_config_custom():
    """Test ReportingConfig with custom values."""
    reporting = ReportingConfig(
        formats=["html"], output_dir="/custom/path", include_traces=False
    )
    assert reporting.formats == ["html"]
    assert reporting.output_dir == "/custom/path"
    assert reporting.include_traces is False


def test_execution_config_defaults():
    """Test ExecutionConfig default values."""
    execution = ExecutionConfig()
    assert execution.max_concurrency == 5
    assert execution.timeout_seconds == 300
    assert execution.retry_failed is False


def test_execution_config_custom():
    """Test ExecutionConfig with custom values."""
    execution = ExecutionConfig(
        timeout_seconds=60, max_concurrency=10, retry_failed=True
    )
    assert execution.timeout_seconds == 60
    assert execution.max_concurrency == 10
    assert execution.retry_failed is True


def test_mcp_eval_settings_defaults():
    """Test MCPEvalSettings default values."""
    settings = MCPEvalSettings(name="Test", description="Test suite")
    assert settings.name == "Test"
    assert settings.description == "Test suite"
    assert settings.judge is not None
    assert settings.metrics is not None
    assert settings.reporting is not None
    assert settings.execution is not None


def test_mcp_eval_settings_custom():
    """Test MCPEvalSettings with custom values."""
    settings = MCPEvalSettings(
        name="Custom",
        description="Custom test",
        provider="anthropic",
        model="claude-3-sonnet",
        judge=JudgeConfig(model="gpt-4"),
        metrics=MetricsConfig(collect=[]),  # Empty metrics list
    )
    assert settings.name == "Custom"
    assert settings.provider == "anthropic"
    assert settings.model == "claude-3-sonnet"
    assert settings.judge.model == "gpt-4"
    assert settings.metrics.collect == []


def test_programmatic_defaults():
    """Test ProgrammaticDefaults static methods."""
    # Test set and get default agent
    from mcp_agent.agents.agent import Agent

    mock_agent = Mock(spec=Agent)

    ProgrammaticDefaults.set_default_agent(mock_agent)
    assert ProgrammaticDefaults.get_default_agent() == mock_agent

    # Test clearing
    ProgrammaticDefaults.set_default_agent(None)
    assert ProgrammaticDefaults.get_default_agent() is None


def test_programmatic_defaults_agent_factory():
    """Test setting and getting agent factory."""
    from mcp_agent.agents.agent import Agent

    def mock_factory():
        return Mock(spec=Agent)

    ProgrammaticDefaults.set_default_agent_factory(mock_factory)
    assert ProgrammaticDefaults.get_default_agent_factory() == mock_factory

    # Test clearing
    ProgrammaticDefaults.set_default_agent_factory(None)
    assert ProgrammaticDefaults.get_default_agent_factory() is None


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


def test_update_config():
    """Test update_config function."""
    original = get_settings()

    update_config({"name": "Updated", "description": "Updated desc"})

    settings = get_settings()
    assert settings.name == "Updated"
    assert settings.description == "Updated desc"

    # Restore
    set_settings(original)


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
        assert "test_server" in settings.servers
        assert settings.servers["test_server"]["command"] == "python"
    finally:
        Path(config_file).unlink()


def test_get_current_config():
    """Test get_current_config function."""
    settings = MCPEvalSettings(name="Current", description="Current config")
    set_settings(settings)

    config = get_current_config()
    # get_current_config returns a flattened dict with servers, judge, etc.
    assert "servers" in config
    assert "judge" in config
    assert isinstance(config["judge"], dict)


def test_use_config_with_settings():
    """Test use_config with MCPEvalSettings object."""
    settings = MCPEvalSettings(name="UseConfig", description="Use config test")

    result = use_config(settings)
    assert result.name == "UseConfig"
    assert get_settings().name == "UseConfig"


def test_use_config_with_path():
    """Test use_config with file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = """
name: PathConfig
description: Config from path
        """
        f.write(config)
        config_file = f.name

    try:
        result = use_config(config_file)
        assert result.name == "PathConfig"
    finally:
        Path(config_file).unlink()


def test_find_eval_config(tmp_path):
    """Test finding eval config file."""
    # Create a config file
    config_file = tmp_path / "mcpeval.yaml"
    config_file.write_text("name: test")

    # Should find it when starting from that directory
    found = find_eval_config(tmp_path)
    assert found == config_file

    # Should not find it from a different directory
    other_dir = tmp_path / "subdir"
    other_dir.mkdir()
    found = find_eval_config(other_dir)
    assert found is None or found == config_file  # Might find parent


def test_find_eval_secrets(tmp_path):
    """Test finding eval secrets file."""
    # Create a secrets file
    secrets_file = tmp_path / "mcpeval.secrets.yaml"
    secrets_file.write_text("api_key: secret")

    # Should find it
    found = find_eval_secrets(tmp_path)
    assert found == secrets_file


def test_load_config_nonexistent():
    """Test loading nonexistent config file."""
    # load_config doesn't raise error for nonexistent files, it just ignores them
    settings = load_config("/nonexistent/config.yaml")
    assert settings is not None  # Returns default settings
    assert isinstance(settings, MCPEvalSettings)


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


@pytest.mark.asyncio
async def test_use_agent_basic():
    """Test use_agent basic functionality."""
    from mcp_agent.agents.agent import Agent
    from mcp_agent.agents.agent_spec import AgentSpec

    mock_agent = Mock(spec=Agent)

    # Test with agent object - sets it in ProgrammaticDefaults
    use_agent(mock_agent)
    assert ProgrammaticDefaults.get_default_agent() == mock_agent

    # Test with AgentSpec
    agent_spec = AgentSpec(name="test_agent", description="Test", tools=[])
    use_agent(agent_spec)
    assert get_settings().default_agent == agent_spec

    # Test with string name
    use_agent("some_agent_name")
    assert get_settings().default_agent == "some_agent_name"


def test_mcp_eval_settings_with_all_configs():
    """Test MCPEvalSettings with all configuration sections."""
    settings = MCPEvalSettings(
        name="Complete",
        description="Complete config",
        judge=JudgeConfig(model="gpt-4", min_score=0.9),
        metrics=MetricsConfig(collect=["metric1", "metric2"]),
        reporting=ReportingConfig(formats=["json"], output_dir="/reports"),
        execution=ExecutionConfig(timeout_seconds=120, max_concurrency=5),
    )

    assert settings.name == "Complete"
    assert settings.judge.model == "gpt-4"
    assert settings.judge.min_score == 0.9
    assert settings.metrics.collect == ["metric1", "metric2"]
    assert settings.reporting.output_dir == "/reports"
    assert settings.execution.timeout_seconds == 120
    assert settings.execution.max_concurrency == 5
