"""Configuration management for MCP-Eval."""

import os
import yaml
from typing import Dict, Any, Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class ServerConfig(BaseSettings):
    """Configuration for an MCP server."""

    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseSettings):
    """Configuration for an agent."""

    name: str = "default_agent"
    instruction: str = "You are a helpful test agent."
    llm_factory: str = "AnthropicAugmentedLLM"
    model: str = "claude-3-haiku-20240307"
    max_iterations: int = 5


class JudgeConfig(BaseSettings):
    """Configuration for LLM judge."""

    model: str = "claude-3-haiku-20240307"
    min_score: float = 0.8
    max_tokens: int = 1000
    system_prompt: str = "You are an expert evaluator of AI assistant responses."


class MetricsConfig(BaseSettings):
    """Configuration for metrics collection."""

    collect: List[str] = Field(
        default_factory=lambda: [
            "response_time",
            "tool_coverage",
            "iteration_count",
            "token_usage",
            "cost_estimate",
        ]
    )
    token_prices: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125},
            "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
        }
    )


class ReportingConfig(BaseSettings):
    """Configuration for reporting."""

    formats: List[str] = Field(default_factory=lambda: ["json", "markdown"])
    output_dir: str = "./test-reports"
    include_traces: bool = True


class OTelConfig(BaseSettings):
    """Configuration for OpenTelemetry."""

    enabled: bool = True
    exporter_type: str = "file"
    exporter_config: Dict[str, Any] = Field(default_factory=dict)
    service_name: str = "mcp-eval"


class ExecutionConfig(BaseSettings):
    """Configuration for test execution."""

    max_concurrency: int = 5
    timeout_seconds: int = 300
    retry_failed: bool = False


class MCPEvalSettings(BaseSettings):
    """Complete MCP-Eval configuration with validation."""

    # Basic info
    name: str = "MCP-Eval Test Suite"
    description: str = "Comprehensive evaluation of MCP servers"

    # Server configurations
    servers: Dict[str, ServerConfig] = Field(default_factory=dict)

    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(
        default_factory=lambda: {"default": AgentConfig()}
    )

    # Component configurations
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    otel: OTelConfig = Field(default_factory=OTelConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Current runtime state
    default_server: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None

    @field_validator("servers", mode="before")
    @classmethod
    def validate_servers(cls, v):
        """Convert dict configs to ServerConfig objects."""
        if isinstance(v, dict):
            return {
                name: ServerConfig(**config) if isinstance(config, dict) else config
                for name, config in v.items()
            }
        return v

    @field_validator("agents", mode="before")
    @classmethod
    def validate_agents(cls, v):
        """Convert dict configs to AgentConfig objects."""
        if isinstance(v, dict):
            return {
                name: AgentConfig(**config) if isinstance(config, dict) else config
                for name, config in v.items()
            }
        return v

    class Config:
        env_prefix = "MCPEVAL_"
        env_file = ".env"
        case_sensitive = False


# Global configuration state
_current_settings: Optional[MCPEvalSettings] = None


def load_config(config_path: Optional[str] = None) -> MCPEvalSettings:
    """Load configuration from file with full validation."""
    global _current_settings

    config_data = {}

    if config_path is None:
        # Look for config in standard locations
        for path in [
            "mcpeval.yaml",
            "mcpeval.yml",
            ".mcpeval.yaml",
            "mcp_eval.yaml",
            "mcp_eval.yml",
            ".mcp_eval.yaml",
            "mcp-eval.yaml",
            "mcp-eval.yml",
            ".mcp-eval.yaml",
        ]:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Create settings with validation
    _current_settings = MCPEvalSettings(**config_data)
    return _current_settings


def get_current_config() -> Dict[str, Any]:
    """Get current configuration as dict for backward compatibility."""
    if _current_settings is None:
        load_config()

    return {
        "default_server": _current_settings.default_server,
        "agent_config": _current_settings.agent_config
        or _current_settings.agents["default"].dict(),
        "servers": {
            name: config.dict() for name, config in _current_settings.servers.items()
        },
        "judge": _current_settings.judge.dict(),
        "metrics": _current_settings.metrics.dict(),
        "reporting": _current_settings.reporting.dict(),
        "otel": _current_settings.otel.dict(),
        "execution": _current_settings.execution.dict(),
    }


def get_settings() -> MCPEvalSettings:
    """Get current settings object."""
    if _current_settings is None:
        load_config()
    return _current_settings


def update_config(config: Dict[str, Any]):
    """Update current configuration."""
    global _current_settings
    if _current_settings is None:
        load_config()

    # Update specific fields
    for key, value in config.items():
        if hasattr(_current_settings, key):
            setattr(_current_settings, key, value)


def use_server(server_name: str):
    """Configure default server."""
    if _current_settings is None:
        load_config()
    _current_settings.default_server = server_name


def use_agent(agent_config: Dict[str, Any]):
    """Configure default agent."""
    if _current_settings is None:
        load_config()
    _current_settings.agent_config = agent_config


def use_llm_factory(llm_factory: type):
    """Configure default LLM factory."""
    if _current_settings is None:
        load_config()

    if _current_settings.agent_config is None:
        _current_settings.agent_config = {}
    _current_settings.agent_config["llm_factory"] = llm_factory


# Initialize with file config on import
_current_settings = load_config()
