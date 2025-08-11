"""Configuration management for MCP-Eval built on top of mcp-agent Settings.

This module extends the mcp-agent configuration with evaluation-specific settings
and provides helpers that preserve backward compatibility with older mcp-eval
code paths while enabling consolidated configuration in a single file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Literal, Union, Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mcp_agent.config import Settings as AgentSettings
from mcp_agent.config import SubagentSettings
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM


class AgentConfig(BaseSettings):
    """Lightweight test-time agent overrides for convenience.

    Note: Prefer AgentSpec discovery via mcp-agent configuration for declarative
    agents. This structure remains for quick overrides in tests.
    """

    name: str = "default_agent"
    instruction: str = "You are a helpful test agent."
    llm_factory: Optional[str] = None
    model: Optional[str] = None
    max_iterations: int = 5


class JudgeConfig(BaseSettings):
    """Configuration for LLM judge."""

    model: str = "claude-3-5-haiku-20241022"
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
            "claude-3-5-haiku-20241022": {"input": 0.00000025, "output": 0.00000125},
            "claude-sonnet-4-20250514": {"input": 0.000003, "output": 0.000015},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
        }
    )


class ReportingConfig(BaseSettings):
    """Configuration for reporting."""

    formats: List[str] = Field(default_factory=lambda: ["json", "markdown"])
    output_dir: str = "./test-reports"
    include_traces: bool = True


class ExecutionConfig(BaseSettings):
    """Configuration for test execution."""

    max_concurrency: int = 5
    timeout_seconds: int = 300
    retry_failed: bool = False


class MCPEvalSettings(AgentSettings):
    """MCP-Eval settings that extend the base mcp-agent Settings.

    This allows a single YAML file (mcp-agent.config.yaml) to include both
    agent/server configuration and evaluation-specific settings under these
    typed fields.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        nested_model_default_partial_update=True,
    )

    # Evaluation metadata
    eval_name: str = "MCP-Eval Test Suite"
    eval_description: str = "Comprehensive evaluation of MCP servers"

    # Evaluation components
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Back-compat default servers for tests (preferred: set on Agent/AgentSpec)
    default_servers: List[str] | None = Field(default_factory=list)

    # Enable subagent discovery by default so AgentSpec files are picked up
    subagents: SubagentSettings | None = Field(default_factory=SubagentSettings)

    # Test-time overrides/state (not persisted)
    agent_config: Optional[Dict[str, object]] = None

    class ProgrammaticAgentConfig(BaseModel):
        kind: Literal[
            "agent_object",
            "llm_object",
            "agent_spec",
            "agent_spec_name",
            "overrides",
        ]
        agent: Agent | None = None
        llm: AugmentedLLM | None = None
        agent_spec: AgentSpec | None = None
        agent_spec_name: str | None = None
        overrides: Dict[str, object] | None = None

    programmatic_agent: ProgrammaticAgentConfig | None = None


def _deep_merge(base: dict, update: dict) -> dict:
    merged = base.copy()
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Global configuration state
_current_settings: Optional[MCPEvalSettings] = None


def _search_upwards_for(paths: List[str]) -> Optional[Path]:
    """Search current and parent directories (including .mcp-eval subdir) for first matching path.

    Supports both direct filenames and subdir patterns like '.mcp-eval/config.yaml'.
    Also checks home-level fallback under '~/.mcp-eval/'.
    """
    cwd = Path.cwd()
    # Walk up
    cur = cwd
    while True:
        for p in paths:
            candidate = cur / p
            if candidate.exists():
                return candidate
        if cur == cur.parent:
            break
        cur = cur.parent

    # Home fallback for .mcp-eval/* patterns
    try:
        home = Path.home()
        for p in paths:
            if p.startswith(".mcp-eval/"):
                candidate = home / p
                if candidate.exists():
                    return candidate
            elif p.startswith(".mcp-eval"):
                candidate = home / ".mcp-eval" / p.replace(".mcp-eval.", "")
                if candidate.exists():
                    return candidate
    except Exception:
        pass
    return None


def _find_eval_config() -> Optional[Path]:
    """Locate an mcp-eval config file.

    Looks for (in current dir upwards and ~/.mcp-eval):
    - mcpeval.yaml | mcpeval.yml
    - mcpeval.config.yaml | mcpeval.config.yml
    - .mcp-eval/config.yaml | .mcp-eval/config.yml
    - .mcp-eval.config.yaml | .mcp-eval.config.yml
    """
    candidates = [
        "mcpeval.yaml",
        "mcpeval.yml",
        "mcpeval.config.yaml",
        "mcpeval.config.yml",
        ".mcp-eval/config.yaml",
        ".mcp-eval/config.yml",
        ".mcp-eval.config.yaml",
        ".mcp-eval.config.yml",
    ]
    return _search_upwards_for(candidates)


def _find_eval_secrets() -> Optional[Path]:
    """Locate an mcp-eval secrets file.

    Looks for (in current dir upwards and ~/.mcp-eval):
    - mcpeval.secrets.yaml | mcpeval.secrets.yml
    - .mcp-eval/secrets.yaml | .mcp-eval/secrets.yml
    - .mcp-eval.secrets.yaml | .mcp-eval.secrets.yml
    """
    candidates = [
        "mcpeval.secrets.yaml",
        "mcpeval.secrets.yml",
        ".mcp-eval/secrets.yaml",
        ".mcp-eval/secrets.yml",
        ".mcp-eval.secrets.yaml",
        ".mcp-eval.secrets.yml",
    ]
    return _search_upwards_for(candidates)


def load_config(config_path: Optional[Union[str, Path]] = None) -> MCPEvalSettings:
    """Load configuration with full validation.

    Priority overlay (later overrides earlier where fields overlap):
    1. mcp-agent.config.yaml (+ secrets)
    2. mcp-eval config (.mcp-eval/config.yaml or .mcp-eval.config.yaml)
       (+ corresponding secrets)
    3. Explicit config_path if provided (highest precedence)
    """
    global _current_settings

    merged: dict[str, Any] = {}

    # 1) Base: mcp-agent config (+secrets)
    agent_cfg = AgentSettings.find_config() or None
    if agent_cfg and Path(agent_cfg).exists():
        with open(agent_cfg, "r", encoding="utf-8") as f:
            merged = yaml.safe_load(f) or {}
        # Merge mcp-agent secrets (same-dir or discovery)
        try:
            config_dir = Path(agent_cfg).parent
            secrets_merged = False
            for secrets_filename in [
                "mcp-agent.secrets.yaml",
                "mcp_agent.secrets.yaml",
            ]:
                secrets_file = config_dir / secrets_filename
                if secrets_file.exists():
                    with open(secrets_file, "r", encoding="utf-8") as sf:
                        secrets_data = yaml.safe_load(sf) or {}
                        merged = _deep_merge(merged, secrets_data)
                    secrets_merged = True
                    break
            if not secrets_merged:
                secrets_file = AgentSettings.find_secrets()
                if secrets_file and Path(secrets_file).exists():
                    with open(secrets_file, "r", encoding="utf-8") as sf:
                        secrets_data = yaml.safe_load(sf) or {}
                        merged = _deep_merge(merged, secrets_data)
        except Exception:
            pass

    # 2) Overlay: mcp-eval config (+secrets)
    eval_cfg = None
    if config_path:
        # Allow passing an explicit mcp-eval config file path
        p = Path(config_path)
        if p.exists():
            eval_cfg = p
    if not eval_cfg:
        eval_cfg = _find_eval_config()
    if eval_cfg and eval_cfg.exists():
        with open(eval_cfg, "r", encoding="utf-8") as f:
            eval_data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, eval_data)
        # Merge mcp-eval secrets
        try:
            # Prefer a sibling secrets file if found
            eval_secrets = _find_eval_secrets()
            if eval_secrets and eval_secrets.exists():
                with open(eval_secrets, "r", encoding="utf-8") as sf:
                    secrets_data = yaml.safe_load(sf) or {}
                    merged = _deep_merge(merged, secrets_data)
        except Exception:
            pass

    _current_settings = MCPEvalSettings(**(merged or {}))
    return _current_settings


def get_current_config() -> Dict[str, Any]:
    """Flattened dict view (no legacy mcpeval.yaml support)."""
    if _current_settings is None:
        load_config()

    if _current_settings is None:
        raise RuntimeError("Configuration could not be loaded.")

    # Servers are now under settings.mcp.servers
    servers_dict: dict[str, Any] = {}
    if _current_settings.mcp and _current_settings.mcp.servers:
        servers_dict = {
            name: cfg.model_dump()
            for name, cfg in _current_settings.mcp.servers.items()
        }

    # OTEL is part of base settings
    otel_dump = (
        _current_settings.otel.model_dump()
        if getattr(_current_settings, "otel", None)
        else {}
    )

    return {
        "agent_config": _current_settings.agent_config or AgentConfig().model_dump(),
        "servers": servers_dict,
        "judge": _current_settings.judge.model_dump(),
        "metrics": _current_settings.metrics.model_dump(),
        "reporting": _current_settings.reporting.model_dump(),
        "otel": otel_dump,
        "execution": _current_settings.execution.model_dump(),
    }


def get_settings() -> MCPEvalSettings:
    """Get current typed settings object (MCPEvalSettings)."""
    if _current_settings is None:
        load_config()
    return _current_settings  # type: ignore[return-value]


def update_config(config: Dict[str, object]):
    """Update current configuration."""
    global _current_settings
    if _current_settings is None:
        load_config()

    # Update specific fields
    for key, value in config.items():
        if hasattr(_current_settings, key):
            setattr(_current_settings, key, value)


def set_settings(settings: Union[MCPEvalSettings, Dict[str, Any]]):
    """Programmatically set MCP‑Eval settings (bypass file discovery).

    Accepts either an MCPEvalSettings instance or a raw dict that will be
    validated against MCPEvalSettings.
    """
    global _current_settings
    if isinstance(settings, MCPEvalSettings):
        _current_settings = settings
    elif isinstance(settings, dict):
        _current_settings = MCPEvalSettings(**settings)
    else:
        raise TypeError("settings must be MCPEvalSettings or dict")


# Back-compat: use_server/use_servers – prefer defining server_names on Agent/AgentSpec


def use_server(server_name: str):
    """Set default server(s) used when constructing agents implicitly.

    Prefer setting `server_names` on your Agent/AgentSpec. This helper is
    provided for compatibility with existing examples/tests.
    """
    if _current_settings is None:
        load_config()
    defaults = _current_settings.default_servers or []
    if server_name not in defaults:
        defaults.append(server_name)
    _current_settings.default_servers = defaults


def use_servers(server_names: List[str]):
    """Replace the default server list used for implicit agent construction."""
    if _current_settings is None:
        load_config()
    _current_settings.default_servers = list(server_names)


def use_agent(
    agent_or_config: Union[Agent, AugmentedLLM, AgentSpec, Dict[str, object], str],
):
    """Configure default agent using a strongly-typed API.

    - Agent: programmatic agent instance
    - AugmentedLLM: programmatic LLM instance (its agent will be used)
    - AgentSpec: declarative agent spec
    - str: AgentSpec name (resolved from discovered subagents)
    - dict: lightweight overrides (name/instruction/llm_factory/model/server_names)
    """
    if _current_settings is None:
        load_config()
    if isinstance(agent_or_config, Agent):
        _current_settings.programmatic_agent = MCPEvalSettings.ProgrammaticAgentConfig(
            kind="agent_object", agent=agent_or_config
        )
        _current_settings.agent_config = None
        return
    if isinstance(agent_or_config, AugmentedLLM):
        _current_settings.programmatic_agent = MCPEvalSettings.ProgrammaticAgentConfig(
            kind="llm_object", llm=agent_or_config
        )
        _current_settings.agent_config = None
        return
    if isinstance(agent_or_config, AgentSpec):
        _current_settings.programmatic_agent = MCPEvalSettings.ProgrammaticAgentConfig(
            kind="agent_spec", agent_spec=agent_or_config
        )
        _current_settings.agent_config = None
        return
    if isinstance(agent_or_config, str):
        _current_settings.programmatic_agent = MCPEvalSettings.ProgrammaticAgentConfig(
            kind="agent_spec_name", agent_spec_name=agent_or_config
        )
        _current_settings.agent_config = None
        return
    if isinstance(agent_or_config, dict):
        _current_settings.programmatic_agent = MCPEvalSettings.ProgrammaticAgentConfig(
            kind="overrides", overrides=agent_or_config
        )
        _current_settings.agent_config = agent_or_config
        return
    raise TypeError("Unsupported agent configuration type")


def use_agent_object(obj: Union[Agent, AugmentedLLM]):
    """Explicitly set a programmatic agent or LLM instance for tests (strongly-typed)."""
    return use_agent(obj)


def use_llm_factory(llm_factory: type):
    """Configure default LLM factory."""
    if _current_settings is None:
        load_config()

    if _current_settings.agent_config is None:
        _current_settings.agent_config = {}
    _current_settings.agent_config["llm_factory"] = llm_factory


# Initialize with file config on import
_current_settings = load_config()
