"""Configuration management for MCP-Eval built on top of mcp-agent Settings.

This module extends the mcp-agent configuration with evaluation-specific settings
and consolidates configuration in a single typed object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Callable
from contextvars import ContextVar

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mcp_agent.config import Settings as AgentSettings
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
import asyncio
from mcp_agent.core.context import initialize_context

# Deprecated AgentConfig removed


class JudgeConfig(BaseSettings):
    """Configuration for LLM judge.

    Supports separate provider/model configuration for judge evaluations.
    If not specified, falls back to global provider/model settings.
    """

    provider: str | None = None  # Judge-specific provider (falls back to global)
    model: str | None = (
        None  # Judge-specific model (falls back to global or ModelSelector)
    )
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

    # Default servers for tests (preferred: set on Agent/AgentSpec)
    default_servers: List[str] | None = Field(default_factory=list)

    # LLM defaults for tests
    provider: str | None = "anthropic"
    model: str | None = "claude-sonnet-4-0"

    default_agent: AgentSpec | str | None = None


def _deep_merge(base: dict, update: dict) -> dict:
    merged = base.copy()
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Global configuration state
_current_settings: MCPEvalSettings | None = None
_programmatic_default_agent: ContextVar[Union[Agent, AugmentedLLM, None]] = ContextVar(
    "programmatic_default_agent", default=None
)
_programmatic_default_agent_factory: ContextVar[
    Optional[Callable[[], Union[Agent, AugmentedLLM]]]
] = ContextVar("programmatic_default_agent_factory", default=None)


def _search_upwards_for(paths: List[str]) -> Path | None:
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


def _find_eval_config() -> Path | None:
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


def _find_eval_secrets() -> Path | None:
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


def load_config(config_path: Union[str, Path] | None = None) -> MCPEvalSettings:
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
    """Flattened dict view of current settings."""
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


class ProgrammaticDefaults:
    """Holds programmatic defaults that should not enter the pydantic schema.

    This avoids JSON schema/serialization issues while allowing users to set
    process-local defaults like a concrete Agent or AugmentedLLM.
    """

    @staticmethod
    def set_default_agent(value: Union[Agent, AugmentedLLM, None]) -> None:
        _programmatic_default_agent.set(value)

    @staticmethod
    def get_default_agent() -> Union[Agent, AugmentedLLM, None]:
        return _programmatic_default_agent.get()

    @staticmethod
    def set_default_agent_factory(
        value: Optional[Callable[[], Union[Agent, AugmentedLLM]]],
    ) -> None:
        _programmatic_default_agent_factory.set(value)

    @staticmethod
    def get_default_agent_factory() -> Optional[
        Callable[[], Union[Agent, AugmentedLLM]]
    ]:
        return _programmatic_default_agent_factory.get()


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


# Deprecated helpers removed: prefer defining server_names on Agent/AgentSpec


def use_config(config: Union[MCPEvalSettings, str]) -> MCPEvalSettings:
    """Programmatically set MCP‑Eval configuration.

    Accepts either a fully-formed MCPEvalSettings object, or a string path to a
    single config file. When a path is provided, ONLY that file is loaded and
    used to construct MCPEvalSettings – no default discovery/merging occurs.

    Returns the active MCPEvalSettings.
    """
    global _current_settings

    if isinstance(config, MCPEvalSettings):
        _current_settings = config
        return _current_settings

    if isinstance(config, str):
        p = Path(config)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {config}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _current_settings = MCPEvalSettings(**data)
        return _current_settings

    raise TypeError("use_config expects MCPEvalSettings or a path string")


def use_agent(
    agent_or_config: Union[Agent, AugmentedLLM, AgentSpec, str],
):
    """Configure default agent for tests.

    Supported:
    - AgentSpec: declarative agent spec (schema-serializable)
    - str: AgentSpec name (resolved from discovered subagents)
    - Agent | AugmentedLLM: programmatic defaults (stored outside settings schema)
    """
    if _current_settings is None:
        load_config()
    # Schema-safe storage for settings.default_agent (AgentSpec | str)
    if isinstance(agent_or_config, (AgentSpec, str)):
        _current_settings.default_agent = agent_or_config
        return
    # For programmatic Agent/AugmentedLLM, store in a side-channel
    if isinstance(agent_or_config, (Agent, AugmentedLLM)):
        ProgrammaticDefaults.set_default_agent(agent_or_config)
        ProgrammaticDefaults.set_default_agent_factory(None)
        return
    if isinstance(agent_or_config, dict):
        raise TypeError("Dict overrides removed. Use AgentSpec or name.")
    raise TypeError("Unsupported agent configuration type")


def use_agent_factory(factory: Callable[[], Union[Agent, AugmentedLLM]]):
    """Configure a factory for creating a default Agent/AugmentedLLM per session.

    This is the concurrency-safe way to set a programmatic default when running
    tests in parallel. Each TestSession will call the factory to obtain a fresh
    instance, avoiding shared mutable state.
    """
    if _current_settings is None:
        load_config()
    ProgrammaticDefaults.set_default_agent(None)
    ProgrammaticDefaults.set_default_agent_factory(factory)


def use_agent_object(obj: Union[Agent, AugmentedLLM]):
    """Explicitly set a programmatic agent or LLM instance for tests (strongly-typed)."""
    return use_agent(obj)


def use_llm_factory(llm_factory: type):
    raise NotImplementedError(
        "use_llm_factory removed. Configure provider/model in settings."
    )


async def create_test_context():
    """
    Create a properly configured context for programmatic agent creation in tests.

    This async function creates a context using mcp-eval settings (from mcpeval.yaml)
    including proper logging configuration, avoiding the issue where programmatic
    Agent creation uses default mcp-agent settings.

    Usage:
        from mcp_eval.config import create_test_context
        from mcp_agent.agents.agent import Agent

        # In an async function or test:
        context = await create_test_context()

        # Now create agents with this context
        agent = Agent(
            name="MyAgent",
            instruction="...",
            server_names=["fetch"],
            context=context  # Uses mcp-eval configured context
        )

    Returns:
        A Context object configured with mcp-eval settings
    """
    # Get the current mcp-eval settings
    settings = get_settings()

    # Create and initialize a context with these settings
    context = await initialize_context(config=settings)
    return context


def create_test_context_sync():
    """
    Synchronous wrapper for create_test_context().

    This is a convenience function for cases where you need to create
    a context outside of an async function. It handles the event loop
    management for you.

    Usage:
        from mcp_eval.config import create_test_context_sync
        from mcp_agent.agents.agent import Agent

        # Can be called from sync code
        context = create_test_context_sync()

        agent = Agent(
            name="MyAgent",
            instruction="...",
            server_names=["fetch"],
            context=context
        )

    Returns:
        A Context object configured with mcp-eval settings
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't use run_until_complete in a running loop
            # This typically happens when called from within an async context
            import warnings

            warnings.warn(
                "create_test_context_sync() called from async context. "
                "Consider using 'await create_test_context()' instead."
            )
            # Create a new event loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, create_test_context())
                return future.result()
        else:
            # No running loop, we can run it directly
            return loop.run_until_complete(create_test_context())
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(create_test_context())


# Initialize with file config on import
_current_settings = load_config()
