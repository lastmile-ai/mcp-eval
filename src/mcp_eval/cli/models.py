"""Data models for CLI operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field as PydField


@dataclass
class ServerImport:
    """Result of importing servers from external sources."""

    servers: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class ConfigPaths:
    """Paths to various configuration files."""

    mcpeval_yaml: Path
    mcpeval_secrets: Path
    mcp_agent_config: Path
    mcp_json: Optional[Path] = None
    dxt_file: Optional[Path] = None


class MCPServerConfig(BaseModel):
    """Strongly typed MCP server configuration."""

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = PydField(default_factory=list)
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None

    def to_mcp_agent_settings(self) -> Dict[str, Any]:
        """Convert to mcp_agent.MCPServerSettings compatible dict."""
        result: Dict[str, Any] = {"name": self.name, "transport": self.transport}
        if self.command:
            result["command"] = self.command
        if self.args:
            result["args"] = self.args
        if self.url:
            result["url"] = self.url
        if self.headers:
            result["headers"] = self.headers
        if self.env:
            result["env"] = self.env
        return result


class AgentConfig(BaseModel):
    """Strongly typed agent configuration."""

    name: str
    instruction: str
    server_names: List[str]
    provider: Optional[str] = None
    model: Optional[str] = None


class MCPEvalConfig(BaseModel):
    """Root configuration for mcpeval.yaml."""

    reporting: Dict[str, Any] = PydField(
        default_factory=lambda: {
            "formats": ["json", "markdown"],
            "output_dir": "./test-reports",
        }
    )
    judge: Dict[str, Any] = PydField(default_factory=lambda: {"min_score": 0.8})
    mcp: Optional[Dict[str, Any]] = None
    agents: Optional[Dict[str, Any]] = None
    default_agent: Optional[str] = None


class GeneratorState(BaseModel):
    """State for the generator command flow."""

    project_dir: Path
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    servers: Dict[str, MCPServerConfig] = PydField(default_factory=dict)
    agents: List[AgentConfig] = PydField(default_factory=list)
    selected_server: Optional[str] = None
    selected_agent: Optional[str] = None
    style: str = "pytest"
    n_examples: int = 6
