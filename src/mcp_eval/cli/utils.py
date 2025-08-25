"""Utilities for CLI operations."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console

from mcp_eval.cli.models import (
    ServerImport,
    ConfigPaths,
    MCPServerConfig,
    AgentConfig,
    MCPEvalConfig,
)

console = Console()


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[red]Error loading {path}: {e}[/red]")
        return {}


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Save data to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def find_mcp_json() -> Path | None:
    """Look for mcp.json in common locations."""
    candidates = [
        Path(".cursor/mcp.json"),
        Path(".vscode/mcp.json"),
        Path("mcp.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_mcpeval_config(project_dir: Path | None = None) -> Path | None:
    """Find mcpeval config file using the same logic as config.py.

    Searches for config files in this order:
    - mcpeval.yaml | mcpeval.yml
    - mcpeval.config.yaml | mcpeval.config.yml
    - .mcp-eval/config.yaml | .mcp-eval/config.yml
    - .mcp-eval.config.yaml | .mcp-eval.config.yml

    Returns the first one found, or None if none exist.
    """
    start_dir = project_dir or Path.cwd()
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

    # Search upwards from start_dir
    cur = start_dir
    while True:
        for candidate in candidates:
            path = cur / candidate
            if path.exists():
                return path
        if cur == cur.parent:
            break
        cur = cur.parent

    # Check home directory for .mcp-eval/* patterns
    try:
        home = Path.home()
        for candidate in candidates:
            if candidate.startswith(".mcp-eval/"):
                path = home / candidate
                if path.exists():
                    return path
            elif candidate.startswith(".mcp-eval"):
                path = home / ".mcp-eval" / candidate.replace(".mcp-eval.", "")
                if path.exists():
                    return path
    except Exception:
        pass

    return None


def get_default_mcpeval_config_name() -> str:
    """Get the default name for new mcpeval config files."""
    return "mcpeval.yaml"


def find_config_files(project_dir: Path) -> ConfigPaths:
    """Find all relevant configuration files."""
    # Use the new utility to find mcpeval config
    mcpeval_config = find_mcpeval_config(project_dir)

    # For backwards compatibility, if no config found, use default location
    if mcpeval_config is None:
        mcpeval_config = project_dir / get_default_mcpeval_config_name()

    return ConfigPaths(
        mcpeval_yaml=mcpeval_config,
        mcpeval_secrets=project_dir / "mcpeval.secrets.yaml",
        mcp_agent_config=project_dir / "mcp-agent.config.yaml",
        mcp_json=find_mcp_json(),
    )


def import_servers_from_json(json_path: Path) -> ServerImport:
    """Import servers from mcp.json format."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        servers: Dict[str, MCPServerConfig] = {}
        mcp_servers = data.get("mcpServers") or data.get("servers") or {}

        for name, cfg in mcp_servers.items():
            try:
                server = MCPServerConfig(
                    name=name,
                    transport=cfg.get(
                        "transport", "stdio" if cfg.get("command") else "sse"
                    ),
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                    env=cfg.get("env"),
                )
                servers[name] = server
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to import server '{name}': {e}[/yellow]"
                )

        return ServerImport(
            servers={name: s.model_dump() for name, s in servers.items()},
            source=str(json_path),
            success=True,
        )
    except Exception as e:
        return ServerImport(
            servers={}, source=str(json_path), success=False, error=str(e)
        )


def import_servers_from_dxt(dxt_path: Path) -> ServerImport:
    """Import servers from DXT manifest file."""
    try:
        text = dxt_path.read_text(encoding="utf-8")

        # Try JSON first, then YAML
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = yaml.safe_load(text)

        if not isinstance(data, dict):
            return ServerImport(
                servers={}, source=str(dxt_path), success=False, error="Invalid format"
            )

        # Look for mcpServers in DXT
        if "mcpServers" in data and isinstance(data["mcpServers"], dict):
            servers: Dict[str, MCPServerConfig] = {}
            for name, cfg in data["mcpServers"].items():
                try:
                    server = MCPServerConfig(
                        name=name,
                        transport="stdio",
                        command=cfg.get("command"),
                        args=cfg.get("args", []),
                        env=cfg.get("env"),
                    )
                    servers[name] = server
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to import server '{name}': {e}[/yellow]"
                    )

            return ServerImport(
                servers={name: s.model_dump() for name, s in servers.items()},
                source=str(dxt_path),
                success=True,
            )

        return ServerImport(
            servers={}, source=str(dxt_path), success=False, error="No mcpServers found"
        )
    except Exception as e:
        return ServerImport(
            servers={}, source=str(dxt_path), success=False, error=str(e)
        )


def load_all_servers(project_dir: Path) -> Dict[str, MCPServerConfig]:
    """Load servers from all configuration sources."""
    servers: Dict[str, MCPServerConfig] = {}
    paths = find_config_files(project_dir)

    # Priority order: mcpeval.yaml > mcp-agent.config.yaml > other sources

    # Load from mcp-agent.config.yaml
    if paths.mcp_agent_config.exists():
        data = load_yaml(paths.mcp_agent_config)
        for name, cfg in data.get("mcp", {}).get("servers", {}).items():
            try:
                servers[name] = MCPServerConfig(
                    name=name,
                    transport=cfg.get("transport", "stdio"),
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                    env=cfg.get("env"),
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Invalid server config for '{name}': {e}[/yellow]"
                )

    # Load from mcpeval.yaml (overrides mcp-agent.config.yaml)
    if paths.mcpeval_yaml.exists():
        data = load_yaml(paths.mcpeval_yaml)
        for name, cfg in data.get("mcp", {}).get("servers", {}).items():
            try:
                servers[name] = MCPServerConfig(
                    name=name,
                    transport=cfg.get("transport", "stdio"),
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                    env=cfg.get("env"),
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Invalid server config for '{name}': {e}[/yellow]"
                )

    # Also check .mcp-eval/config.yaml
    alt_config = project_dir / ".mcp-eval" / "config.yaml"
    if alt_config.exists():
        data = load_yaml(alt_config)
        for name, cfg in data.get("mcp", {}).get("servers", {}).items():
            if name not in servers:  # Don't override existing
                try:
                    servers[name] = MCPServerConfig(
                        name=name,
                        transport=cfg.get("transport", "stdio"),
                        command=cfg.get("command"),
                        args=cfg.get("args", []),
                        url=cfg.get("url"),
                        headers=cfg.get("headers"),
                        env=cfg.get("env"),
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Invalid server config for '{name}': {e}[/yellow]"
                    )

    return servers


def load_all_agents(project_dir: Path) -> List[AgentConfig]:
    """Load agent definitions from configuration."""
    agents: List[AgentConfig] = []
    paths = find_config_files(project_dir)

    if paths.mcpeval_yaml.exists():
        data = load_yaml(paths.mcpeval_yaml)
        agent_defs = data.get("agents", {}).get("definitions", [])

        for agent_def in agent_defs:
            if isinstance(agent_def, dict) and "name" in agent_def:
                try:
                    agents.append(
                        AgentConfig(
                            name=agent_def["name"],
                            instruction=agent_def.get(
                                "instruction", "Complete the task as requested."
                            ),
                            server_names=agent_def.get("server_names", []),
                            provider=agent_def.get("provider"),
                            model=agent_def.get("model"),
                        )
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Invalid agent config: {e}[/yellow]"
                    )

    return agents


def ensure_mcpeval_yaml(project_dir: Path) -> Path:
    """Ensure mcpeval config exists with minimal defaults.

    First tries to find an existing config file using the full search logic.
    If none found, creates a new one with the default name.
    """
    # Try to find existing config
    config_path = find_mcpeval_config(project_dir)

    if config_path is None:
        # No existing config, create with default name
        config_path = project_dir / get_default_mcpeval_config_name()
        console.print(
            f"[yellow]Creating {get_default_mcpeval_config_name()} with defaults...[/yellow]"
        )
        config = MCPEvalConfig()
        save_yaml(config_path, config.model_dump(exclude_none=True))
        console.print(f"[green]✓ Created {config_path}[/green]")

    return config_path


def write_server_to_mcpeval(project_dir: Path, server: MCPServerConfig) -> None:
    """Write server configuration to mcpeval.yaml."""
    config_path = ensure_mcpeval_yaml(project_dir)
    data = load_yaml(config_path)

    if "mcp" not in data:
        data["mcp"] = {}
    if "servers" not in data["mcp"]:
        data["mcp"]["servers"] = {}

    # Convert to dict for storage
    server_dict = {"transport": server.transport}
    if server.command:
        server_dict["command"] = server.command
    if server.args:
        server_dict["args"] = server.args
    if server.url:
        server_dict["url"] = server.url
    if server.headers:
        server_dict["headers"] = server.headers
    if server.env:
        server_dict["env"] = server.env

    data["mcp"]["servers"][server.name] = server_dict
    save_yaml(config_path, data)
    console.print(f"[green]✓ Added server '{server.name}' to {config_path}[/green]")


def write_agent_to_mcpeval(
    project_dir: Path, agent: AgentConfig, set_default: bool = False
) -> None:
    """Write agent configuration to mcpeval.yaml."""
    config_path = ensure_mcpeval_yaml(project_dir)
    data = load_yaml(config_path)

    if "agents" not in data:
        data["agents"] = {}
    if "definitions" not in data["agents"]:
        data["agents"]["definitions"] = []

    # Remove existing agent with same name
    existing = [a for a in data["agents"]["definitions"] if isinstance(a, dict)]
    existing = [a for a in existing if a.get("name") != agent.name]

    # Add new agent
    agent_dict = agent.model_dump(exclude_none=True)
    existing.append(agent_dict)
    data["agents"]["definitions"] = existing

    # Set as default if requested
    if set_default:
        data["default_agent"] = agent.name

    save_yaml(config_path, data)
    console.print(f"[green]✓ Added agent '{agent.name}' to {config_path}[/green]")
    if set_default:
        console.print(f"[green]✓ Set '{agent.name}' as default agent[/green]")
