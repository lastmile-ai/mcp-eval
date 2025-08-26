"""Utilities for accessing MCP-Eval package data."""

from pathlib import Path
from importlib import resources
from typing import Optional


def get_subagents_search_path() -> Optional[str]:
    """Get the search path for MCP-Eval subagents.

    Returns:
        Path to subagents directory in the installed package, or None if not found.
    """
    try:
        # For Python 3.9+
        subagents = resources.files("mcp_eval.data").joinpath("subagents")
        if hasattr(subagents, "iterdir"):
            # It's a real directory (editable install or extracted wheel)
            return str(subagents)
        else:
            # It's in a zip/egg, we can't use it as a search path
            # Users would need to copy the subagents out
            return None
    except Exception:
        # Fallback for development
        # Check if we're running from source
        import mcp_eval

        package_dir = Path(mcp_eval.__file__).parent
        subagents_dir = package_dir / "data" / "subagents"
        if subagents_dir.exists():
            return str(subagents_dir)
        return None


def get_recommended_agents_config() -> dict:
    """Get the recommended agents configuration for mcpeval.yaml.

    Returns:
        Dictionary with agents configuration including search paths.
    """
    config = {
        "enabled": True,
        "pattern": "*.md",
        "search_paths": [
            ".claude/agents",
            "~/.claude/agents",
            ".mcp-agent/agents",
            "~/.mcp-agent/agents",
        ],
    }

    # Add package subagents path if available
    subagents_path = get_subagents_search_path()
    if subagents_path:
        config["search_paths"].insert(0, subagents_path)

    return config
