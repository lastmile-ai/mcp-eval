"""Configuration management for MCP-Eval."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


# Global configuration state
_current_config: Dict[str, Any] = {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path is None:
        # Look for config in standard locations
        for path in ["mcpeval.yaml", "mcpeval.yml", ".mcpeval.yaml"]:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    return {}


def get_current_config() -> Dict[str, Any]:
    """Get current configuration."""
    return _current_config.copy()


def update_config(config: Dict[str, Any]):
    """Update current configuration."""
    _current_config.update(config)


def use_server(server_name: str):
    """Configure default server."""
    _current_config['default_server'] = server_name


def use_agent(agent_config: Dict[str, Any]):
    """Configure default agent."""
    _current_config['agent_config'] = agent_config


def use_llm_factory(llm_factory: type):
    """Configure default LLM factory."""
    if 'agent_config' not in _current_config:
        _current_config['agent_config'] = {}
    _current_config['agent_config']['llm_factory'] = llm_factory


# Initialize with file config on import
_current_config.update(load_config())