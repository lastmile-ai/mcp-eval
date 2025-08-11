"""Shared utilities for report generation."""

import os
import sys
import platform
from datetime import datetime
from typing import Dict, Any, Optional
import yaml


def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.getcwd(),
    }


def load_config_info() -> Optional[Dict[str, Any]]:
    """Load MCP-Eval configuration from standard locations."""
    config_paths = ["mcpeval.yaml", "mcpeval.yml"]
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_info = yaml.safe_load(f)
                    config_info["_config_path"] = config_path
                return config_info
            except Exception:
                continue
    return None


def format_config_for_display(config_info: Dict[str, Any]) -> str:
    """Format configuration for display."""
    if not config_info:
        return "No configuration available"

    # Remove the _config_path key for display
    display_config = {k: v for k, v in config_info.items() if k != "_config_path"}
    return yaml.dump(display_config, default_flow_style=False, sort_keys=False)
