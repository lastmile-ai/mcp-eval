"""Utilities for accessing MCP-Eval package data."""

from pathlib import Path
from importlib import resources
from typing import Optional, List


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


def get_sample_resources_path() -> Optional[Path]:
    """Return the path-like object to the packaged sample template directory.

    Returns None if the resources cannot be accessed as a real directory
    (e.g., when installed as a zip). In that case, copying individual files
    is still supported via importlib.resources APIs.
    """
    try:
        files = resources.files("mcp_eval.data").joinpath("sample")
        # If the files object supports iterdir, we can treat it as a directory
        if hasattr(files, "iterdir"):
            return Path(str(files))
    except Exception:
        pass

    # Development fallback
    try:
        import mcp_eval

        package_dir = Path(mcp_eval.__file__).parent
        sample_dir = package_dir / "data" / "sample"
        if sample_dir.exists():
            return sample_dir
    except Exception:
        return None

    return None


def copy_sample_template(
    project_dir: Path,
    *,
    files_to_copy: Optional[List[str]] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Copy the packaged sample template files into a project directory.

    Args:
        project_dir: Destination directory.
        files_to_copy: Optional allowlist of filenames to copy. If omitted,
            a sensible default set is used.
        overwrite: If True, existing files will be overwritten.

    Returns:
        A list of destination file paths that were written.
    """
    project_dir.mkdir(parents=True, exist_ok=True)

    default_files = [
        "usage_example.py",
        "sample_server.py",
        "sample_server.eval.py",
        "README.md",
        "mcpeval.yaml",
        "mcpeval.secrets.yaml.example",
    ]
    targets = files_to_copy or default_files

    written: List[Path] = []

    # Use importlib.resources to read bytes/text regardless of packaging
    pkg = "mcp_eval.data.sample"
    for name in targets:
        try:
            src = resources.files(pkg).joinpath(name)
        except Exception:
            continue

        if not src:
            continue

        dest = project_dir / name
        if dest.exists() and not overwrite:
            # Skip existing files by default
            continue

        # Read as binary to preserve content exactly
        with resources.as_file(src) as src_path:
            data = Path(src_path).read_bytes()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(dest)

    return written
