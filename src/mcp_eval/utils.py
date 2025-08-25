"""Utility functions for mcp-eval."""

import re
from pathlib import Path
from typing import Tuple


def sanitize_test_name_for_filesystem(test_name: str) -> str:
    """Sanitize a test name for use as a filename.

    Args:
        test_name: The test name to sanitize

    Returns:
        A sanitized version safe for filesystem use
    """
    # Replace filesystem-unsafe characters with underscore
    # This matches the pattern used in session.py
    safe_name = re.sub(r'[<>:"/\\|?*\[\]]', "_", test_name)
    return safe_name


def get_test_artifact_paths(
    test_name: str, output_dir: Path = None
) -> Tuple[Path, Path]:
    """Get the paths for test artifact files (trace and metrics).

    Args:
        test_name: The test name
        output_dir: Optional output directory (defaults to ./test-reports)

    Returns:
        Tuple of (trace_file_path, metrics_file_path)
    """
    if output_dir is None:
        # Try to get from config
        try:
            from mcp_eval.config import get_current_config

            config = get_current_config()
            reporting_config = config.get("reporting", {})
            output_dir = Path(reporting_config.get("output_dir", "./test-reports"))
        except Exception:
            output_dir = Path("./test-reports")

    # Sanitize the test name
    safe_test_name = sanitize_test_name_for_filesystem(test_name)

    # Generate file paths
    trace_file = output_dir / f"{safe_test_name}_trace.jsonl"
    metrics_file = output_dir / f"{safe_test_name}.json"

    return trace_file, metrics_file
