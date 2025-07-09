"""Monkey patch for mcp-agent to allow per-test trace files.

This patch modifies the FileSpanExporter to support dynamic file paths
based on the current test context.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global registry for test-specific trace files
_test_trace_files = {}


def register_test_trace_file(test_name: str, trace_file: str):
    """Register a trace file path for a specific test."""
    _test_trace_files[test_name] = trace_file
    logger.info(f"Registered trace file for {test_name}: {trace_file}")


def unregister_test_trace_file(test_name: str):
    """Unregister a test's trace file."""
    if test_name in _test_trace_files:
        del _test_trace_files[test_name]
        logger.info(f"Unregistered trace file for {test_name}")


def get_current_trace_file() -> Optional[str]:
    """Get the most recently registered trace file."""
    if _test_trace_files:
        # Return the most recently added trace file
        return list(_test_trace_files.values())[-1]
    return None


def patch_file_span_exporter():
    """Patch FileSpanExporter to use test-specific trace files."""
    try:
        from mcp_agent.tracing.file_span_exporter import FileSpanExporter

        # Store the original export method
        original_export = FileSpanExporter.export

        # Create a new export that uses test-specific files
        def patched_export(self, spans):
            """Patched export to use test-specific trace files."""
            # Check if we have a test-specific trace file
            test_file = get_current_trace_file()
            if test_file:
                # Temporarily override the filepath
                original_filepath = self.filepath
                self.filepath = Path(test_file)
                # Ensure directory exists
                self.filepath.parent.mkdir(parents=True, exist_ok=True)

                try:
                    # Call original export with the test-specific file
                    result = original_export(self, spans)
                finally:
                    # Restore original filepath
                    self.filepath = original_filepath
                return result
            else:
                # No test-specific file, use original behavior
                return original_export(self, spans)

        # Replace the export method
        FileSpanExporter.export = patched_export
        logger.info("Successfully patched FileSpanExporter for test isolation")

    except Exception as e:
        logger.warning(f"Failed to patch FileSpanExporter: {e}")


# Apply the patch when this module is imported
patch_file_span_exporter()
