"""Report generation module for MCP-Eval.

This module provides functionality for generating various types of reports
from test results and evaluation data.
"""

from mcp_eval.report_generation.models import EvaluationReport, CaseResult
from mcp_eval.report_generation.console import generate_failure_message
from mcp_eval.report_generation.summary import generate_combined_summary
from mcp_eval.report_generation.markdown import generate_combined_markdown_report
from mcp_eval.report_generation.html import (
    generate_combined_html_report,
    format_config_for_html,
)

__all__ = [
    # Data models
    "EvaluationReport",
    "CaseResult",
    # Console utilities
    "generate_failure_message",
    # Report generators
    "generate_combined_summary",
    "generate_combined_markdown_report",
    "generate_combined_html_report",
    "format_config_for_html",
]
