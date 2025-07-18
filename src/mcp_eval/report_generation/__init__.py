"""Report generation module for MCP-Eval.

This module provides functionality for generating various types of reports
from test results and evaluation data.
"""

from .summary import generate_combined_summary
from .markdown import generate_combined_markdown_report
from .html import generate_combined_html_report, format_config_for_html

__all__ = [
    "generate_combined_summary",
    "generate_combined_markdown_report", 
    "generate_combined_html_report",
    "format_config_for_html",
]