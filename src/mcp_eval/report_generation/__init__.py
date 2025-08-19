"""Report generation module for MCP-Eval.

This module provides functionality for generating various types of reports
from test results and evaluation data.
"""

from .models import EvaluationReport, CaseResult
from .console import generate_failure_message
from .summary import generate_combined_summary
from .markdown import generate_combined_markdown_report
from .html import generate_combined_html_report

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
]
