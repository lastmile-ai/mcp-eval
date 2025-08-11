"""Summary report generation for MCP-Eval."""

from typing import List
from rich.console import Console
from rich.table import Table

from ..core import TestResult
from .models import EvaluationReport
from .base import calculate_overall_stats


def generate_combined_summary(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
) -> None:
    """Generate a combined summary of all results."""
    table = Table(title="Combined Test Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Cases/Tests", justify="right")
    table.add_column("Duration", justify="right")

    # Add decorator test results
    for result in test_results:
        status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
        duration = f"{result.duration_ms:.1f}ms" if result.duration_ms else "N/A"
        table.add_row("Test", result.test_name, status, "1", duration)

    # Add dataset results
    for report in dataset_reports:
        status = f"[green]{report.passed_cases}/{report.total_cases}[/]"
        duration = f"{report.average_duration_ms:.1f}ms"
        table.add_row(
            "Dataset", report.dataset_name, status, str(report.total_cases), duration
        )

    console.print(table)

    # Overall summary
    stats = calculate_overall_stats(test_results, dataset_reports)

    console.print("\n[bold]Overall Summary:[/]")
    console.print(
        f"  Decorator Tests: {stats['passed_decorator_tests']}/{stats['total_decorator_tests']} passed"
    )
    console.print(
        f"  Dataset Cases: {stats['passed_dataset_cases']}/{stats['total_dataset_cases']} passed"
    )
    console.print(
        f"  [bold]Total: {stats['total_passed']}/{stats['total_tests']} passed ({stats['overall_success_rate']:.1f}%)[/]"
    )
