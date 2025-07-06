import json
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from mcp_eval.core import TestResult
import dataclasses


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def generate_console_report(results: List[TestResult], console: Console):
    """Generates a summary report to the console."""
    summary_table = Table(title="MCP-Eval Test Summary")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Description", style="magenta")
    summary_table.add_column("Latency (ms)", justify="right", style="green")

    for result in results:
        status = "[bold green]PASS[/]" if result.passed else "[bold red]FAIL[/]"
        summary_table.add_row(
            status,
            result.test_name,
            result.description,
            f"{result.metrics.latency_ms:.2f}",
        )
        if not result.passed:
            assertion_table = Table(show_header=True, header_style="bold red", box=None)
            assertion_table.add_column("Assertion")
            assertion_table.add_column("Status")
            assertion_table.add_column("Error")

            has_failed_assertions = False
            for assertion in result.assertions:
                if not assertion.passed:
                    has_failed_assertions = True
                    assertion_table.add_row(
                        f"[red]{assertion.name}[/red]",
                        "[red]FAIL[/red]",
                        f"[italic red]{assertion.error}[/italic]",
                    )

            if result.error:
                summary_table.add_row(
                    "", "[red]FATAL[/]", f"[italic red]{result.error}[/]", ""
                )

            if has_failed_assertions:
                summary_table.add_row("", assertion_table, "", "")

    console.print(summary_table)


def generate_coverage_report(coverage_reports: Dict[str, Any], console: Console):
    """Generates a tool coverage report to the console."""
    console.print("\n" + "=" * 20 + " TOOL COVERAGE " + "=" * 20)

    for server_name, report in coverage_reports.items():
        table = Table(title=f"Tool Coverage for Server: '{server_name}'")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Coverage", f"{report['coverage']:.2f}%")
        table.add_row("Total Tools", str(report["total_tools"]))
        table.add_row("Called Tools", str(report["called_tools"]))

        console.print(table)

        if report["uncalled_tools"]:
            uncalled_table = Table(
                title=f"Uncalled Tools for '{server_name}'", box=None
            )
            uncalled_table.add_column("Tool Name", style="yellow")
            for tool in sorted(report["uncalled_tools"]):
                uncalled_table.add_row(tool)
            console.print(uncalled_table)


def generate_json_report(results: List[TestResult], output_path: str):
    """Generates a detailed JSON report."""
    report_data = [result for result in results]
    with open(output_path, "w") as f:
        json.dump(report_data, f, cls=EnhancedJSONEncoder, indent=2)
