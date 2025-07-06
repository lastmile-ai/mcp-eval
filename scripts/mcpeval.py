# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich",
#     "typer",
# ]
# ///

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

app = typer.Typer()
console = Console()


def create_test_results_table():
    """Create the main test results table"""
    table = Table(
        title="TEST RESULTS",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold white",
    )

    # Add columns
    table.add_column("Status", style="bold", width=8)
    table.add_column("Test", width=40)
    table.add_column("Turns", justify="center", width=7)
    table.add_column("Tool Calls", justify="center", width=12)
    table.add_column("Cost ($)", justify="center", width=10)
    table.add_column("Latency (ms)", justify="center", width=14)

    # Add test rows
    table.add_row(
        "[bold green]PASS[/]", "test_get_time_in_london", "1", "1", "0.00016", "2105.73"
    )

    table.add_row(
        "[bold red]FAIL[/]",
        "test_summarization_quality_fails",
        "2",
        "1",
        "0.00043",
        "4821.15",
    )

    # Add empty row
    table.add_row("", "", "", "", "", "")

    # Create failure panel for first test
    failure_text_1 = Text()
    failure_text_1.append("FAILURES", style="bold red")
    failure_text_1.append("\nAssertion: ", style="")
    failure_text_1.append(
        "llm_judge('The summary must be coherent...', min_score=0.8)", style="red"
    )
    failure_text_1.append("\nStatus: ", style="")
    failure_text_1.append("FAIL", style="red")
    failure_text_1.append("\nError: ", style="")
    failure_text_1.append(
        "Judge score 0.2 is below the minimum of 0.8. Reasoning: The summary is not",
        style="italic red",
    )
    failure_text_1.append(
        "\ncoherent. It is an abrupt truncation of the original sentence and does not form a",
        style="italic red",
    )
    failure_text_1.append("\ncomplete thought. It ends with '...'.", style="italic red")

    failure_panel_1 = Panel(failure_text_1, box=box.ROUNDED, border_style="red")

    table.add_row("", failure_panel_1, "", "", "", "")

    table.add_row(
        "[bold green]PASS[/]", "test_chained_tool_use", "1", "2", "0.00038", "3910.44"
    )

    table.add_row(
        "[bold red]FAIL[/]",
        "test_invalid_timezone_error_handling",
        "1",
        "1",
        "0.00018",
        "2344.91",
    )

    # Add empty row
    table.add_row("", "", "", "", "", "")

    # Create failure panel for second test
    failure_text_2 = Text()
    failure_text_2.append("FAILURES", style="bold red")
    failure_text_2.append("\nAssertion: ", style="")
    failure_text_2.append(
        "objective_succeeded('What time is it in the made-u...')", style="red"
    )
    failure_text_2.append("\nStatus: ", style="")
    failure_text_2.append("FAIL", style="red")
    failure_text_2.append("\nError: ", style="")
    failure_text_2.append(
        "Objective was not met. Reasoning: The agent correctly identified that",
        style="italic red",
    )
    failure_text_2.append(
        "\n'Atlantis' is not a real timezone and reported the error from the tool, but it",
        style="italic red",
    )
    failure_text_2.append(
        "\nwas unable to fulfill the user's core objective of getting a time.",
        style="italic red",
    )

    failure_panel_2 = Panel(failure_text_2, box=box.ROUNDED, border_style="red")

    table.add_row("", failure_panel_2, "", "", "", "")

    return table


def create_tool_coverage_tables():
    """Create the tool coverage tables"""
    # Main coverage table
    coverage_table = Table(
        title="Tool Coverage for Server: 'sample_server'",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold white",
    )
    coverage_table.add_column("Metric", width=36)
    coverage_table.add_column("Value", width=14)

    coverage_table.add_row("Coverage", "100.00%")
    coverage_table.add_row("Total Tools", "2")
    coverage_table.add_row("Called Tools", "2")

    # Tool statistics table
    stats_table = Table(
        title="Called Tool Statistics for 'sample_server'",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold white",
    )
    stats_table.add_column("Tool Name", width=36)
    stats_table.add_column("Call Count", width=12)

    stats_table.add_row("get_current_time", "2")
    stats_table.add_row("summarize_text", "2")

    return coverage_table, stats_table


@app.command()
def run_mock_output(server: str = "sample_server", tests: int = 4):
    """Display mock CLI eval tool output"""

    # Mock the initial output lines
    console.print("Reading inline script metadata from 'scripts/mcpeval.py'")
    console.print()
    console.print(f"Found [magenta]{tests}[/] [cyan]test(s)[/]. Running...")
    console.print()
    console.print(f"Running tests for server: '[green]{server}[/]'")
    console.print()

    # Print separator
    console.print("=" * 119, style="bold")

    # Create and display test results table
    test_table = create_test_results_table()
    console.print(test_table)
    console.print()

    # Print separator for tool coverage
    console.print("=" * 115, style="bold")

    # Create and display tool coverage tables
    coverage_table, stats_table = create_tool_coverage_tables()
    console.print(coverage_table)
    console.print()
    console.print(stats_table)
    console.print("\n")


@app.command()
def demo():
    """Run the demo output exactly as shown in your example"""
    run_mock_output()


if __name__ == "__main__":
    app()
