"""Console output formatting for mcp-eval test results."""

from typing import List, Dict, Any, Literal
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.columns import Columns
from rich.console import Group

from ..core import TestResult
from ..reports import EvaluationReport, CaseResult


def pad(text: str, char: str = "=", length=80) -> Text:
    """Add padding to text for console output."""
    padding = (length - len(text)) // 2
    padded_text = Text()
    padded_text.append(f"{char * padding} ")
    padded_text.append(text)
    padded_text.append(f" {char * padding} ")
    return padded_text


def print_failure_details(console: Console, failed_results: List[TestResult]) -> None:
    """Print detailed failure information."""
    if not failed_results:
        return

    console.print(pad("FAILURES"))
    for result in failed_results:
        # Extract function name for header
        func_name = result.test_name.split("[")[0]  # Remove parameters
        console.print(pad(func_name, "_"), style="red bold")
        console.print("")
        if result.error:
            console.print(Text(result.error), style="red")
        console.print("")


def print_test_summary_info(console: Console, failed_results: List[TestResult]) -> None:
    """Print pytest-style test summary info."""
    if not failed_results:
        return

    console.print(pad("short test summary info"), style="blue")
    for result in failed_results:
        func_name = result.test_name.split("[")[0]
        # Extract file name from module or use test_name as fallback
        file_name = result.test_name.replace(".", "/") + ".py"
        console.print(f"[red]FAILED[/red] {file_name}::{func_name}")


def print_final_summary(console: Console, test_results: List[TestResult]) -> None:
    """Print final test summary with pass/fail counts and duration."""
    failed_results = [r for r in test_results if not r.passed]
    passed_count = len([r for r in test_results if r.passed])
    failed_count = len(failed_results)
    total_time = sum(r.duration_ms for r in test_results) / 1000.0  # Convert to seconds

    summary_text = Text()
    if failed_count > 0:
        summary_text.append(f"{failed_count} failed", style="red bold")
        if passed_count > 0:
            summary_text.append(",")
            summary_text.append(f" {passed_count} passed", style="green bold")
    else:
        summary_text.append(f"{passed_count} passed", style="green bold")

    summary_text.append(f" in {total_time:.2f}s", style=None)
    console.print(summary_text)


def print_dataset_summary_info(
    console: Console, failed_cases: List[CaseResult], dataset_name: str
) -> None:
    """Print pytest-style dataset summary info for failed cases."""
    if not failed_cases:
        return

    console.print(pad("short dataset summary info"), style="blue")
    for case in failed_cases:
        console.print(f"[red]FAILED[/red] {dataset_name}::{case.case_name}")


def print_dataset_final_summary(
    console: Console, dataset_reports: List[EvaluationReport]
) -> None:
    """Print final dataset summary with pass/fail counts and duration."""
    if not dataset_reports:
        return

    total_cases = sum(report.total_cases for report in dataset_reports)
    passed_cases = sum(report.passed_cases for report in dataset_reports)
    failed_cases = total_cases - passed_cases

    # Calculate total duration from all cases across all reports
    total_time = 0.0
    for report in dataset_reports:
        total_time += sum(case.duration_ms for case in report.results)
    total_time = total_time / 1000.0  # Convert to seconds

    summary_text = Text()
    if failed_cases > 0:
        summary_text.append(f"{failed_cases} failed", style="red bold")
        if passed_cases > 0:
            summary_text.append(",")
            summary_text.append(f" {passed_cases} passed", style="green bold")
    else:
        summary_text.append(f"{passed_cases} passed", style="green bold")

    summary_text.append(f" in {total_time:.2f}s", style=None)
    console.print(summary_text)


class TestProgressDisplay:
    """Live display for test progress with pytest-style dots per group (file/dataset)."""

    def __init__(self, groups_with_counts: Dict[str, int]):
        self.groups_with_counts = groups_with_counts
        self.total_tests = sum(groups_with_counts.values())
        self.completed = 0
        self.group_results = {
            group_key: Text() for group_key in groups_with_counts.keys()
        }
        self.current_group = None

    def create_display(self, type: Literal["case", "test"] = "test"):
        # Create spinner and progress text as columns
        spinner_and_text = Columns(
            [
                Spinner("dots", style="blue"),
                Text(
                    f"Running {type}s... {self.completed}/{self.total_tests}",
                    style="blue",
                ),
            ],
            padding=(0, 1),
        )

        # Add group results as separate text objects
        group_displays = []
        for group_key, dots in self.group_results.items():
            group_name = (
                group_key.name if hasattr(group_key, "name") else str(group_key)
            )
            group_displays.append(Text.assemble((group_name, "cyan"), " ", dots))

        # Combine everything using Group
        all_content = [spinner_and_text, Text("")]  # Empty text for spacing
        all_content.extend(group_displays)

        return Panel(
            Group(*all_content),
            width=80,
            title="Progress",
            border_style="blue",
        )

    def set_current_group(self, group_key):
        """Set the current group being processed."""
        self.current_group = group_key

    def set_current_file(self, file_path):
        """Set the current file being processed (backward compatibility)."""
        self.set_current_group(file_path)

    def add_result(self, passed: bool, error: bool = False, group_key=None):
        """Add a test result to the current or specified group and update display."""
        self.completed += 1
        target_group = group_key or self.current_group
        if target_group and target_group in self.group_results:
            dots = self.group_results[target_group]
            if error:
                dots.append("E", style="yellow")
            elif passed:
                dots.append(".", style="green")
            else:
                dots.append("F", style="red")


def run_tests_with_live_display(test_cases: List[Dict[str, Any]], test_runner_func):
    """Run tests with live progress display showing pytest-style dots."""
    display = TestProgressDisplay(len(test_cases))

    with Live(display.create_display(), refresh_per_second=10) as live:
        results, failed_results = test_runner_func(test_cases, display, live)

    return results, failed_results
