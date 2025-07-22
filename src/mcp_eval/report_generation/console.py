"""Console output formatting for mcp-eval test results."""

from typing import List, Dict, Any
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.panel import Panel

from ..core import TestResult


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


class TestProgressDisplay:
    """Live display for test progress with pytest-style dots per file."""

    def __init__(self, files_with_counts: Dict[str, int]):
        self.files_with_counts = files_with_counts
        self.total_tests = sum(files_with_counts.values())
        self.completed = 0
        self.file_results = {
            file_path: Text() for file_path in files_with_counts.keys()
        }
        self.current_file = None

    def create_display(self):
        progress_text = f"Running tests... {self.completed}/{self.total_tests}"

        # Build display with file results
        display_parts = [(progress_text, "blue"), "\n\n"]

        for file_path, dots in self.file_results.items():
            file_name = file_path.name if hasattr(file_path, "name") else str(file_path)
            display_parts.extend([(file_name, "cyan"), " ", dots, "\n"])

        return Panel(
            Text.assemble(*display_parts),
            width=80,
            title="Test Progress",
            border_style="blue",
        )

    def set_current_file(self, file_path):
        """Set the current file being processed."""
        self.current_file = file_path

    def add_result(self, passed: bool, error: bool = False):
        """Add a test result to the current file and update display."""
        self.completed += 1
        if self.current_file and self.current_file in self.file_results:
            dots = self.file_results[self.current_file]
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
