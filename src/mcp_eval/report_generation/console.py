"""Console output formatting for mcp-eval test results."""

from typing import List, Dict, Literal
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner
from rich.columns import Columns
from rich.console import Group

from mcp_eval.evaluators.shared import EvaluationRecord

from ..core import TestResult
from .models import EvaluationReport, CaseResult


def pad(
    text: str, char: str = "=", console: Console = None, length: int = None
) -> Text:
    """Add padding to text for console output."""
    # Use provided length, or console width if available, otherwise default to 80
    if length is None:
        if console is not None:
            length = console.width
        else:
            length = 80

    # Calculate padding, accounting for spaces around the text
    text_with_spaces = f" {text} "
    total_padding = length - len(text_with_spaces)

    if total_padding < 0:
        # Text is too long, just return it with minimal padding
        return Text(text_with_spaces)

    left_padding = total_padding // 2
    right_padding = total_padding - left_padding  # Handle odd numbers

    padded_text = Text()
    padded_text.append(char * left_padding)
    padded_text.append(text_with_spaces)
    padded_text.append(char * right_padding)
    return padded_text


def print_failure_details(console: Console, failed_results: List[TestResult]) -> None:
    """Print detailed failure information."""
    if not failed_results:
        return

    console.print(pad("FAILURES", console=console))
    for result in failed_results:
        # Extract function name for header
        func_name = result.test_name.split("[")[0]  # Remove parameters
        console.print(pad(func_name, "_", console=console), style="red bold")
        console.print("")
        if result.error:
            console.print(Text(result.error), style="red")
        console.print("")


def print_test_summary_info(console: Console, failed_results: List[TestResult]) -> None:
    """Print pytest-style test summary info."""
    if not failed_results:
        return

    console.print(pad("short test summary info", console=console), style="blue")
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

    console.print(pad("short dataset summary info", console=console), style="blue")
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


def generate_failure_message(eval_records: list[EvaluationRecord]) -> str | None:
    """Generate failure messages for mcp-eval evaluations"""
    failure_details = []

    failed_eval_records = [r for r in eval_records if not r.passed]

    for eval_record in failed_eval_records:
        name = eval_record.name
        error = eval_record.error
        evaluation_result = eval_record.result

        if error:
            failure_details.append(f"  ✗ {name}: {error}")
        else:
            # Extract expected vs actual information from detailed results
            expected = evaluation_result.expected
            actual = evaluation_result.actual
            score = evaluation_result.score

            detail_parts = []
            if expected is not None:
                if actual is not None:
                    detail_parts.append(f"expected {expected}, got {actual!r}")
                else:
                    detail_parts.append(f"expected {expected}")
            elif actual is not None:
                detail_parts.append(f"got {expected!r}")
            elif score is not None:
                detail_parts.append(f"score {score}")

            if detail_parts:
                failure_details.append(f"  ✗ {name}: {', '.join(detail_parts)}")
            else:
                failure_details.append(f"  ✗ {name}: {evaluation_result.model_dump()}")

    failure_message = (
        "Evaluation failures:\n" + "\n".join(failure_details)
        if len(failure_details)
        else None
    )

    return failure_message
