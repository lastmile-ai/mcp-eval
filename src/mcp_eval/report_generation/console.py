"""Console output formatting for mcp-eval test results."""

from typing import List, Dict, Literal, Optional
from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner
from rich.columns import Columns
from rich.console import Group

from mcp_eval.core import TestResult
from mcp_eval.evaluators.shared import EvaluationRecord
from mcp_eval.report_generation.models import EvaluationReport, CaseResult
from mcp_eval.utils import get_test_artifact_paths


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


def print_failure_details(
    console: Console,
    failed_results: List[TestResult],
    verbose: bool = False,
    captured_logs: Optional[Dict[str, str]] = None,
) -> None:
    """Print detailed failure information.

    Args:
        console: Rich console for output
        failed_results: List of failed test results
        verbose: If True, show detailed trace information
        captured_logs: Optional dict mapping test names to captured log output
    """
    if not failed_results:
        return

    console.print(pad("FAILURES", console=console))
    for result in failed_results:
        # Extract function name for header
        func_name = result.test_name.split("[")[0]  # Remove parameters
        console.print(pad(func_name, "_", console=console), style="red bold")

        if verbose:
            # Show detailed information in verbose mode
            log_output = captured_logs.get(result.test_name) if captured_logs else None
            _print_verbose_failure_details(console, result, log_output)
        else:
            # Standard failure output
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


def _print_verbose_failure_details(
    console: Console, result: TestResult, log_output: Optional[str] = None
) -> None:
    """Print verbose failure details including trace information.

    Args:
        console: Rich console for output
        result: Failed test result
        log_output: Optional captured log output for this test
    """
    console.print("")

    # Print test configuration and context
    console.print("[cyan]Test Configuration:[/cyan]")
    console.print(f"  Test Name: {result.test_name}")
    if result.description:
        console.print(f"  Description: {result.description}")

    # Print agent information (from the actual agent used in the test)
    if result.agent_name:
        console.print(f"  Agent: {result.agent_name}")
    if result.servers:
        console.print(f"  MCP Servers: {', '.join(result.servers)}")
    elif result.server_name and result.server_name != "unknown":
        console.print(f"  MCP Server: {result.server_name}")

    # Print agent instruction if available (show more in verbose mode)
    if hasattr(result, "_agent_details") and "instruction" in result._agent_details:
        instruction = result._agent_details["instruction"]
        # In verbose mode, show up to 1000 characters of instructions
        max_length = 1000
        if len(instruction) > max_length:
            console.print(f"  Instructions: {instruction[:max_length]}...")
            console.print(
                f"  [dim](truncated, {len(instruction)} total characters)[/dim]"
            )
        else:
            console.print(f"  Instructions: {instruction}")

    # Print additional session info if available
    if hasattr(result, "_session_info"):
        session_info = result._session_info
        if "model" in session_info:
            console.print(f"  Model: {session_info['model']}")
        if "provider" in session_info:
            console.print(f"  Provider: {session_info['provider']}")

    # Check if any evaluator used LLM Judge and extract its config
    judge_configs = []
    if result.evaluation_results:
        for eval_record in result.evaluation_results:
            if eval_record.result and hasattr(eval_record.result, "details"):
                details = eval_record.result.details
                if isinstance(details, dict) and "judge_config" in details:
                    judge_config = details["judge_config"]
                    if judge_config and judge_config not in judge_configs:
                        judge_configs.append(judge_config)

    # Print LLM Judge configuration if any was used
    if judge_configs:
        console.print("\n[cyan]LLM Judge Configuration (actual):[/cyan]")
        for config in judge_configs:
            if config:
                console.print(f"  Provider: {config.get('provider', 'Not set')}")
                console.print(f"  Model: {config.get('model', 'Not set')}")
                break  # Show first valid config
    elif hasattr(result, "_session_info") and "llm_judge" in result._session_info:
        # Fallback to session info if no actual config found
        judge = result._session_info["llm_judge"]
        console.print("\n[cyan]LLM Judge Configuration (settings):[/cyan]")
        console.print(f"  Provider: {judge.get('provider', 'Not set')}")
        console.print(f"  Model: {judge.get('model', 'Not set')}")

    # Print basic failure message
    console.print("\n[red]Evaluation failures:[/red]")
    if result.error:
        # Parse and format the error message for better readability
        if "Evaluation failures:" in result.error:
            lines = result.error.split("\n")
            for line in lines:
                if line.strip().startswith("✗"):
                    console.print(f"  {line.strip()}", style="red")
                elif line.strip() and not line.startswith("Evaluation failures:"):
                    console.print(f"  {line.strip()}", style="yellow")
        else:
            console.print(Text(result.error), style="red")

    # Print actual output if available (for dataset cases)
    if hasattr(result, "actual_output") and result.actual_output:
        console.print("\n[cyan]Actual Output:[/cyan]")
        # Truncate very long outputs
        output = str(result.actual_output)
        if len(output) > 500:
            output = output[:500] + "... (truncated)"
        console.print(f"  {output}", style="dim")

    # Print test parameters
    if result.parameters:
        console.print("\n[cyan]Test Parameters:[/cyan]")
        for key, value in result.parameters.items():
            # Format parameters nicely
            if isinstance(value, dict):
                console.print(f"  {key}:")
                for k, v in value.items():
                    console.print(f"    {k}: {v}")
            elif isinstance(value, list) and len(value) > 3:
                console.print(
                    f"  {key}: [{', '.join(str(v) for v in value[:3])}, ... ({len(value)} items)]"
                )
            else:
                console.print(f"  {key}: {value}")

    # Print detailed evaluation results
    if result.evaluation_results:
        console.print("\n[cyan]Evaluation Details:[/cyan]")
        for eval_record in result.evaluation_results:
            if not eval_record.passed:
                console.print(f"  [red]✗ {eval_record.name}[/red]")
                if eval_record.result:
                    if eval_record.result.expected is not None:
                        console.print(f"    Expected: {eval_record.result.expected}")
                    if eval_record.result.actual is not None:
                        # Truncate very long actual values (like LLM Judge responses)
                        actual_str = str(eval_record.result.actual)
                        if len(actual_str) > 200:
                            actual_str = actual_str[:200] + "..."
                        console.print(f"    Actual: {actual_str}")
                    if eval_record.result.score is not None:
                        console.print(f"    Score: {eval_record.result.score}")
                    if (
                        hasattr(eval_record.result, "message")
                        and eval_record.result.message
                    ):
                        console.print(f"    Message: {eval_record.result.message}")
                if eval_record.error:
                    console.print(f"    Error: {eval_record.error}", style="red")

    # Print metrics if available
    if result.metrics:
        console.print("\n[cyan]Test Metrics:[/cyan]")
        if isinstance(result.metrics, dict):
            # Duration metrics
            if "total_duration_ms" in result.metrics:
                console.print(
                    f"  Total Duration: {result.metrics['total_duration_ms']:.2f}ms"
                )
            elif "duration_ms" in result.metrics:
                console.print(f"  Duration: {result.metrics['duration_ms']:.2f}ms")

            # Iteration count
            if "iteration_count" in result.metrics:
                console.print(f"  Iterations: {result.metrics['iteration_count']}")

            # Tool metrics
            if "tool_calls" in result.metrics:
                tool_calls = result.metrics["tool_calls"]
                console.print(f"  Tool Calls: {len(tool_calls)}")
                if tool_calls:
                    # Show tool call details (we're already in verbose mode)
                    tool_names = {}
                    for call in tool_calls[:10]:  # Show first 10
                        if isinstance(call, dict):
                            # Handle both 'tool_name' and 'name' keys
                            name = call.get("tool_name") or call.get("name")
                            if name:
                                tool_names[name] = tool_names.get(name, 0) + 1
                    if tool_names:
                        console.print("    Tools used:")
                        for name, count in tool_names.items():
                            console.print(f"      - {name}: {count}x")

            # LLM metrics
            if "llm_metrics" in result.metrics:
                llm = result.metrics["llm_metrics"]
                if isinstance(llm, dict):
                    console.print("  LLM Usage:")
                    if "model_name" in llm:
                        console.print(f"    Model: {llm['model_name']}")
                    if "input_tokens" in llm:
                        console.print(f"    Input tokens: {llm['input_tokens']}")
                    if "output_tokens" in llm:
                        console.print(f"    Output tokens: {llm['output_tokens']}")
                    if "total_tokens" in llm:
                        console.print(f"    Total tokens: {llm['total_tokens']}")
                    if "cost_estimate" in llm:
                        console.print(f"    Cost estimate: ${llm['cost_estimate']:.6f}")
                    if "latency_ms" in llm:
                        console.print(f"    LLM latency: {llm['latency_ms']:.2f}ms")

            # Error metrics
            if "error_count" in result.metrics and result.metrics["error_count"] > 0:
                console.print(f"  Errors: {result.metrics['error_count']}")
            if "success_rate" in result.metrics:
                console.print(f"  Success Rate: {result.metrics['success_rate']:.1%}")
        else:
            # If metrics is not a dict, it might be a TestMetrics object
            if hasattr(result.metrics, "total_duration_ms"):
                console.print(f"  Duration: {result.metrics.total_duration_ms:.2f}ms")
            if hasattr(result.metrics, "tool_calls"):
                console.print(f"  Tool calls: {len(result.metrics.tool_calls)}")
            if hasattr(result.metrics, "llm_metrics"):
                llm = result.metrics.llm_metrics
                if hasattr(llm, "total_tokens"):
                    console.print(f"  LLM tokens: {llm.total_tokens}")
                if hasattr(llm, "cost_estimate"):
                    console.print(f"  LLM cost: ${llm.cost_estimate:.4f}")

    # Print captured logs if available
    if log_output and log_output.strip():
        console.print("\n[cyan]Captured Log Output:[/cyan]")
        # Format log output similar to pytest
        console.print(
            pad(" Captured stdout call ", "-", console=console, length=80), style="dim"
        )
        for line in log_output.strip().split("\n"):
            console.print(line)

    # Print links to trace and metrics files
    _print_artifact_links(console, result.test_name)

    # Print duration
    console.print(f"\n[dim]Test duration: {result.duration_ms:.2f}ms[/dim]")
    console.print("")


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


def _print_artifact_links(console: Console, test_name: str) -> None:
    """Print links to test artifact files if they exist.

    Args:
        console: Rich console for output
        test_name: Name of the test
    """
    # Get standardized artifact paths using shared utility
    trace_file, metrics_file = get_test_artifact_paths(test_name)

    files_found = []
    if trace_file.exists():
        files_found.append(
            ("Trace (JSONL)", trace_file, "View detailed execution trace")
        )
    if metrics_file.exists():
        files_found.append(
            ("Metrics (JSON)", metrics_file, "View test metrics and results")
        )

    if files_found:
        console.print("\n[cyan]Test Artifacts:[/cyan]")
        for file_type, file_path, description in files_found:
            # Get relative path if possible, otherwise absolute
            try:
                display_path = file_path.relative_to(Path.cwd())
            except ValueError:
                display_path = file_path

            # Format the output with description
            console.print(f"  • {file_type}: [blue]{display_path}[/blue]")
            console.print(f"    [dim]{description}[/dim]")
            console.print(f"    [dim]Path: {file_path.absolute()}[/dim]")
    else:
        # Only show in verbose mode that artifacts weren't found
        console.print(
            "\n[dim]Test artifacts not found - they may not have been saved yet[/dim]"
        )
