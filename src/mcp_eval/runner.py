"""Enhanced test runner supporting both decorator and dataset approaches."""

import asyncio
import atexit
import importlib.util
import inspect
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from mcp_eval.report_generation.console import generate_failure_message
from mcp_eval.session import TestAgent, TestSession

from mcp_eval.core import (
    TestResult,
    generate_test_id,
)
from mcp_eval.datasets import Dataset
from mcp_eval.report_generation.models import EvaluationReport
from mcp_eval.report_generation import (
    generate_combined_summary,
    generate_combined_markdown_report,
    generate_combined_html_report,
)
from mcp_eval.report_generation.utils import load_config_info
from mcp_eval.report_generation.console import (
    pad,
    print_failure_details,
    print_test_summary_info,
    print_final_summary,
    print_dataset_summary_info,
    print_dataset_final_summary,
    TestProgressDisplay,
)

app = typer.Typer()
console = Console()


# Register an atexit handler to suppress subprocess cleanup warnings
def suppress_cleanup_warnings():
    """Suppress stderr during final cleanup to avoid subprocess warnings."""
    sys.stderr = open(os.devnull, "w")


# Register the suppression to happen at program exit
atexit.register(suppress_cleanup_warnings)


def discover_tests_and_datasets(test_spec: str) -> Dict[str, List]:
    """Discover both decorator-style tests and dataset-style evaluations.

    Args:
        test_spec: Can be a directory, file, or file::function_name
    """
    tasks = []
    datasets = []

    # Parse pytest-style test specifier
    if "::" in test_spec:
        file_path, function_name = test_spec.split("::", 1)
        path = Path(file_path)
        target_function = function_name
    else:
        path = Path(test_spec)
        target_function = None

    # Handle both files and directories
    if path.is_file():
        # Single file case
        py_files = [path] if path.suffix == ".py" else []
    else:
        # Directory case
        py_files = path.rglob("*.py")

    for py_file in py_files:
        if py_file.name.startswith("__"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Discover decorator-style tests
                for name, obj in inspect.getmembers(module):
                    if (
                        callable(obj)
                        and hasattr(obj, "_is_mcpeval_task")
                        and obj._is_mcpeval_task
                    ):
                        # If target_function is specified, only include matching function
                        if target_function is None or name == target_function:
                            # Add file path info to the task function
                            obj._source_file = py_file
                            tasks.append(obj)

                # Discover datasets (only if no specific function is targeted)
                if target_function is None:
                    for name, obj in inspect.getmembers(module):
                        if isinstance(obj, Dataset):
                            # Add source file info to dataset
                            obj._source_file = py_file
                            datasets.append(obj)

        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not load {py_file}: {e}")

    return {"tasks": tasks, "datasets": datasets}


def expand_parametrized_tests(tasks: List[callable]) -> List[Dict[str, Any]]:
    """Expand parametrized tests into individual test cases."""
    expanded = []

    for task_func in tasks:
        # Check for new pytest-style parametrization first
        param_combinations = getattr(task_func, "_mcpeval_param_combinations", None)
        if param_combinations:
            for kwargs in param_combinations:
                expanded.append(
                    {
                        "func": task_func,
                        "kwargs": kwargs,
                        "source_file": getattr(task_func, "_source_file", None),
                    }
                )
            continue
        else:
            expanded.append(
                {
                    "func": task_func,
                    "kwargs": {},
                    "source_file": getattr(task_func, "_source_file", None),
                }
            )
            continue

    return expanded


def group_tests_by_file(
    test_cases: List[Dict[str, Any]],
) -> Dict[Path, List[Dict[str, Any]]]:
    """Group test cases by their source file."""
    grouped = {}
    for test_case in test_cases:
        source_file = test_case.get("source_file")
        if source_file not in grouped:
            grouped[source_file] = []
        grouped[source_file].append(test_case)
    return grouped


async def run_decorator_tests(
    test_cases: List[Dict[str, Any]], verbose: bool
) -> List[TestResult]:
    """Run decorator-style tests grouped by file."""
    results: list[TestResult] = []
    failed_results = []
    captured_logs: Dict[str, str] = {}  # Store captured logs for each test

    # Group tests by file
    grouped_tests = group_tests_by_file(test_cases)
    files_with_counts = {
        file_path: len(tests) for file_path, tests in grouped_tests.items()
    }

    display = TestProgressDisplay(files_with_counts)

    with Live(display.create_display(type="test"), refresh_per_second=10) as live:
        # Process each file's tests
        for source_file, file_test_cases in grouped_tests.items():
            display.set_current_file(source_file)

            for test_case in file_test_cases:
                func = test_case["func"]
                kwargs = test_case["kwargs"]

                # Create test name with parameters
                test_name = func.__name__
                if kwargs:
                    param_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
                    test_name += f"[{param_str}]"

                # Set up log capture if verbose
                log_stream = None
                log_handler = None
                if verbose:
                    log_stream = io.StringIO()
                    log_handler = logging.StreamHandler(log_stream)
                    log_handler.setLevel(logging.INFO)
                    # Add formatter to match pytest style
                    formatter = logging.Formatter(
                        "[%(levelname)s] %(asctime)s %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    log_handler.setFormatter(formatter)
                    # Get root logger and add handler
                    root_logger = logging.getLogger()
                    root_logger.addHandler(log_handler)

                try:
                    # Call task decorated function
                    result: TestResult = await func(**kwargs)

                    # Add session information if verbose mode
                    if verbose:
                        # Extract actual agent and session info from the result and config
                        session_info = {}
                        try:
                            from mcp_eval.config import get_settings

                            settings = get_settings()

                            # Get provider and model (these are global settings)
                            if settings.provider:
                                session_info["provider"] = settings.provider
                            if settings.model:
                                session_info["model"] = settings.model

                            # The actual agent info is already in the result
                            # No need to get it from default_agent_spec
                            # as it may have been overridden

                            # Get LLM Judge configuration if available
                            if hasattr(settings, "judge") and settings.judge:
                                judge_config = {}
                                if hasattr(settings.judge, "provider"):
                                    judge_config["provider"] = settings.judge.provider
                                if hasattr(settings.judge, "model"):
                                    judge_config["model"] = settings.judge.model
                                if hasattr(settings.judge, "temperature"):
                                    judge_config["temperature"] = (
                                        settings.judge.temperature
                                    )
                                if judge_config:
                                    session_info["llm_judge"] = judge_config
                        except Exception as e:
                            console.print(f"  [red]ERROR[/] {test_name}: {e}")
                            pass

                        # Attach session info to result for verbose display
                        if session_info:
                            result._session_info = session_info

                    if result.passed:
                        display.add_result(passed=True)
                    else:
                        display.add_result(passed=False)
                        failure_message = result.error or generate_failure_message(
                            result.evaluation_results
                        )
                        result.error = failure_message
                        # Capture logs if verbose
                        if verbose and log_stream:
                            captured_logs[test_name] = log_stream.getvalue()
                        failed_results.append(result)

                except Exception as e:
                    display.add_result(passed=False, error=True)
                    console.print(f"  [red]ERROR[/] {test_name}: {e}")
                    file_name = Path(source_file).name
                    test_id = generate_test_id(file_name, test_name)
                    result = TestResult(
                        id=test_id,
                        test_name=test_name,
                        description=getattr(func, "_description", ""),
                        server_name=getattr(func, "_server", "unknown"),
                        servers=[],
                        agent_name="",
                        parameters=kwargs,
                        passed=False,
                        evaluation_results=[],
                        metrics=None,
                        duration_ms=0,
                        file=file_name,
                        error=str(e),
                    )
                    # Capture logs if verbose
                    if verbose and log_stream:
                        captured_logs[test_name] = log_stream.getvalue()
                    failed_results.append(result)

                finally:
                    # Clean up log handler
                    if log_handler:
                        root_logger = logging.getLogger()
                        root_logger.removeHandler(log_handler)
                        log_handler.close()

                results.append(result)
                live.update(display.create_display(type="test"))

    # Print detailed failures section if there are any failures
    print_failure_details(
        console,
        failed_results,
        verbose=verbose,
        captured_logs=captured_logs if verbose else None,
    )

    return results


async def run_dataset_evaluations(
    datasets: List[Dataset],
    *,
    max_concurrency: int | None = None,
    verbose: bool = False,
) -> List[EvaluationReport]:
    """Run dataset-style evaluations with live progress display."""
    reports: list[EvaluationReport] = []
    failed_results: list[TestResult] = []
    captured_logs: Dict[str, str] = {}  # Store captured logs for each test

    # Create progress display for all datasets
    dataset_counts = {dataset.name: len(dataset.cases) for dataset in datasets}
    display = TestProgressDisplay(dataset_counts)

    with Live(display.create_display(type="case"), refresh_per_second=10) as live:
        for ds in datasets:

            async def standard_task(inputs, agent: TestAgent, session: TestSession):
                response = await agent.generate_str(inputs)
                return response

            display.set_current_group(ds.name)

            def progress_callback(
                passed: bool,
                error: bool,
            ):
                """Progress callback for dataset evaluation."""

                display.add_result(passed=passed, error=error, group_key=ds.name)

                # Update the live display immediately
                live.update(display.create_display(type="case"))

            report = await ds.evaluate(
                standard_task,
                max_concurrency=max_concurrency,
                progress_callback=progress_callback,
            )

            reports.append(report)

            # Collect failed cases for detailed reporting with proper failure messages
            for result in report.results:
                if not result.passed:
                    # Convert CaseResult to TestResult format for consistency
                    source_file = getattr(dataset, "_source_file", None)
                    file_name = Path(source_file).name if source_file else "unknown"
                    test_id = generate_test_id(file_name, result.case_name)
                    test_result = TestResult(
                        id=test_id,
                        test_name=f"{ds.name}::{result.case_name}",
                        description=f"Dataset case from {ds.name}",
                        server_name=ds.server_name or "unknown",
                        servers=[ds.server_name]
                        if getattr(ds, "server_name", None)
                        else [],
                        agent_name="",
                        parameters={
                            "inputs": result.inputs
                            if hasattr(result, "inputs")
                            else {},
                            "expected_output": result.expected_output
                            if hasattr(result, "expected_output")
                            else None,
                        },
                        passed=result.passed,
                        evaluation_results=result.evaluation_results,
                        metrics=result.metrics,
                        duration_ms=result.duration_ms,
                        file=file_name,
                        error=result.error,
                    )

                    # Generate detailed failure message
                    failure_message = result.error or generate_failure_message(
                        result.evaluation_results
                    )
                    test_result.error = failure_message

                    # Store additional metadata for verbose output
                    if verbose:
                        # Store the actual output for verbose display
                        if hasattr(result, "output"):
                            test_result.actual_output = result.output

                        # Add session information for dataset tests
                        session_info = {}
                        try:
                            from mcp_eval.config import get_settings

                            settings = get_settings()
                            if settings.provider:
                                session_info["provider"] = settings.provider
                            if settings.model:
                                session_info["model"] = settings.model

                            # Get LLM Judge configuration if available
                            if hasattr(settings, "judge") and settings.judge:
                                judge_config = {}
                                if hasattr(settings.judge, "provider"):
                                    judge_config["provider"] = settings.judge.provider
                                if hasattr(settings.judge, "model"):
                                    judge_config["model"] = settings.judge.model
                                if hasattr(settings.judge, "temperature"):
                                    judge_config["temperature"] = (
                                        settings.judge.temperature
                                    )
                                if judge_config:
                                    session_info["llm_judge"] = judge_config
                        except Exception as e:
                            console.print(f"  [red]ERROR[/] {ds.name}: {e}")
                            pass

                        if session_info:
                            test_result._session_info = session_info

                    failed_results.append(test_result)

    # Print detailed failures section if there are any failures
    print_failure_details(
        console,
        failed_results,
        verbose=verbose,
        captured_logs=captured_logs if verbose else None,
    )

    return reports


@app.callback(invoke_without_command=True)
def run_tests(
    ctx: typer.Context,
    test_dir: str = typer.Argument(
        "tests", help="Directory to scan for tests and datasets"
    ),
    format: str = typer.Option("auto", help="Output format (auto, decorator, dataset)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    json_report: str | None = typer.Option(None, "--json", help="Save JSON report"),
    markdown_report: str | None = typer.Option(
        None, "--markdown", help="Save Markdown report"
    ),
    html_report: str | None = typer.Option(None, "--html", help="Save HTML report"),
    max_concurrency: int | None = typer.Option(
        None, "--max-concurrency", help="Maximum concurrent evaluations"
    ),
):
    """Run MCP-Eval tests and datasets."""
    if ctx.invoked_subcommand is None:
        # Collect any extra args (e.g., additional paths from shell globs)
        extra_specs = [arg for arg in (ctx.args or []) if not arg.startswith("-")]
        test_specs = [test_dir] + extra_specs if test_dir else extra_specs

        asyncio.run(
            _run_async_multi(
                test_specs,
                format,
                verbose,
                json_report,
                markdown_report,
                html_report,
                max_concurrency,
            )
        )


async def _run_async(
    test_dir: str,
    format: str,
    verbose: bool,
    json_report: str | None,
    markdown_report: str | None,
    html_report: str | None,
    max_concurrency: int | None,
):
    """Async implementation of the run command."""
    console.print(pad("MCP-Eval", char="*", console=console), style="magenta")
    # Parse pytest-style test specifier for path validation
    if "::" in test_dir:
        file_path, _ = test_dir.split("::", 1)
        test_path = Path(file_path)
    else:
        test_path = Path(test_dir)

    if not test_path.exists():
        console.print(f"[red]Error:[/] Test path '{test_path}' not found")
        raise typer.Exit(1)

    console.print("[blue]Discovering tests and datasets...[/blue]")
    discovered = discover_tests_and_datasets(test_dir)

    tasks = discovered["tasks"]
    datasets = discovered["datasets"]

    if not tasks and not datasets:
        console.print("[yellow]No tests or datasets found[/]")
        return

    console.print(
        f"[blue]Found {len(tasks)} test function(s) and {len(datasets)} dataset(s)[/blue]",
    )

    # Run tests and evaluations
    test_results = []
    dataset_reports = []

    if tasks and format in ["auto", "decorator"]:
        test_cases = expand_parametrized_tests(tasks)
        console.print(
            f"\n[blue]Running {len(test_cases)} decorator-style test cases...[/blue]"
        )

        # Setup/teardown functions are now handled per-file in the task decorator
        test_results = await run_decorator_tests(test_cases, verbose)

    if datasets and format in ["auto", "dataset"]:
        console.print(f"\n[blue]Running {len(datasets)} dataset evaluations...[/blue]")
        dataset_reports = await run_dataset_evaluations(
            datasets, max_concurrency=max_concurrency, verbose=verbose
        )

    # Print short test summary info (pytest-like)
    if test_results:
        failed_tests = [r for r in test_results if not r.passed]
        print_test_summary_info(console, failed_tests)
        print_final_summary(console, test_results)

    # Print dataset summary info (pytest-like)
    if dataset_reports:
        for report in dataset_reports:
            failed_cases = [case for case in report.results if not case.passed]
            if failed_cases:
                print_dataset_summary_info(console, failed_cases, report.dataset_name)

        print_dataset_final_summary(console, dataset_reports)

    # Generate combined summary for all test results
    if test_results or dataset_reports:
        console.print(Text(console.width * "="))
        generate_combined_summary(
            test_results, dataset_reports, console, verbose=verbose
        )

    # Generate reports
    if json_report or markdown_report or html_report:
        combined_report = {
            "decorator_tests": [r.__dict__ for r in test_results],
            "dataset_reports": [r.to_dict() for r in dataset_reports],
            "summary": {
                "total_decorator_tests": len(test_results),
                "passed_decorator_tests": sum(1 for r in test_results if r.passed),
                "total_dataset_cases": sum(r.total_cases for r in dataset_reports),
                "passed_dataset_cases": sum(r.passed_cases for r in dataset_reports),
            },
        }

        if json_report:
            import json

            with open(json_report, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=2, default=str)
            console.print(f"JSON report saved to {json_report}", style="blue")

        # Load config to get output directory for test reports
        config_info = load_config_info()
        output_dir = "./test-reports"  # default
        if config_info and "reporting" in config_info:
            output_dir = config_info["reporting"].get("output_dir", "./test-reports")

        if markdown_report:
            generate_combined_markdown_report(
                combined_report, markdown_report, output_dir=output_dir
            )
            console.print(f"Markdown report saved to {markdown_report}", style="blue")

        if html_report:
            generate_combined_html_report(combined_report, html_report)
            console.print(f"HTML report saved to {html_report}", style="blue")

    # Give subprocess transports time to close properly before exit
    await asyncio.sleep(0.2)

    # Exit with error if any tests failed
    total_failed = sum(1 for r in test_results if not r.passed) + sum(
        r.failed_cases for r in dataset_reports
    )

    if total_failed > 0:
        raise typer.Exit(1)


async def _run_async_multi(
    test_specs: List[str],
    format: str,
    verbose: bool,
    json_report: str | None,
    markdown_report: str | None,
    html_report: str | None,
    max_concurrency: int | None,
):
    """Run tests for multiple specs (dirs/files/file::func), aggregating discovery.

    Skips common junk paths like __pycache__ when provided via shell globs.
    """
    console.print(pad("MCP-Eval", char="*", console=console), style="magenta")

    normalized_specs: List[str] = []
    for spec in test_specs:
        # Handle pytest-style function spec
        if "::" in spec:
            file_path, _ = spec.split("::", 1)
            path = Path(file_path)
        else:
            path = Path(spec)

        if path.name == "__pycache__":
            continue
        if not path.exists():
            console.print(f"[yellow]Skipping missing path:[/] {path}")
            continue
        normalized_specs.append(spec)

    if not normalized_specs:
        console.print("[yellow]No valid test paths provided[/]")
        return

    console.print("[blue]Discovering tests and datasets...[/blue]")
    tasks: list[Any] = []
    datasets: list[Any] = []
    for spec in normalized_specs:
        discovered = discover_tests_and_datasets(spec)
        tasks.extend(discovered["tasks"])
        datasets.extend(discovered["datasets"])

    if not tasks and not datasets:
        console.print("[yellow]No tests or datasets found[/]")
        return

    console.print(
        f"[blue]Found {len(tasks)} test function(s) and {len(datasets)} dataset(s)[/blue]",
    )

    # Run tests and evaluations (reuse existing flow)
    test_results = []
    dataset_reports = []

    if tasks and format in ["auto", "decorator"]:
        test_cases = expand_parametrized_tests(tasks)
        console.print(
            f"\n[blue]Running {len(test_cases)} decorator-style test cases...[/blue]"
        )

        # Setup/teardown functions are now handled per-file in the task decorator
        test_results = await run_decorator_tests(test_cases, verbose)

    if datasets and format in ["auto", "dataset"]:
        console.print(f"\n[blue]Running {len(datasets)} dataset evaluations...[/blue]")
        dataset_reports = await run_dataset_evaluations(
            datasets, max_concurrency=max_concurrency, verbose=verbose
        )

    if test_results:
        failed_tests = [r for r in test_results if not r.passed]
        print_test_summary_info(console, failed_tests)
        print_final_summary(console, test_results)

    if dataset_reports:
        for report in dataset_reports:
            failed_cases = [case for case in report.results if not case.passed]
            if failed_cases:
                print_dataset_summary_info(console, failed_cases, report.dataset_name)

        print_dataset_final_summary(console, dataset_reports)

    if test_results or dataset_reports:
        console.print(Text(console.width * "="))
        generate_combined_summary(
            test_results, dataset_reports, console, verbose=verbose
        )

    if json_report or markdown_report or html_report:
        combined_report = {
            "decorator_tests": [r.__dict__ for r in test_results],
            "dataset_reports": [r.to_dict() for r in dataset_reports],
            "summary": {
                "total_decorator_tests": len(test_results),
                "passed_decorator_tests": sum(1 for r in test_results if r.passed),
                "total_dataset_cases": sum(r.total_cases for r in dataset_reports),
                "passed_dataset_cases": sum(r.passed_cases for r in dataset_reports),
            },
        }

        if json_report:
            import json

            with open(json_report, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=2, default=str)
            console.print(f"JSON report saved to {json_report}", style="blue")

        config_info = load_config_info()
        output_dir = "./test-reports"
        if config_info and "reporting" in config_info:
            output_dir = config_info["reporting"].get("output_dir", "./test-reports")

        if markdown_report:
            generate_combined_markdown_report(
                combined_report, markdown_report, output_dir=output_dir
            )
            console.print(f"Markdown report saved to {markdown_report}", style="blue")

        if html_report:
            generate_combined_html_report(combined_report, html_report)
            console.print(f"HTML report saved to {html_report}", style="blue")

    await asyncio.sleep(0.2)

    total_failed = sum(1 for r in test_results if not r.passed) + sum(
        r.failed_cases for r in dataset_reports
    )
    if total_failed > 0:
        raise typer.Exit(1)


@app.command()
def dataset(
    dataset_file: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option("report", help="Output file prefix"),
):
    """Run evaluation on a specific dataset file."""

    async def _run_dataset():
        try:
            dataset = Dataset.from_file(dataset_file)
            console.print(f"Loaded dataset: {dataset.name}")
            console.print(f"Cases: {len(dataset.cases)}")

            # Use a real agent/session so tool/evaluator checks can pass
            async def standard_task(inputs, agent: TestAgent, session: TestSession):
                response = await agent.generate_str(inputs)
                return response

            report = await dataset.evaluate(standard_task)
            # Print a simple summary to the console
            total = report.total_cases
            passed = report.passed_cases
            failed = report.failed_cases
            success_rate = report.success_rate * 100.0
            console.print(
                f"Results: {passed}/{total} passed ({success_rate:.1f}%), failed: {failed}",
                style="blue",
            )
            for r in report.results:
                status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
                console.print(f"  {status} {r.case_name}")

            # Save reports
            import json

            with open(f"{output}.json", "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            console.print(f"Report saved to {output}.json")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_run_dataset())


if __name__ == "__main__":
    app()
