"""Enhanced test runner supporting both decorator and dataset approaches."""

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress

from mcp_eval.reporting import generate_failure_message

from .core import TestResult, _setup_functions, _teardown_functions
from .datasets import Dataset
from .reports import EvaluationReport
from .report_generation import (
    generate_combined_summary,
    generate_combined_markdown_report,
    generate_combined_html_report,
)

app = typer.Typer()
console = Console()


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
                            tasks.append(obj)

                # Discover datasets (only if no specific function is targeted)
                if target_function is None:
                    for name, obj in inspect.getmembers(module):
                        if isinstance(obj, Dataset):
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
                expanded.append({"func": task_func, "kwargs": kwargs})
            continue
        else:
            expanded.append({"func": task_func, "kwargs": {}})
            continue

    return expanded


async def run_decorator_tests(
    test_cases: List[Dict[str, Any]], verbose: bool
) -> List[TestResult]:
    """Run decorator-style tests."""
    results: list[TestResult] = []

    with Progress() as progress:
        task_id = progress.add_task("[cyan]Running tests...", total=len(test_cases))

        for test_case in test_cases:
            func = test_case["func"]
            kwargs = test_case["kwargs"]

            # Create test name with parameters
            test_name = func.__name__
            if kwargs:
                param_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
                test_name += f"[{param_str}]"

            try:
                # Call task decorated function
                result: TestResult = await func(**kwargs)

                results.append(result)

                status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
                console.print(f"  {status} {test_name}")

                if not result.passed:
                    failure_message = generate_failure_message(
                        result.evaluation_results
                    )
                    console.print(f"[red]ERROR[/] {failure_message}")

            except Exception as e:
                console.print(f"  [red]ERROR[/] {test_name}: {e}")
                result = TestResult(
                    test_name=test_name,
                    description=getattr(func, "_description", ""),
                    server_name=getattr(func, "_server", "unknown"),
                    parameters=kwargs,
                    passed=False,
                    evaluation_results=[],
                    metrics=None,
                    duration_ms=0,
                    error=str(e),
                )
                results.append(result)

            progress.update(task_id, advance=1)

    return results


async def run_dataset_evaluations(datasets: List[Dataset]) -> List[EvaluationReport]:
    """Run dataset-style evaluations."""
    reports = []

    for dataset in datasets:
        console.print(f"\n[blue]Evaluating dataset: {dataset.name}[/blue]")

        # Find the task function in the same module
        # This is a simplified approach - in practice, would be more sophisticated
        async def mock_task(inputs):
            return f"Mock response for: {inputs}"

        try:
            report = await dataset.evaluate(mock_task)
            reports.append(report)

            console.print(
                f"[green]Completed:[/] {report.passed_cases}/{report.total_cases} cases passed"
            )

            # Print brief summary
            report.print(
                include_input=False, include_output=False, include_durations=True
            )

        except Exception as e:
            console.print(f"[red]Error evaluating dataset {dataset.name}: {e}[/red]")

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
        asyncio.run(
            _run_async(
                test_dir,
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
    # Parse pytest-style test specifier for path validation
    if "::" in test_dir:
        file_path, _ = test_dir.split("::", 1)
        test_path = Path(file_path)
    else:
        test_path = Path(test_dir)

    if not test_path.exists():
        console.print(f"[red]Error:[/] Test path '{test_path}' not found")
        raise typer.Exit(1)

    console.print(f"[blue]Discovering tests and datasets in {test_dir}...[/blue]")
    discovered = discover_tests_and_datasets(test_dir)

    tasks = discovered["tasks"]
    datasets = discovered["datasets"]

    if not tasks and not datasets:
        console.print("[yellow]No tests or datasets found[/]")
        return

    console.print(f"Found {len(tasks)} test function(s) and {len(datasets)} dataset(s)")

    # Run tests and evaluations
    test_results = []
    dataset_reports = []

    if tasks and format in ["auto", "decorator"]:
        console.print(f"\n[blue]Running {len(tasks)} decorator-style tests...[/blue]")

        # Execute setup functions before running tests
        for setup_func in _setup_functions:
            try:
                setup_func()
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Setup function {setup_func.__name__} failed: {e}[/]"
                )

        test_cases = expand_parametrized_tests(tasks)
        test_results = await run_decorator_tests(test_cases, verbose)

        # Execute teardown functions after running tests
        for teardown_func in _teardown_functions:
            try:
                teardown_func()
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Teardown function {teardown_func.__name__} failed: {e}[/]"
                )

    if datasets and format in ["auto", "dataset"]:
        console.print(f"\n[blue]Running {len(datasets)} dataset evaluations...[/blue]")
        dataset_reports = await run_dataset_evaluations(datasets)

    # Generate combined summary
    if test_results or dataset_reports:
        console.print(f"\n{'=' * 60}")
        generate_combined_summary(test_results, dataset_reports, console)

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

            with open(json_report, "w") as f:
                json.dump(combined_report, f, indent=2, default=str)
            console.print(f"JSON report saved to {json_report}")

        if markdown_report:
            generate_combined_markdown_report(combined_report, markdown_report)
            console.print(f"Markdown report saved to {markdown_report}")

        if html_report:
            generate_combined_html_report(combined_report, html_report)
            console.print(f"HTML report saved to {html_report}")

    # Exit with error if any tests failed
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
    from .datasets import Dataset

    async def _run_dataset():
        try:
            dataset = Dataset.from_file(dataset_file)
            console.print(f"Loaded dataset: {dataset.name}")
            console.print(f"Cases: {len(dataset.cases)}")

            # Mock task function for demo
            async def mock_task(inputs):
                return f"Mock response for: {inputs}"

            report = await dataset.evaluate(mock_task)
            report.print(include_input=True, include_output=True)

            # Save reports
            import json

            with open(f"{output}.json", "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            console.print(f"Report saved to {output}.json")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_run_dataset())


if __name__ == "__main__":
    app()
