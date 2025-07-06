"""Enhanced test runner supporting both decorator and dataset approaches."""

import asyncio
import importlib.util
import inspect
import itertools
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from .core import TestResult
from .datasets import Dataset
from .reports import EvaluationReport

app = typer.Typer()
console = Console()


def discover_tests_and_datasets(path: Path) -> Dict[str, List]:
    """Discover both decorator-style tests and dataset-style evaluations."""
    tasks = []
    datasets = []

    for py_file in path.rglob("*.py"):
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
                        tasks.append(obj)

                # Discover datasets
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
        params = getattr(task_func, "_mcpeval_parameters", {})
        if not params:
            expanded.append({"func": task_func, "kwargs": {}})
            continue

        # Create cartesian product of all parameters
        param_names = list(params.keys())
        param_values = list(params.values())

        for combination in itertools.product(*param_values):
            kwargs = dict(zip(param_names, combination))
            expanded.append({"func": task_func, "kwargs": kwargs})

    return expanded


async def run_decorator_tests(test_cases: List[Dict[str, Any]]) -> List[TestResult]:
    """Run decorator-style tests."""
    results = []

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
                result = await func(**kwargs)
                results.append(result)

                status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
                console.print(f"  {status} {test_name}")

                if not result.passed:
                    if result.error:
                        console.print(f"    Error: {result.error}")
                    for eval_result in result.evaluation_results:
                        if not eval_result.get("passed", True):
                            console.print(f"    Failed: {eval_result['name']}")
                            if eval_result.get("error"):
                                console.print(f"      {eval_result['error']}")

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


@app.command()
async def run(
    test_dir: str = typer.Argument(
        "tests", help="Directory to scan for tests and datasets"
    ),
    json_report: str = typer.Option(None, "--json", help="Save JSON report"),
    markdown_report: str = typer.Option(
        None, "--markdown", help="Save Markdown report"
    ),
    format: str = typer.Option("auto", help="Output format (auto, decorator, dataset)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    max_concurrency: int = typer.Option(
        None, "--max-concurrency", help="Maximum concurrent evaluations"
    ),
):
    """Run MCP-Eval tests and datasets."""
    test_path = Path(test_dir)

    if not test_path.exists():
        console.print(f"[red]Error:[/] Test directory '{test_dir}' not found")
        raise typer.Exit(1)

    console.print(f"[blue]Discovering tests and datasets in {test_dir}...[/blue]")
    discovered = discover_tests_and_datasets(test_path)

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
        test_cases = expand_parametrized_tests(tasks)
        test_results = await run_decorator_tests(test_cases)

    if datasets and format in ["auto", "dataset"]:
        console.print(f"\n[blue]Running {len(datasets)} dataset evaluations...[/blue]")
        dataset_reports = await run_dataset_evaluations(datasets)

    # Generate combined summary
    if test_results or dataset_reports:
        console.print(f"\n{'=' * 60}")
        _generate_combined_summary(test_results, dataset_reports)

    # Generate reports
    if json_report or markdown_report:
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
            _generate_combined_markdown_report(combined_report, markdown_report)
            console.print(f"Markdown report saved to {markdown_report}")

    # Exit with error if any tests failed
    total_failed = sum(1 for r in test_results if not r.passed) + sum(
        r.failed_cases for r in dataset_reports
    )

    if total_failed > 0:
        raise typer.Exit(1)


def _generate_combined_summary(
    test_results: List[TestResult], dataset_reports: List[EvaluationReport]
):
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
    total_decorator_tests = len(test_results)
    passed_decorator_tests = sum(1 for r in test_results if r.passed)

    total_dataset_cases = sum(r.total_cases for r in dataset_reports)
    passed_dataset_cases = sum(r.passed_cases for r in dataset_reports)

    total_tests = total_decorator_tests + total_dataset_cases
    total_passed = passed_decorator_tests + passed_dataset_cases

    console.print("\n[bold]Overall Summary:[/]")
    console.print(
        f"  Decorator Tests: {passed_decorator_tests}/{total_decorator_tests} passed"
    )
    console.print(
        f"  Dataset Cases: {passed_dataset_cases}/{total_dataset_cases} passed"
    )
    console.print(
        f"  [bold]Total: {total_passed}/{total_tests} passed ({total_passed / total_tests * 100:.1f}%)[/]"
    )


def _generate_combined_markdown_report(report_data: Dict[str, Any], output_path: str):
    """Generate a combined markdown report."""
    summary = report_data["summary"]

    report = f"""# MCP-Eval Combined Test Report

## Summary

- **Decorator Tests**: {summary["passed_decorator_tests"]}/{summary["total_decorator_tests"]} passed
- **Dataset Cases**: {summary["passed_dataset_cases"]}/{summary["total_dataset_cases"]} passed
- **Overall Success Rate**: {(summary["passed_decorator_tests"] + summary["passed_dataset_cases"]) / (summary["total_decorator_tests"] + summary["total_dataset_cases"]) * 100:.1f}%

## Decorator Test Results

| Test | Status | Duration | Server |
|------|--------|----------|--------|
"""

    for test_data in report_data["decorator_tests"]:
        status = "✅ PASS" if test_data["passed"] else "❌ FAIL"
        duration = f"{test_data.get('duration_ms', 0):.1f}ms"
        server = test_data.get("server_name", "unknown")

        report += f"| {test_data['test_name']} | {status} | {duration} | {server} |\n"

    report += "\n## Dataset Evaluation Results\n\n"

    for dataset_data in report_data["dataset_reports"]:
        dataset_summary = dataset_data["summary"]
        report += f"### {dataset_data['dataset_name']}\n\n"
        report += f"- **Cases**: {dataset_summary['passed_cases']}/{dataset_summary['total_cases']} passed\n"
        report += f"- **Success Rate**: {dataset_summary['success_rate'] * 100:.1f}%\n"
        report += f"- **Average Duration**: {dataset_summary['average_duration_ms']:.1f}ms\n\n"

    with open(output_path, "w") as f:
        f.write(report)


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
