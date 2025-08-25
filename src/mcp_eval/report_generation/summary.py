"""Summary report generation for MCP-Eval."""

from typing import List, Dict, Set
from rich.console import Console
from rich.table import Table

from mcp_eval.core import TestResult
from mcp_eval.report_generation.models import EvaluationReport
from mcp_eval.report_generation.base import calculate_overall_stats
from mcp_eval.config import get_settings


def generate_combined_summary(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
    verbose: bool = False,
) -> None:
    """Generate a combined summary of all results.

    Args:
        test_results: List of decorator test results
        dataset_reports: List of dataset evaluation reports
        console: Rich console for output
        verbose: If True, show detailed configuration and metrics
    """
    # Main summary table
    table = Table(title="Combined Test Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Cases/Tests", justify="right")
    table.add_column("Duration", justify="right")

    if verbose:
        table.add_column("Agent", style="blue")
        table.add_column("Servers (Configured)", style="yellow")
        table.add_column("Servers (Used)", style="green")

    # Add decorator test results
    for result in test_results:
        status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
        duration = f"{result.duration_ms:.1f}ms" if result.duration_ms else "N/A"

        row = ["Test", result.test_name, status, "1", duration]

        if verbose:
            agent_name = result.agent_name or "default"
            # Configured servers (what the agent has access to)
            configured_servers = ", ".join(result.servers) if result.servers else "none"
            # Used servers (what was actually invoked)
            used_servers = set()
            if result.metrics and isinstance(result.metrics, dict):
                if "unique_servers_used" in result.metrics:
                    used_servers.update(result.metrics["unique_servers_used"])
            used_servers_str = (
                ", ".join(sorted(used_servers)) if used_servers else "none"
            )
            row.extend([agent_name, configured_servers, used_servers_str])

        table.add_row(*row)

    # Add dataset results
    for report in dataset_reports:
        passed = report.passed_cases
        total = report.total_cases
        status = (
            f"[green]{passed}/{total}[/]"
            if passed == total
            else f"[yellow]{passed}/{total}[/]"
        )
        duration = f"{report.average_duration_ms:.1f}ms"

        row = ["Dataset", report.dataset_name, status, str(total), duration]

        if verbose:
            # Extract agent name from report or first case
            agent_name = (
                report.agent_name
                if hasattr(report, "agent_name") and report.agent_name
                else "unknown"
            )

            # Configured servers (what the agent has access to)
            configured_servers = set()
            for case in report.results:
                if hasattr(case, "servers") and case.servers:
                    configured_servers.update(case.servers)
                    break  # All cases should have the same configuration

            # Used servers (what was actually invoked)
            used_servers = set()
            for case in report.results:
                if case.metrics and hasattr(case.metrics, "unique_servers_used"):
                    used_servers.update(case.metrics.unique_servers_used)

            configured_str = (
                ", ".join(sorted(configured_servers)) if configured_servers else "none"
            )
            used_str = ", ".join(sorted(used_servers)) if used_servers else "none"
            row.extend([agent_name, configured_str, used_str])

        table.add_row(*row)

    console.print(table)

    # In verbose mode, show configuration details
    if verbose:
        _print_configuration_details(test_results, dataset_reports, console)
        _print_test_breakdown(test_results, dataset_reports, console)

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

    # Show aggregate information in verbose mode
    if verbose:
        _print_aggregate_info(test_results, dataset_reports, console)
        _print_tool_coverage(test_results, dataset_reports, console)


def _print_configuration_details(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
) -> None:
    """Print test configuration details."""
    console.print("\n[bold cyan]Test Environment Configuration[/bold cyan]")
    console.print("=" * 50)

    try:
        settings = get_settings()

        # Primary LLM Configuration
        console.print("\n[cyan]Primary LLM:[/cyan]")
        if settings.provider:
            console.print(f"  Provider: {settings.provider}")
        if settings.model:
            console.print(f"  Model: {settings.model}")

        # LLM Judge Configuration
        console.print("\n[cyan]LLM Judge:[/cyan]")
        if hasattr(settings, "judge") and settings.judge:
            judge = settings.judge
            # Judge inherits from primary config if not set
            judge_provider = judge.provider if judge.provider else settings.provider
            judge_model = judge.model if judge.model else settings.model

            if judge_provider:
                console.print(f"  Provider: {judge_provider}")
            else:
                console.print("  Provider: [dim]Auto-detected from model[/dim]")

            if judge_model:
                console.print(f"  Model: {judge_model}")
            else:
                console.print("  Model: [dim]Using model selector[/dim]")
        else:
            console.print("  [dim]Using default configuration[/dim]")

        # Default servers if configured
        if hasattr(settings, "default_servers") and settings.default_servers:
            console.print("\n[cyan]Default MCP Servers:[/cyan]")
            for server in settings.default_servers:
                console.print(f"  • {server}")
    except Exception as e:
        console.print(f"[yellow]Unable to load configuration: {e}[/yellow]")


def _print_test_breakdown(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
) -> None:
    """Print detailed breakdown of each test."""

    if test_results or dataset_reports:
        console.print("\n[bold cyan]Test Details[/bold cyan]")
        console.print("=" * 50)

    # Group tests by agent
    agents_used: Dict[str, List[TestResult]] = {}
    for result in test_results:
        agent_name = result.agent_name or "default"
        if agent_name not in agents_used:
            agents_used[agent_name] = []
        agents_used[agent_name].append(result)

    if agents_used:
        console.print("\n[cyan]Tests by Agent:[/cyan]")
        for agent_name, tests in agents_used.items():
            console.print(f"\n  [blue]{agent_name}:[/blue]")
            for test in tests:
                status = "✓" if test.passed else "✗"
                status_color = "green" if test.passed else "red"
                console.print(
                    f"    [{status_color}]{status}[/{status_color}] {test.test_name}"
                )

                # Show servers used
                if test.servers:
                    console.print(
                        f"      Servers: {', '.join(test.servers)}", style="dim"
                    )

                # Show tools called if available
                if test.metrics and isinstance(test.metrics, dict):
                    if "tool_calls" in test.metrics:
                        tool_names = set()
                        for call in test.metrics["tool_calls"]:
                            if isinstance(call, dict) and "tool_name" in call:
                                tool_names.add(call["tool_name"])
                        if tool_names:
                            console.print(
                                f"      Tools: {', '.join(sorted(tool_names))}",
                                style="dim",
                            )

                # Show evaluators used (especially for failures)
                if not test.passed and test.evaluation_results:
                    failed_evals = [
                        e.name for e in test.evaluation_results if not e.passed
                    ]
                    if failed_evals:
                        console.print(
                            f"      Failed: {', '.join(failed_evals)}", style="red dim"
                        )

    # Dataset breakdown
    if dataset_reports:
        console.print("\n[cyan]Dataset Results:[/cyan]")
        for report in dataset_reports:
            console.print(f"\n  [blue]{report.dataset_name}:[/blue]")
            console.print(
                f"    Cases: {report.passed_cases}/{report.total_cases} passed"
            )

            # Show failed cases
            failed_cases = [case for case in report.results if not case.passed]
            if failed_cases:
                console.print("    Failed cases:", style="red")
                for case in failed_cases[:5]:  # Show first 5 failures
                    console.print(f"      • {case.case_name}", style="red dim")
                    if case.evaluation_results:
                        failed_evals = [
                            e.name for e in case.evaluation_results if not e.passed
                        ]
                        if failed_evals:
                            console.print(
                                f"        Failed: {', '.join(failed_evals)}",
                                style="red dim",
                            )
                if len(failed_cases) > 5:
                    console.print(
                        f"      ... and {len(failed_cases) - 5} more", style="dim"
                    )


def _print_aggregate_info(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
) -> None:
    """Print aggregate information across all tests."""
    console.print("\n[bold cyan]Aggregate Statistics[/bold cyan]")
    console.print("=" * 50)

    # Collect all unique servers from test results and metrics
    all_servers: Set[str] = set()
    for result in test_results:
        if result.servers:
            all_servers.update(result.servers)
        # Also check metrics for servers used
        if result.metrics and isinstance(result.metrics, dict):
            if "unique_servers_used" in result.metrics:
                all_servers.update(result.metrics["unique_servers_used"])

    # Collect all unique tools
    all_tools: Dict[str, int] = {}
    total_tokens = 0
    total_cost = 0.0
    total_duration = 0.0

    for result in test_results:
        total_duration += result.duration_ms
        if result.metrics and isinstance(result.metrics, dict):
            if "tool_calls" in result.metrics:
                for call in result.metrics["tool_calls"]:
                    if isinstance(call, dict) and "name" in call:
                        tool_name = call["name"]
                        all_tools[tool_name] = all_tools.get(tool_name, 0) + 1

            if "llm_metrics" in result.metrics:
                llm = result.metrics["llm_metrics"]
                if isinstance(llm, dict):
                    total_tokens += llm.get("total_tokens", 0)

            # Cost is stored at the root level of metrics
            if "cost_estimate" in result.metrics:
                total_cost += result.metrics["cost_estimate"]

    # Process dataset results
    for report in dataset_reports:
        for case in report.results:
            total_duration += case.duration_ms
            if case.metrics:
                # Handle TestMetrics object properly
                if hasattr(case.metrics, "tool_calls"):
                    # case.metrics.tool_calls is a list of ToolCall objects
                    for call in case.metrics.tool_calls:
                        # ToolCall object has a 'name' attribute
                        if hasattr(call, "name"):
                            all_tools[call.name] = all_tools.get(call.name, 0) + 1
                        elif isinstance(call, dict) and "name" in call:
                            all_tools[call["name"]] = all_tools.get(call["name"], 0) + 1

                # Also collect servers from metrics
                if hasattr(case.metrics, "unique_servers_used"):
                    all_servers.update(case.metrics.unique_servers_used)

                if hasattr(case.metrics, "llm_metrics"):
                    # case.metrics.llm_metrics is an LLMMetrics object
                    llm = case.metrics.llm_metrics
                    if hasattr(llm, "total_tokens"):
                        total_tokens += llm.total_tokens
                    if hasattr(llm, "cost_estimate"):
                        total_cost += llm.cost_estimate
                    elif hasattr(case.metrics, "cost_estimate"):
                        # Fallback to main metrics cost_estimate
                        total_cost += case.metrics.cost_estimate

    # Print aggregates
    console.print("\n[cyan]MCP Servers Used:[/cyan]")
    if all_servers:
        for server in sorted(all_servers):
            console.print(f"  • {server}")
    else:
        console.print("  [dim]None[/dim]")

    console.print("\n[cyan]Tools Called:[/cyan]")
    if all_tools:
        # Sort by frequency
        for tool, count in sorted(all_tools.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  • {tool}: {count} calls")
    else:
        console.print("  [dim]None[/dim]")

    console.print("\n[cyan]Resource Usage:[/cyan]")
    console.print(
        f"  Total Duration: {total_duration:.1f}ms ({total_duration / 1000:.2f}s)"
    )
    console.print(f"  Total Tokens: {total_tokens:,}")
    # Only show cost if there is actual cost data
    if total_cost > 0:
        console.print(f"  Estimated Cost: ${total_cost:.4f}")
    else:
        console.print("  Estimated Cost: $0.0000")


def _print_tool_coverage(
    test_results: List[TestResult],
    dataset_reports: List[EvaluationReport],
    console: Console,
) -> None:
    """Print tool coverage metrics per server."""
    console.print("\n[bold cyan]Tool Coverage by Server[/bold cyan]")
    console.print("=" * 50)

    # Aggregate tool coverage across all tests
    server_coverage: Dict[
        str, Dict[str, set]
    ] = {}  # server -> {available: set, used: set}

    # Process test results
    for result in test_results:
        if result.metrics and isinstance(result.metrics, dict):
            tool_coverage = result.metrics.get("tool_coverage", {})
            for server_name, coverage_data in tool_coverage.items():
                if server_name not in server_coverage:
                    server_coverage[server_name] = {"available": set(), "used": set()}
                # Coverage data should have available_tools and used_tools
                if isinstance(coverage_data, dict):
                    server_coverage[server_name]["available"].update(
                        coverage_data.get("available_tools", [])
                    )
                    server_coverage[server_name]["used"].update(
                        coverage_data.get("used_tools", [])
                    )

    # Process dataset results
    for report in dataset_reports:
        for case in report.results:
            if case.metrics and hasattr(case.metrics, "tool_coverage"):
                for server_name, coverage in case.metrics.tool_coverage.items():
                    if server_name not in server_coverage:
                        server_coverage[server_name] = {
                            "available": set(),
                            "used": set(),
                        }
                    server_coverage[server_name]["available"].update(
                        coverage.available_tools
                    )
                    server_coverage[server_name]["used"].update(coverage.used_tools)

    # Display coverage per server
    if server_coverage:
        from rich.table import Table

        coverage_table = Table(title="Tool Coverage")
        coverage_table.add_column("Server", style="cyan")
        coverage_table.add_column("Available Tools", justify="right")
        coverage_table.add_column("Used Tools", justify="right")
        coverage_table.add_column("Coverage", justify="right")
        coverage_table.add_column("Unused Tools", style="dim")

        for server_name in sorted(server_coverage.keys()):
            available = server_coverage[server_name]["available"]
            used = server_coverage[server_name]["used"]

            if available:
                coverage_pct = (len(used) / len(available)) * 100
                unused = available - used

                # Format coverage percentage with color coding
                if coverage_pct >= 80:
                    coverage_str = f"[green]{coverage_pct:.1f}%[/green]"
                elif coverage_pct >= 50:
                    coverage_str = f"[yellow]{coverage_pct:.1f}%[/yellow]"
                else:
                    coverage_str = f"[red]{coverage_pct:.1f}%[/red]"

                # List first 3 unused tools
                unused_list = sorted(unused)[:3]
                unused_str = ", ".join(unused_list)
                if len(unused) > 3:
                    unused_str += f" (+{len(unused) - 3} more)"
                if not unused:
                    unused_str = "[green]All tools covered[/green]"

                coverage_table.add_row(
                    server_name,
                    str(len(available)),
                    str(len(used)),
                    coverage_str,
                    unused_str,
                )
            else:
                coverage_table.add_row(
                    server_name,
                    "0",
                    str(len(used)),
                    "[dim]N/A[/dim]",
                    "[dim]No tools available[/dim]",
                )

        console.print(coverage_table)
    else:
        console.print("  [dim]No tool coverage data available[/dim]")
