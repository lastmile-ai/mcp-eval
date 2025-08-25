"""Issue command for creating GitHub issues."""

import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import quote
from datetime import datetime

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from mcp_eval.cli.utils import load_yaml, find_config_files
from mcp_eval.config import find_eval_config
from mcp_eval.cli.doctor import (
    check_python_version,
    check_package_versions,
    check_system_info,
    get_last_error,
)

app = typer.Typer(help="Create GitHub issues with diagnostic info")
console = Console()


def gather_diagnostic_info(project: Path, config_path: Path | None) -> Dict[str, Any]:
    """Gather diagnostic information for the report."""
    info = {}

    # System info
    sys_info = check_system_info()
    info["system"] = sys_info.details

    # Python version
    py_info = check_python_version()
    info["python"] = py_info.message

    # Package versions
    pkg_info = check_package_versions()
    info["packages"] = (
        pkg_info.details if pkg_info.details else {"error": pkg_info.message}
    )

    # Config files
    paths = find_config_files(project)
    info["config_files"] = {
        "mcpeval.yaml": paths.mcpeval_yaml.exists(),
        "mcpeval.secrets.yaml": paths.mcpeval_secrets.exists(),
        "mcp-agent.config.yaml": paths.mcp_agent_config.exists(),
    }

    # Include actual config content
    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                info["config_content"] = f.read()
        except Exception as e:
            info["config_content"] = f"Error reading config: {e}"

    # Working directory info
    import os

    info["working_directory"] = os.getcwd()
    info["project_directory"] = str(project.absolute())

    # Last error
    last_error = get_last_error(project, config_path)
    if last_error:
        info["last_error"] = last_error

    return info


def get_recent_test_outputs(
    project: Path, config_path: Path | None, max_files: int = 3
) -> List[Dict[str, Any]]:
    """Get recent test outputs."""
    if config_path is None:
        console.print(
            "[yellow]Warning: No mcpeval config found, cannot retrieve test outputs[/yellow]"
        )
        return []

    cfg = load_yaml(config_path)
    report_dir = Path(cfg.get("reporting", {}).get("output_dir", "./test-reports"))

    if not report_dir.is_absolute():
        report_dir = project / report_dir

    if not report_dir.exists():
        return []

    outputs = []

    # Get recent result files
    result_files = sorted(
        report_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:max_files]

    for file in result_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

            # Extract key information
            output = {
                "filename": file.name,
                "test_name": data.get("test_name", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
                "all_passed": data.get("all_passed", False),
                "duration_ms": data.get("duration_ms", 0),
            }

            # Find failures
            failures = []
            if "results" in data:
                for result in data["results"]:
                    if not result.get("passed", True):
                        failures.append(
                            {
                                "name": result.get("name", "unknown"),
                                "error": result.get("error", "Unknown error")[:200],
                            }
                        )

            if failures:
                output["failures"] = failures[:3]  # Limit to 3 failures

            outputs.append(output)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file}: {e}[/yellow]")

    return outputs


def format_issue_body(
    description: str,
    category: str,
    diagnostics: Dict[str, Any],
    test_outputs: List[Dict[str, Any]],
    command: str | None = None,
    error_message: str | None = None,
) -> str:
    """Format the GitHub issue body."""

    lines = []
    lines.append("## Description")
    lines.append(description)
    lines.append("")

    if error_message:
        lines.append("## Error Message")
        lines.append("```")
        lines.append(error_message[:1000])  # Limit length
        lines.append("```")
        lines.append("")

    if command:
        lines.append("## Command Run")
        lines.append("```bash")
        lines.append(command)
        lines.append("```")
        lines.append("")

    lines.append("## Environment")
    lines.append("```json")
    env_info = {
        "python": diagnostics["python"],
        "system": diagnostics["system"],
        "working_directory": diagnostics.get("working_directory", "unknown"),
        "project_directory": diagnostics.get("project_directory", "unknown"),
        "packages": diagnostics.get("packages", {}),
        "config_files": diagnostics.get("config_files", {}),
    }
    lines.append(json.dumps(env_info, indent=2))
    lines.append("```")
    lines.append("")

    # Add config content if available
    if "config_content" in diagnostics:
        lines.append("## Configuration (mcpeval.yaml)")
        lines.append("```yaml")
        lines.append(diagnostics["config_content"])
        lines.append("```")
        lines.append("")

    if test_outputs:
        lines.append("## Recent Test Results")
        for output in test_outputs[:2]:  # Limit to 2
            lines.append(f"### {output['test_name']}")
            lines.append(f"- **File**: {output['filename']}")
            lines.append(f"- **Passed**: {output['all_passed']}")
            lines.append(f"- **Duration**: {output['duration_ms']:.0f}ms")

            if "failures" in output:
                lines.append("- **Failures**:")
                for failure in output["failures"]:
                    lines.append(f"  - {failure['name']}: `{failure['error'][:100]}`")
            lines.append("")

    if "last_error" in diagnostics:
        lines.append("## Last Error Details")
        lines.append("```")
        lines.append(diagnostics["last_error"]["error"][:500])
        lines.append("```")
        lines.append(f"From test: {diagnostics['last_error']['test']}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by mcp-eval doctor on {datetime.now().isoformat()}*")

    return "\n".join(lines)


@app.command()
def issue(
    project_dir: str = typer.Option(".", help="Project directory"),
    title: str | None = typer.Option(None, help="Issue title"),
    include_outputs: bool = typer.Option(True, help="Include recent test outputs"),
    open_browser: bool = typer.Option(True, help="Open browser to create issue"),
):
    """Create a GitHub issue with diagnostic information.

    Gathers:

    - System and environment information

    - Recent test results and errors

    - Configuration status

    - Last command run



    Examples:

    Create issue interactively: $ mcp-eval issue

    Create with title: $ mcp-eval issue --title "Server connection fails"

    Just show URL without opening browser: $ mcp-eval issue --no-open-browser
    """
    project = Path(project_dir)

    console.print("\n[bold cyan]📝 Creating GitHub Issue[/bold cyan]\n")

    # Find config file once
    config_path = find_eval_config(project)
    if config_path:
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        console.print(
            "[yellow]Warning: No mcpeval config found (searched for mcpeval.yaml, mcpeval.config.yaml, .mcp-eval/config.yaml, etc.)[/yellow]"
        )
        if not Confirm.ask("Continue without configuration?", default=False):
            raise typer.Exit(0)

    # Gather diagnostic info
    console.print("Gathering diagnostic information...")
    diagnostics = gather_diagnostic_info(project, config_path)

    # Get recent test outputs if requested
    test_outputs = []
    if include_outputs:
        console.print("Collecting recent test results...")
        test_outputs = get_recent_test_outputs(project, config_path)
        if test_outputs:
            console.print(f"Found {len(test_outputs)} recent test results")

    # Get issue details
    if not title:
        title = Prompt.ask("Issue title (brief summary)", default="MCP-Eval Issue")

    console.print("\n[bold]Issue Categories:[/bold]")
    console.print("1. Bug - Something isn't working")
    console.print("2. Server Issue - Problem with MCP server")
    console.print("3. Configuration - Setup or config problem")
    console.print("4. Test Generation - Issues with test generation")
    console.print("5. Documentation - Docs unclear or missing")
    console.print("6. Feature Request - New feature idea")
    console.print("7. Other")

    category_num = Prompt.ask(
        "Select category", choices=["1", "2", "3", "4", "5", "6", "7"], default="1"
    )
    categories = {
        "1": "bug",
        "2": "server",
        "3": "configuration",
        "4": "generation",
        "5": "documentation",
        "6": "enhancement",
        "7": "question",
    }
    category = categories[category_num]

    description = Prompt.ask("Describe the issue", default="")

    # Ask for command that caused the issue
    command = Prompt.ask(
        "Command that caused the issue (e.g., 'mcp-eval run test_fetch.py')", default=""
    )

    # Ask for error message
    console.print("\n[bold]Error message:[/bold]")
    console.print(
        "[dim]Paste your error message below, then type CTRL+C to finish (or just press Enter to skip)[/dim]"
    )

    error_lines = []
    empty_line_count = 0

    while True:
        try:
            line = input()
            if line == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    # Two consecutive empty lines means we're done
                    break
                elif len(error_lines) == 0:
                    # First line is empty, skip error message
                    break
                else:
                    # Add the empty line (it might be part of the error)
                    error_lines.append(line)
            else:
                empty_line_count = 0
                error_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D (EOFError) or Ctrl+C (KeyboardInterrupt)
            break

    error_message = "\n".join(error_lines) if error_lines else None

    # Format the issue
    body = format_issue_body(
        description=description or "Issue encountered while using mcp-eval",
        category=category,
        diagnostics=diagnostics,
        test_outputs=test_outputs,
        command=command,
        error_message=error_message,
    )

    # Create GitHub URL
    repo_url = "https://github.com/lastmile-ai/mcp-eval"

    # Build labels
    labels = [category]
    if "last_error" in diagnostics or error_message:
        labels.append("has-error")

    # Create issue URL with pre-filled content
    params = {
        "title": title,
        "body": body,
        "labels": ",".join(labels),
    }

    # URL encode parameters
    query_parts = []
    for key, value in params.items():
        if value:
            query_parts.append(f"{key}={quote(str(value))}")

    issue_url = f"{repo_url}/issues/new?{'&'.join(query_parts)}"

    # Show summary
    console.print("\n[bold]Issue Summary:[/bold]")
    panel_content = f"""[bold]Title:[/bold] {title}
[bold]Category:[/bold] {category}
[bold]Labels:[/bold] {", ".join(labels)}
[bold]Test Results:[/bold] {len(test_outputs)} included
[bold]Has Error:[/bold] {"Yes" if error_message or "last_error" in diagnostics else "No"}"""

    console.print(Panel(panel_content, title="Issue Details", border_style="cyan"))

    # Show the body preview
    if Confirm.ask("\nShow issue body preview?", default=False):
        console.print("\n[bold]Issue Body:[/bold]")
        console.print(
            Panel(body[:500] + "..." if len(body) > 500 else body, border_style="dim")
        )

    # Open browser or show URL
    console.print("\n[bold green]✅ Issue report prepared![/bold green]")

    if open_browser:
        console.print("\nOpening browser to create issue...")
        try:
            webbrowser.open(issue_url)
            console.print("[green]Browser opened successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not open browser: {e}[/yellow]")
            console.print("\nPlease visit this URL manually:")
            console.print(f"[cyan]{repo_url}/issues/new[/cyan]")
            console.print("\nAnd paste the following content:")
            console.print(Panel(body, title="Issue Content", border_style="cyan"))
    else:
        console.print("\n[bold]To create the issue:[/bold]")
        console.print(f"1. Visit: [cyan]{repo_url}/issues/new[/cyan]")
        console.print("2. Paste the following content:\n")
        console.print(Panel(body, title="Issue Content", border_style="cyan"))

    console.print("\n💡 [dim]Thank you for creating this issue![/dim]")


if __name__ == "__main__":
    app()
