"""Doctor command for diagnosing MCP-Eval setup issues."""

import asyncio
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mcp_eval.cli.validate import validate as run_validation, ValidationResult
from mcp_eval.cli.utils import find_config_files, load_yaml

app = typer.Typer(help="Diagnose MCP-Eval setup")
console = Console()


def check_python_version() -> ValidationResult:
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return ValidationResult(
            name="Python Version",
            success=False,
            message=f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.9)",
        )
    return ValidationResult(
        name="Python Version",
        success=True,
        message=f"Python {version.major}.{version.minor}.{version.micro}",
    )


def check_package_versions() -> ValidationResult:
    """Check installed package versions."""
    try:
        import mcp_eval
        import mcp_agent
        import mcp
        
        versions = {
            "mcp-eval": getattr(mcp_eval, "__version__", "unknown"),
            "mcp-agent": getattr(mcp_agent, "__version__", "unknown"),
            "mcp": getattr(mcp, "__version__", "unknown"),
        }
        
        return ValidationResult(
            name="Package Versions",
            success=True,
            message="All required packages installed",
            details=versions
        )
    except ImportError as e:
        return ValidationResult(
            name="Package Versions",
            success=False,
            message=f"Missing package: {e.name}",
        )


def check_config_files(project: Path) -> ValidationResult:
    """Check for required configuration files."""
    paths = find_config_files(project)
    missing = []
    found = []
    
    if not paths.mcpeval_yaml.exists():
        missing.append("mcpeval.yaml")
    else:
        found.append("mcpeval.yaml")
    
    if not paths.mcpeval_secrets.exists():
        missing.append("mcpeval.secrets.yaml")
    else:
        found.append("mcpeval.secrets.yaml")
    
    if paths.mcp_agent_config.exists():
        found.append("mcp-agent.config.yaml")
    
    if paths.mcp_json and paths.mcp_json.exists():
        found.append(str(paths.mcp_json))
    
    if missing:
        return ValidationResult(
            name="Config Files",
            success=False,
            message=f"Missing: {', '.join(missing)}",
            details={"found": found, "missing": missing}
        )
    
    return ValidationResult(
        name="Config Files",
        success=True,
        message=f"Found: {', '.join(found)}",
        details={"found": found}
    )


def check_test_reports(project: Path) -> ValidationResult:
    """Check test reports directory."""
    cfg = load_yaml(project / "mcpeval.yaml")
    report_dir = Path(cfg.get("reporting", {}).get("output_dir", "./test-reports"))
    
    if not report_dir.is_absolute():
        report_dir = project / report_dir
    
    if not report_dir.exists():
        return ValidationResult(
            name="Test Reports",
            success=True,
            message=f"Directory not created yet: {report_dir}",
        )
    
    # Count files
    json_files = list(report_dir.glob("*.json"))
    trace_files = list(report_dir.glob("*.jsonl"))
    md_files = list(report_dir.glob("*.md"))
    
    total = len(json_files) + len(trace_files) + len(md_files)
    
    if total == 0:
        return ValidationResult(
            name="Test Reports",
            success=True,
            message="No test reports found (run tests first)",
        )
    
    return ValidationResult(
        name="Test Reports",
        success=True,
        message=f"Found {total} files ({len(json_files)} json, {len(trace_files)} traces, {len(md_files)} markdown)",
        details={
            "json_files": len(json_files),
            "trace_files": len(trace_files),
            "md_files": len(md_files),
            "total": total
        }
    )


def check_environment_vars() -> ValidationResult:
    """Check for relevant environment variables."""
    import os
    
    vars_found = {}
    vars_checked = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MCP_EVAL_CONFIG",
        "PYTHONPATH",
        "PATH",
    ]
    
    for var in vars_checked:
        value = os.environ.get(var)
        if value:
            if "API_KEY" in var:
                # Mask API keys
                vars_found[var] = value[:8] + "..." if len(value) > 8 else "***"
            elif var == "PATH":
                # Just show if it contains common tool paths
                paths_of_interest = ["node", "python", "npm", "npx", "uvx"]
                found_paths = [p for p in paths_of_interest if p in value.lower()]
                vars_found[var] = f"Contains: {', '.join(found_paths)}" if found_paths else "Set"
            else:
                vars_found[var] = value[:50] + "..." if len(value) > 50 else value
    
    if not vars_found:
        return ValidationResult(
            name="Environment",
            success=True,
            message="No relevant environment variables set",
        )
    
    return ValidationResult(
        name="Environment",
        success=True,
        message=f"Found {len(vars_found)} relevant variables",
        details=vars_found
    )


def check_system_info() -> ValidationResult:
    """Gather system information."""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
    }
    
    return ValidationResult(
        name="System Info",
        success=True,
        message=f"{platform.system()} {platform.machine()}",
        details=info
    )


def get_last_error(project: Path) -> Optional[Dict[str, Any]]:
    """Try to find the last error from test reports."""
    cfg = load_yaml(project / "mcpeval.yaml")
    report_dir = Path(cfg.get("reporting", {}).get("output_dir", "./test-reports"))
    
    if not report_dir.is_absolute():
        report_dir = project / report_dir
    
    if not report_dir.exists():
        return None
    
    # Find most recent JSON result file
    json_files = sorted(report_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    for result_file in json_files[:5]:  # Check last 5 files
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Look for failures
            if "results" in data:
                for result in data["results"]:
                    if not result.get("passed", True):
                        return {
                            "file": str(result_file),
                            "test": data.get("test_name", "unknown"),
                            "error": result.get("error", "Unknown error"),
                            "timestamp": data.get("timestamp", "unknown")
                        }
        except Exception:
            continue
    
    return None


@app.command()
def doctor(
    project_dir: str = typer.Option(".", help="Project directory"),
    full: bool = typer.Option(False, help="Run full validation including connection tests"),
):
    """Run comprehensive diagnostics on MCP-Eval setup.
    
    Checks:
    - Python version and packages
    - Configuration files
    - Environment variables
    - System information
    - Runs validation checks
    - Looks for recent errors
    
    Examples:
      - Quick diagnosis:
        mcp-eval doctor
        
      - Full diagnosis with connection tests:
        mcp-eval doctor --full
    """
    project = Path(project_dir)
    
    console.print("\n[bold cyan]🩺 Running MCP-Eval Doctor[/bold cyan]\n")
    
    results: List[ValidationResult] = []
    
    # System checks
    console.print("[bold]System Checks[/bold]")
    
    result = check_python_version()
    results.append(result)
    _print_result(result)
    
    result = check_package_versions()
    results.append(result)
    _print_result(result)
    
    result = check_system_info()
    results.append(result)
    _print_result(result)
    
    result = check_environment_vars()
    results.append(result)
    _print_result(result)
    
    # Configuration checks
    console.print("\n[bold]Configuration Checks[/bold]")
    
    result = check_config_files(project)
    results.append(result)
    _print_result(result)
    
    if (project / "mcpeval.yaml").exists():
        # Run validation
        console.print("\n[bold]Validation Checks[/bold]")
        console.print("[dim]Running validation...[/dim]")
        
        # Call the validate command
        try:
            from mcp_eval.cli.validate import (
                check_api_keys,
                check_judge_config,
                validate_server,
                validate_agent,
                load_all_servers,
                load_all_agents,
            )
            
            # API Keys
            result = check_api_keys(project)
            results.append(result)
            _print_result(result)
            
            # Judge
            result = check_judge_config(project)
            results.append(result)
            _print_result(result)
            
            if full:
                # Full validation with connections
                servers = load_all_servers(project)
                for name, server in servers.items():
                    result = asyncio.run(validate_server(server))
                    results.append(result)
                    _print_result(result)
                
                agents = load_all_agents(project)
                for agent in agents:
                    result = asyncio.run(validate_agent(agent, project))
                    results.append(result)
                    _print_result(result)
            
        except Exception as e:
            console.print(f"[red]Validation error: {e}[/red]")
    
    # Check for test reports
    console.print("\n[bold]Test Reports[/bold]")
    result = check_test_reports(project)
    results.append(result)
    _print_result(result)
    
    # Check for recent errors
    last_error = get_last_error(project)
    if last_error:
        console.print("\n[bold yellow]⚠️  Last Test Error[/bold yellow]")
        console.print(f"Test: {last_error['test']}")
        console.print(f"Error: {last_error['error'][:200]}...")
        console.print(f"File: {last_error['file']}")
    
    # Summary
    console.print("\n[bold]Diagnosis Summary[/bold]")
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count
    
    if fail_count == 0:
        console.print(f"[green]✅ All {len(results)} checks passed![/green]")
        console.print("\nYour MCP-Eval setup looks good! 🎉")
    else:
        console.print(f"[yellow]⚠️  {success_count} passed, {fail_count} need attention[/yellow]")
        
        # Provide suggestions
        console.print("\n[bold]Suggested Actions:[/bold]")
        
        failed = [r for r in results if not r.success]
        for r in failed:
            if "Config Files" in r.name and "mcpeval.yaml" in r.message:
                console.print("  • Run [cyan]mcp-eval init[/cyan] to create configuration")
            elif "API Keys" in r.name:
                console.print("  • Add API keys to mcpeval.secrets.yaml")
            elif "Python Version" in r.name:
                console.print("  • Upgrade to Python 3.9 or higher")
            elif "Package" in r.name:
                console.print("  • Install missing packages with [cyan]pip install mcp-eval[/cyan]")
            elif "server" in r.name.lower():
                console.print(f"  • Check server '{r.name}' configuration and connectivity")
            elif "agent" in r.name.lower():
                console.print(f"  • Fix agent '{r.name}' configuration")
        
        console.print("\nRun [cyan]mcp-eval doctor --full[/cyan] for complete diagnostics")
        
        if last_error:
            console.print("\n💡 To create an issue, run: [cyan]mcp-eval issue[/cyan]")


def _print_result(result: ValidationResult):
    """Print a validation result."""
    if result.success:
        icon = "[green]✓[/green]"
    else:
        icon = "[red]✗[/red]"
    
    console.print(f"{icon} {result.name}: {result.message}")
    
    # Show some details
    if result.details and not result.success:
        for key, value in result.details.items():
            if key != "error":  # Don't repeat error
                console.print(f"  [dim]{key}: {value}[/dim]")


if __name__ == "__main__":
    app()