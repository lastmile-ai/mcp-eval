import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
from mcp_eval.core import TestResult
from mcp_eval.reporting import generate_console_report, generate_json_report, generate_coverage_report
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
import dataclasses

app = typer.Typer()
console = Console()

def discover_tasks(path: Path):
    """Discovers all functions decorated with @mcpeval.task."""
    tasks = []
    for py_file in path.rglob("*.py"):
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, obj in inspect.getmembers(module):
            if callable(obj) and hasattr(obj, "_is_mcpeval_task"):
                tasks.append(obj)
    return tasks

async def get_available_tools(server_name: str) -> set:
    """Connects to a server to get a list of its available tools."""
    available_tools = set()
    try:
        temp_app = MCPApp()
        await temp_app.initialize()
        temp_agent = Agent(name=f"mcpeval-lister-{server_name}", server_names=[server_name], context=temp_app.context)
        await temp_agent.initialize()
        tools_result = await temp_agent.list_tools()
        available_tools = {tool.name for tool in tools_result.tools}
        await temp_agent.shutdown()
        await temp_app.cleanup()
    except Exception as e:
        console.print(f"[bold red]Error:[/] Could not list tools for server '{server_name}': {e}")
    return available_tools

async def main(test_dir: str, json_report_path: str):
    test_path = Path(test_dir)
    if not test_path.is_dir():
        console.print(f"[bold red]Error:[/] Directory not found: {test_dir}")
        raise typer.Exit(code=1)

    tasks = discover_tasks(test_path)
    if not tasks:
        console.print(f"[bold yellow]Warning:[/] No tests found in {test_dir}")
        raise typer.Exit()

    tasks_by_server: Dict[str, List] = {}
    for task in tasks:
        server = task._server
        if server not in tasks_by_server:
            tasks_by_server[server] = []
        tasks_by_server[server].append(task)
        
    all_results: List[TestResult] = []
    coverage_reports: Dict[str, Any] = {}

    for server_name, tasks_for_server in tasks_by_server.items():
        console.print(f"\n[bold blue]Running tests for server: '{server_name}'[/bold blue]")
        
        available_tools = await get_available_tools(server_name)
        
        server_results = []
        for task in tasks_for_server:
            try:
                result = await task()
                server_results.append(result)
            except Exception as e:
                console.print(f"[bold red]FATAL ERROR[/] running test '{task.__name__}': {e}")
        
        all_results.extend(server_results)
        
        called_tools = set()
        for result in server_results:
            if result and result.metrics:
                for tool_name in result.metrics.tool_metrics.keys():
                    called_tools.add(tool_name)
                    
        uncalled_tools = available_tools - called_tools
        coverage_percentage = 0
        if available_tools:
            # Exclude the human input tool from coverage calculation
            if "__human_input__" in available_tools:
                available_tools.remove("__human_input__")
            coverage_percentage = (len(called_tools) / len(available_tools)) * 100 if available_tools else 100
            
        coverage_reports[server_name] = {
            "coverage": coverage_percentage,
            "total_tools": len(available_tools),
            "called_tools": len(called_tools),
            "uncalled_tools": sorted(list(uncalled_tools))
        }

    console.print("\n" + "="*20 + " TEST RESULTS " + "="*20)
    generate_console_report(all_results, console)

    generate_coverage_report(coverage_reports, console)

    if json_report_path:
        report_data = {
            "results": [dataclasses.asdict(r) for r in all_results],
            "coverage": coverage_reports
        }
        with open(json_report_path, "w") as f:
            json.dump(report_data, f, cls=EnhancedJSONEncoder, indent=2)
        console.print(f"\n[bold green]JSON report saved to:[/] {json_report_path}")


@app.command()
def run(
    test_dir: str = typer.Argument("tests", help="Directory to scan for tests."),
    json_report: str = typer.Option(None, help="Path to save a JSON report."),
):
    asyncio.run(main(test_dir, json_report))

# Need a custom JSON encoder for the report
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

if __name__ == "__main__":
    app()