"""Enhanced command-line interface for MCP-Eval."""

import typer
from pathlib import Path
from rich.console import Console

from mcp_eval.runner import app as runner_app
from mcp_eval.cli.generator import run_generator

app = typer.Typer(help="MCP-Eval: Comprehensive testing framework for MCP servers")
console = Console()

# Subcommands
app.add_typer(runner_app, name="run", help="Run tests")


@app.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata

        version = importlib.metadata.version("mcp-eval")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (development)"
    console.print(f"MCP-Eval {version}")


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize"),
    template: str = typer.Option("basic", help="Template to use (basic, advanced)"),
):
    """Initialize a new MCP-Eval project."""
    project_path = Path(directory)
    project_path.mkdir(exist_ok=True)

    # Create basic structure
    (project_path / "tests").mkdir(exist_ok=True)
    (project_path / "datasets").mkdir(exist_ok=True)

    # Create example files
    if template == "basic":
        _create_basic_template(project_path)
    elif template == "advanced":
        _create_advanced_template(project_path)

    console.print(f"[green]Initialized MCP-Eval project in {project_path}[/green]")


@app.command("generate")
def generate(
    out_dir: str = typer.Option(".", help="Project directory to write configs/tests"),
    style: str = typer.Option("pytest", help="Test style: pytest|decorators|dataset"),
    n_examples: int = typer.Option(6, help="Number of scenarios to generate"),
    provider: str = typer.Option("anthropic", help="LLM provider (anthropic|openai)"),
    model: str = typer.Option(None, help="Model hint for generation (optional)"),
):
    """Interactive generator to create configs and tests for an MCP server."""
    run_generator(out_dir=out_dir, style=style, n_examples=n_examples, provider=provider, model=model)


def _create_basic_template(project_path: Path):
    """Create basic template files."""

    # mcpeval.yaml
    config_content = """
name: "My MCP Server Tests"
description: "Test suite for my MCP server"

# Server configuration (references mcp_agent.config.yaml)
servers:
  my_server:
    command: "python"
    args: ["my_server.py"]

# Default agent configuration
agents:
  default:
    name: "test_agent"
    instruction: "You are a test agent. Complete tasks as requested."
    server_names: ["my_server"]
    provider: "anthropic"

# Judge configuration
judge:
  model: "claude-3-5-haiku-20241022"
  min_score: 0.8

# Reporting configuration
reporting:
  formats: ["json", "markdown"]
  output_dir: "./reports"
"""

    (project_path / "mcpeval.yaml").write_text(config_content.strip())

    # Example test file
    test_content = """
import mcp_eval
from mcp_eval import task, setup, ToolWasCalled, ResponseContains

@setup
def configure_tests():
    mcp_eval.use_server("my_server")

@task("Basic functionality test")
async def test_basic_functionality(agent):
    \"\"\"Test basic server functionality.\"\"\"
    response = await agent.generate_str("Perform a basic operation")
    
    agent.evaluate_now(ResponseContains("result"), response, "has_result")
    agent.add_deferred_evaluator(ToolWasCalled("basic_tool"), "tool_called")

@task("Error handling test")
async def test_error_handling(agent):
    \"\"\"Test server error handling.\"\"\"
    response = await agent.generate_str("Perform an invalid operation")
    
    agent.evaluate_now(ResponseContains("error"), response, "has_error")
"""

    (project_path / "tests" / "test_my_server.py").write_text(test_content.strip())


def _create_advanced_template(project_path: Path):
    """Create advanced template with dataset examples."""
    _create_basic_template(project_path)

    # Example dataset file
    dataset_content = """
import asyncio
from mcp_eval import Case, Dataset, ToolWasCalled, ResponseContains, LLMJudge, test_session
# Deprecated imports removed; use provider/model in configs

# Define test cases
cases = [
    Case(
        name='basic_operation',
        inputs='Perform a basic operation',
        expected_output='Operation completed successfully',
        metadata={'difficulty': 'easy', 'category': 'basic'},
        evaluators=[
            ToolWasCalled('basic_tool'),
            ResponseContains('completed'),
        ]
    ),
    Case(
        name='complex_operation',
        inputs='Perform a complex multi-step operation',
        metadata={'difficulty': 'hard', 'category': 'advanced'},
        evaluators=[
            ToolWasCalled('tool1'),
            ToolWasCalled('tool2'),
            LLMJudge('Response shows successful completion of all steps'),
        ]
    )
]

# Create dataset
dataset = Dataset(
    name='My Server Advanced Tests',
    cases=cases,
    server_name='my_server',
    
)

async def my_server_task(inputs: str) -> str:
    \"\"\"System under test.\"\"\"
    async with test_session('dataset_task') as agent:
        return await agent.generate_str(inputs)

async def main():
    # Run evaluation
    report = await dataset.evaluate(my_server_task)
    report.print(include_input=True, include_output=True)
    
    # Save results
    import json
    with open('results.json', 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(main())
"""

    (project_path / "datasets" / "advanced_dataset.py").write_text(
        dataset_content.strip()
    )


if __name__ == "__main__":
    app()
