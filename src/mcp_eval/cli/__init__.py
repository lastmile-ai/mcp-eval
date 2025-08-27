"""Enhanced command-line interface for MCP-Eval."""

import typer
from pathlib import Path
from rich.console import Console

from mcp_eval.runner import app as runner_app
from mcp_eval.cli.generator import (
    add_app,
    init_project_cli,
    run_generator_cli,
    update_tests_cli,
    server_app,
    agent_app,
)
from mcp_eval.cli.list_command import app as list_app
from mcp_eval.cli.validate import validate
from mcp_eval.cli.doctor import doctor
from mcp_eval.cli.issue import issue

app = typer.Typer(help="MCP-Eval: Comprehensive testing framework for MCP servers")
console = Console()

# Subcommands
app.add_typer(runner_app, name="run", help="Run tests")
app.add_typer(list_app, name="list", help="List configured resources")
app.add_typer(server_app, name="server", help="Manage MCP servers (add, list)")
app.add_typer(agent_app, name="agent", help="Manage agents (add, list)")
app.add_typer(add_app, name="add", help="Add servers or agents (legacy)")


@app.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata

        version = importlib.metadata.version("mcpevals")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (development)"
    console.print(f"MCP-Eval {version}")


# Add init, generate and update commands from generator to top level
app.command("init")(init_project_cli)
app.command("generate")(run_generator_cli)
app.command("update")(update_tests_cli)
app.command("validate")(validate)
app.command("doctor")(doctor)
app.command("issue")(issue)


# The generate command is now properly registered above


# Template functions removed - handled by new generator flow
def _create_basic_template(project_path: Path):
    """[Deprecated] Create basic template files."""
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
  model: "claude-sonnet-4-0"
  min_score: 0.8

# Reporting configuration
reporting:
  formats: ["json", "markdown"]
  output_dir: "./reports"
"""

    (project_path / "mcpeval.yaml").write_text(config_content.strip())

    # Example test file
    test_content = """
from mcp_eval import task, Expect, setup, ToolWasCalled, ResponseContains

@setup
def configure_tests():
    pass

@task("Basic functionality test")
async def test_basic_functionality(agent, session):
    \"\"\"Test basic server functionality.\"\"\"
    response = await agent.generate_str("Perform a basic operation")
    await session.assert_that(Expect.content.contains("result"), response=response)
    await session.assert_that(Expect.tools.was_called("basic_tool"))

@task("Error handling test")
async def test_error_handling(agent, session):
    \"\"\"Test server error handling.\"\"\"
    response = await agent.generate_str("Perform an invalid operation")
    await session.assert_that(Expect.content.contains("error"), response=response)
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
