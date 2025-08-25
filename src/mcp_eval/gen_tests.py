import asyncio
from pathlib import Path
from typing import List, Optional

import typer
from jinja2 import Environment, PackageLoader
from pydantic import BaseModel, Field
from rich.console import Console
from rich.syntax import Syntax

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.factory import _llm_factory
from mcp_agent.mcp.gen_client import gen_client

from mcp_eval.generation import ToolSchema, AssertionSpec
from mcp_eval.generation import (
    _assertion_catalog_prompt,
)  # internal helper for catalog text

app = typer.Typer()
console = Console()


class GeneratedAgentSpec(BaseModel):
    """LLM-generated AgentSpec for suite-level configuration."""

    name: str
    instruction: str
    server_names: List[str]
    provider: Optional[str] = None
    model: Optional[str] = None


class GeneratedAgent(BaseModel):
    """LLM-generated Agent for per-test override."""

    name: str
    instruction: str
    server_names: List[str]


class StructuredTestCase(BaseModel):
    test_name: str = Field(
        description="A valid Python function name for the test, e.g., 'test_search_for_weather'."
    )
    description: str = Field(
        description="A human-readable description for the @task decorator."
    )
    objective: str = Field(
        description="The natural language prompt to send to the agent to execute the test."
    )
    assertions: List[AssertionSpec] = Field(
        description="A list of structured assertions using the catalog (discriminated by 'kind')."
    )
    agent: Optional[GeneratedAgent] = Field(
        default=None,
        description="Optional per-test Agent override (Agent fields)",
    )


class GeneratedTests(BaseModel):
    suite_agent: Optional[GeneratedAgentSpec] = Field(
        default=None, description="Optional suite-level AgentSpec"
    )
    tests: List[StructuredTestCase]


async def list_tools_for_server(server_name: str) -> List[ToolSchema]:
    """Connect to a server and return a typed list of tools for the given server."""
    tools: List[ToolSchema] = []
    try:
        app = MCPApp()
        async with app.run() as running:
            async with gen_client(
                server_name, server_registry=running.context.server_registry
            ) as client:
                result = await client.list_tools()
                if hasattr(result, "tools") and isinstance(result.tools, list):
                    for tool in result.tools:
                        name = getattr(tool, "name", None)
                        if not name or name == "__human_input__":
                            continue
                        description: Optional[str] = getattr(tool, "description", None)
                        input_schema = (
                            getattr(tool, "inputSchema", None)
                            or getattr(tool, "input_schema", None)
                            or getattr(tool, "input", None)
                        )
                        tools.append(
                            ToolSchema(
                                name=name,
                                description=description,
                                input_schema=input_schema,
                            )
                        )
    except Exception as e:
        console.print(
            f"[bold red]Error:[/] Could not list tools for server '{server_name}': {e}"
        )
    return tools


def find_gitignore(start_path: Path) -> Path:
    """Walk up the directory tree to find .gitignore file."""
    current = start_path.resolve()

    # Walk up the tree until we find .gitignore or reach root
    while current != current.parent:
        gitignore_path = current / ".gitignore"
        if gitignore_path.exists():
            return gitignore_path
        current = current.parent

    # Check root directory
    root_gitignore = current / ".gitignore"
    if root_gitignore.exists():
        return root_gitignore

    return None


def update_gitignore(output_path: Path):
    """Updates .gitignore to exclude test-reports directory if .gitignore exists."""
    # Find .gitignore by walking up from the output file location
    gitignore_path = find_gitignore(output_path.parent if output_path else Path.cwd())

    if not gitignore_path:
        console.print(
            "[yellow]No .gitignore found in project tree, skipping update[/yellow]"
        )
        return

    # Lines to ensure are in .gitignore
    gitignore_entries = ["\n# MCP-Eval test reports", "test-reports/"]

    # Read existing .gitignore
    with open(gitignore_path, "r") as f:
        content = f.read()

    # Check if test-reports is already ignored
    if "test-reports/" in content:
        return  # Already ignored

    # Add our entries
    if content and not content.endswith("\n"):
        content += "\n"

    for entry in gitignore_entries:
        content += entry + "\n"

    # Write updated content
    with open(gitignore_path, "w") as f:
        f.write(content)

    console.print(f"[green]Updated {gitignore_path} to exclude test-reports/[/green]")


def get_generation_prompt(tools: List[ToolSchema]) -> str:
    """Create a structured prompt for the LLM to generate test cases from typed tool specs."""
    import json

    tool_payload = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.input_schema or {},
        }
        for t in tools
    ]

    # Provide assertion catalog and few-shot examples to guide the model
    catalog_text = _assertion_catalog_prompt()

    few_shot = {
        "tests": [
            # Basic content + tool usage + judge
            {
                "test_name": "test_fetch_example",
                "description": "Fetch example.com and verify content and tool usage",
                "objective": "Fetch https://example.com and summarize",
                "assertions": [
                    {
                        "kind": "response_contains",
                        "text": "Example Domain",
                        "case_sensitive": False,
                    },
                    {"kind": "tool_was_called", "tool_name": "fetch", "min_times": 1},
                    {
                        "kind": "llm_judge",
                        "rubric": "Response should accurately summarize the page contents",
                        "min_score": 0.8,
                    },
                    {
                        "kind": "tool_sequence",
                        "sequence": ["fetch"],
                        "allow_other_calls": False,
                    },
                ],
            },
            # Structured tool output matching + performance
            {
                "test_name": "test_tool_output_check",
                "description": "Ensure tool output returns text content",
                "objective": "Print the first paragraph from https://example.com",
                "assertions": [
                    {
                        "kind": "tool_output_matches",
                        "tool_name": "fetch",
                        "expected_output": {"content": [{"type": "text"}]},
                        "match_type": "partial",
                        "case_sensitive": True,
                    },
                    {"kind": "max_iterations", "max_iterations": 3},
                    {"kind": "response_time_under", "ms": 15000},
                ],
            },
            # Argument verification for tools
            {
                "test_name": "test_called_with_url",
                "description": "Verify fetch is called with expected URL argument",
                "objective": "Fetch https://httpbin.org/json and summarize",
                "assertions": [
                    {"kind": "tool_was_called", "tool_name": "fetch", "min_times": 1},
                    {
                        "kind": "tool_called_with",
                        "tool_name": "fetch",
                        "arguments": {"url": "https://httpbin.org/json"},
                    },
                ],
            },
            # Multi-tool flow and sequence
            {
                "test_name": "test_multi_tool_sequence",
                "description": "Use two tools in sequence and verify the path",
                "objective": "Fetch https://example.com then write a short summary to a file named summary.md",
                "assertions": [
                    {
                        "kind": "tool_sequence",
                        "sequence": ["fetch", "write"],
                        "allow_other_calls": False,
                    },
                    {"kind": "tool_was_called", "tool_name": "write", "min_times": 1},
                ],
            },
            # Edge case / error handling
            {
                "test_name": "test_error_handling_invalid_url",
                "description": "Invalid URL should be handled gracefully",
                "objective": "Try to fetch https://this-does-not-exist-999999.tld",
                "assertions": [
                    {"kind": "tool_was_called", "tool_name": "fetch", "min_times": 1},
                    {
                        "kind": "llm_judge",
                        "rubric": "Response should acknowledge the error and explain it appropriately",
                        "min_score": 0.8,
                    },
                ],
            },
        ]
    }

    guidance = {
        "instructions": (
            "You are an expert AI Test Engineer. Generate a diverse suite of high-quality tests for the MCP server's tools. "
            "Include basic functionality, edge cases, argument checks, tool output checks, sequences, timing, and quality via LLM judges. "
            "Use the assertion catalog below and the discriminated 'kind' field to structure assertions."
        ),
        "tools": tool_payload,
        "assertion_catalog": catalog_text,
        "requirements": [
            "At least one test per tool where applicable",
            "Include edge/error cases (invalid inputs, non-existent URLs, etc.)",
            "Add argument validation checks using tool_called_with when meaningful",
            "Use tool_output_matches for structured outputs",
            "Include path/sequence checks for multi-tool workflows when appropriate",
        ],
        "suite_agent_spec": {
            "description": "Propose a single suite-level AgentSpec appropriate for these tools",
            "schema": {
                "name": "str",
                "instruction": "str",
                "server_names": "List[str]",
                "provider": "Optional[str]",
                "model": "Optional[str]",
            },
        },
        "per_test_agent": {
            "description": "Optionally propose per-test Agent overrides when beneficial (e.g., specialized instructions)",
            "schema": {
                "name": "str",
                "instruction": "str",
                "server_names": "List[str]",
            },
            "when_to_use": [
                "When a test benefits from custom instruction or server selection",
                "When experimenting with different tool invocation strategies",
            ],
        },
        "few_shot_examples": few_shot,
        "output_schema": GeneratedTests.model_json_schema(),
    }

    return (
        "Design a JSON object adhering to the output_schema using the tools and assertion catalog.\n"
        + json.dumps(guidance, indent=2)
    )


async def generate_tests_from_llm(
    tools: List[ToolSchema], *, batch_size: int = 20
) -> GeneratedTests:
    """Use an LLM to generate tests following the GeneratedTests schema, from typed tool specs."""
    prompt = get_generation_prompt(tools)

    app = MCPApp()
    async with app.run() as running:
        agent = Agent(
            name="mcpeval-test-generator",
            instruction="You generate high-quality MCP server tests.",
            server_names=[],
            context=running.context,
        )
        llm_factory = _llm_factory(provider=None, model=None, context=running.context)
        llm = llm_factory(agent)

        # If there are many tools, split into manageable batches for better focus
        BATCH_SIZE = max(1, int(batch_size))
        aggregated: List[StructuredTestCase] = []
        seen_names: set[str] = set()

        def _uniquify(name: str) -> str:
            if name not in seen_names:
                return name
            i = 2
            while f"{name}_{i}" in seen_names:
                i += 1
            return f"{name}_{i}"

        if len(tools) <= BATCH_SIZE:
            model = await llm.generate_structured(prompt, response_model=GeneratedTests)
            for t in model.tests:
                t.test_name = _uniquify(t.test_name)
                seen_names.add(t.test_name)
            aggregated.extend(model.tests)
        else:
            total_batches = (len(tools) + BATCH_SIZE - 1) // BATCH_SIZE
            console.print(
                f"[cyan]Large tool list detected:[/] {len(tools)} tools. Generating in {total_batches} batches of up to {BATCH_SIZE} tools each..."
            )
            for i in range(0, len(tools), BATCH_SIZE):
                batch = tools[i : i + BATCH_SIZE]
                console.print(
                    f"[cyan]Batch {i // BATCH_SIZE + 1}/{total_batches}:[/] {len(batch)} tools"
                )
                batch_prompt = get_generation_prompt(batch)
                model = await llm.generate_structured(
                    batch_prompt, response_model=GeneratedTests
                )
                for t in model.tests:
                    t.test_name = _uniquify(t.test_name)
                    seen_names.add(t.test_name)
                aggregated.extend(model.tests)
                console.print(
                    f"  [green]✓[/] Generated {len(model.tests)} tests (cumulative: {len(aggregated)})"
                )

        # Merge suite agent from the last model if available; otherwise None
        suite_agent = None
        try:
            # If we had a single batch, model is defined above. If multiple, take the last one.
            # We won't track each model separately for brevity here.
            if "model" in locals() and hasattr(model, "suite_agent"):
                suite_agent = model.suite_agent
        except Exception:
            pass

        return GeneratedTests(suite_agent=suite_agent, tests=aggregated)


@app.command()
def generate(
    server_name: str = typer.Argument(
        ..., help="The name of the MCP server to generate tests for."
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="The path to save the generated test file."
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        help=(
            "Maximum number of tools to include per LLM generation batch. "
            "If a server exposes many tools, they are split into batches to keep prompts focused and reliable."
        ),
    ),
):
    """
    Generates a Python test file for a given MCP server by introspecting its tools.
    """
    console.print(f"Generating tests for server: [bold cyan]{server_name}[/bold cyan]")

    tools = asyncio.run(list_tools_for_server(server_name))

    if not tools:
        console.print(
            "[bold red]No tools found or server is unavailable. Cannot generate tests.[/bold red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Found {len(tools)} tools[/green] — asking LLM to generate test cases (batch_size={batch_size})..."
    )

    try:
        generated_tests = asyncio.run(
            generate_tests_from_llm(tools, batch_size=batch_size)
        )
    except Exception as e:
        console.print(f"[bold red]Failed to generate tests from LLM: {e}[/bold red]")
        raise typer.Exit(1)

    env = Environment(loader=PackageLoader("mcp_eval", "templates"))
    template = env.get_template("test_file.py.j2")

    output_code = template.render(
        server_name=server_name,
        tests=generated_tests.tests,
        suite_agent=generated_tests.suite_agent,
    )
    console.print(f"[cyan]Generated {len(generated_tests.tests)} tests.[/cyan]")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_code)
        console.print(
            f"\n[bold green]Successfully generated tests and saved to {output_file}[/bold green]"
        )

        # Update .gitignore to exclude test-reports
        update_gitignore(Path(output_file))
    else:
        console.print("\n[bold green]Generated Test Code:[/bold green]")
        syntax = Syntax(
            output_code, "python", theme="solarized-dark", line_numbers=True
        )
        console.print(syntax)

        # Update .gitignore in current directory
        update_gitignore(Path.cwd())

    # Print summary of objectives
    console.print("\n[bold]Generated test objectives:[/bold]")
    for t in generated_tests.tests[:30]:
        console.print(f" - [green]{t.test_name}[/green]: {t.description}")
    if len(generated_tests.tests) > 30:
        console.print(f" ... and {len(generated_tests.tests) - 30} more")


if __name__ == "__main__":
    app()
