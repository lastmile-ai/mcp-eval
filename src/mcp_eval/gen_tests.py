import typer
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from pydantic import BaseModel, Field
from typing import List
import asyncio
from jinja2 import Environment, PackageLoader
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer()
console = Console()


class TestCase(BaseModel):
    test_name: str = Field(
        description="A valid Python function name for the test, e.g., 'test_search_for_weather'."
    )
    description: str = Field(
        description="A human-readable description for the @task decorator."
    )
    objective: str = Field(
        description="The natural language prompt to send to the agent to execute the test."
    )
    assertions: List[str] = Field(
        description="A list of keywords that should be present in the response for a basic 'contains' assertion."
    )


class GeneratedTests(BaseModel):
    tests: List[TestCase]


async def list_tools_for_server(server_name: str) -> str:
    """Connects to a server and returns a formatted string of its tools."""
    tool_info = []
    try:
        temp_app = MCPApp()
        await temp_app.initialize()
        temp_agent = Agent(
            name="mcpeval-test-generator",
            server_names=[server_name],
            context=temp_app.context,
        )
        await temp_agent.initialize()
        tools_result = await temp_agent.list_tools()

        for tool in tools_result.tools:
            if tool.name == "__human_input__":
                continue
            tool_info.append(f"- Tool: {tool.name}\n  Description: {tool.description}")

        await temp_agent.shutdown()
        await temp_app.cleanup()
    except Exception as e:
        console.print(
            f"[bold red]Error:[/] Could not list tools for server '{server_name}': {e}"
        )

    return "\n".join(tool_info)


def get_generation_prompt(tool_descriptions: str) -> str:
    """Creates the prompt for the LLM to generate test cases."""
    return f"""
    You are an expert AI Test Engineer. Your task is to generate a suite of test cases for an MCP server based on its available tools.

    Here is the list of tools available on the server:
    ---
    {tool_descriptions}
    ---

    Please generate a diverse set of test cases that cover the following aspects:
    1.  **Basic Functionality**: Simple tests for each tool.
    2.  **Edge Cases**: Tests with unusual or tricky inputs.
    3.  **Combination of Tools**: Tests that might require the agent to use multiple tools in sequence.

    For each test case, provide:
    - A valid Python function name (e.g., 'test_search_for_weather').
    - A human-readable description for the test.
    - A natural language objective (prompt) to give to the agent.
    - A list of simple keywords that are expected in the agent's response, which will be used for 'contains' assertions.

    Please format your output as a JSON object that conforms to the following Pydantic model:
    {GeneratedTests.schema_json(indent=2)}
    """


async def generate_tests_from_llm(
    server_name: str, tool_descriptions: str
) -> GeneratedTests:
    """Uses an LLM to generate test cases."""
    prompt = get_generation_prompt(tool_descriptions)

    # We need an agent to generate the tests
    app = MCPApp()
    await app.initialize()
    # We need to find the first available openai llm provider
    llm_provider_name = next(
        (p.name for p in app.config.llm_providers if p.provider == "openai"), None
    )

    if not llm_provider_name:
        raise ValueError("No OpenAI LLM provider found in config to generate tests.")

    def llm_factory(agent):
        return app.context.llm_provider.get_llm(llm_provider_name, agent=agent)

    generator_agent = Agent(name="test-case-generator", context=app.context)
    await generator_agent.attach_llm(llm_factory=llm_factory)

    generated_model = await generator_agent.llm.generate_structured(
        prompt, response_model=GeneratedTests
    )
    await app.cleanup()
    return generated_model


@app.command()
def generate(
    server_name: str = typer.Argument(
        ..., help="The name of the MCP server to generate tests for."
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="The path to save the generated test file."
    ),
):
    """
    Generates a Python test file for a given MCP server by introspecting its tools.
    """
    console.print(f"Generating tests for server: [bold cyan]{server_name}[/bold cyan]")

    tool_descriptions = asyncio.run(list_tools_for_server(server_name))

    if not tool_descriptions:
        console.print(
            "[bold red]No tools found or server is unavailable. Cannot generate tests.[/bold red]"
        )
        raise typer.Exit(1)

    console.print("[green]Found tools...[/green] Asking LLM to generate test cases...")

    try:
        generated_tests = asyncio.run(
            generate_tests_from_llm(server_name, tool_descriptions)
        )
    except Exception as e:
        console.print(f"[bold red]Failed to generate tests from LLM: {e}[/bold red]")
        raise typer.Exit(1)

    env = Environment(loader=PackageLoader("mcp_eval", "templates"))
    template = env.get_template("test_file.py.j2")

    output_code = template.render(server_name=server_name, tests=generated_tests.tests)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_code)
        console.print(
            f"\n[bold green]Successfully generated tests and saved to {output_file}[/bold green]"
        )
    else:
        console.print("\n[bold green]Generated Test Code:[/bold green]")
        syntax = Syntax(
            output_code, "python", theme="solarized-dark", line_numbers=True
        )
        console.print(syntax)


if __name__ == "__main__":
    app()
