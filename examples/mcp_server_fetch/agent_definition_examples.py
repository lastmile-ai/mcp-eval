"""
Comprehensive examples showing all ways to define agents in mcp-eval.

This file demonstrates:
1. Using AgentSpec from config files
2. Programmatic Agent creation
3. Programmatic AugmentedLLM creation
4. Default agent (fallback)
5. Per-test agent overrides
6. Using factory functions from mcp-agent
"""

import asyncio
from pathlib import Path
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.factory import create_llm, agent_from_spec
from mcp_eval import task, Expect, test_session
from mcp_eval.core import with_agent
from mcp_eval.config import use_agent_factory
from mcp_eval.session import TestAgent, TestSession


# =============================================================================
# Method 1: AgentSpec defined in configuration files
# =============================================================================


@task("Test with config-defined agent")
@with_agent("default")  # Reference agent by name from config
async def test_config_agent(agent: TestAgent, session: TestSession):
    """
    Uses an agent defined in configuration files.

    Agents can be defined in:
    - mcpeval.yaml (under 'agents.definitions' section)
    - mcp-agent.config.yaml (under 'agents' section)
    - .claude/agents/*.yaml (discovered subagents)
    - .mcp-agent/agents/*.yaml
    - package-provided examples via `mcp_eval.data.sample` (use `mcp-eval init --template sample`)

    Example mcpeval.yaml:
    ```yaml
    agents:
      definitions:
        - name: default
          instruction: You fetch URLs and provide summaries
          server_names: ["fetch"]
          model: claude-sonnet-4-0  # Or omit to use ModelSelector
    ```

    The @with_agent("default") decorator references the agent by name.
    """
    response = await agent.generate_str("Fetch https://example.com")
    await session.assert_that(
        Expect.tools.was_called("fetchCallIJustMadeUp"), name="fetch_called"
    )
    await session.assert_that(
        Expect.content.contains("Example Domain"),
        response=response,
        name="contains_domain",
    )


# =============================================================================
# Method 2: Programmatic AgentSpec
# =============================================================================


@task("Test with programmatic AgentSpec")
@with_agent(
    AgentSpec(
        name="ProgrammaticFetcher",
        instruction="You are a URL fetching specialist. Be concise.",
        server_names=["fetch"],
        # Optional: specify provider/model at the spec level
        provider="anthropic",
        # model not specified - will use ModelSelector
    )
)
async def test_agent_spec(agent: TestAgent, session: TestSession):
    """
    Define an AgentSpec programmatically for this specific test.

    AgentSpec is a declarative configuration that includes:
    - name: Identifier for the agent
    - instruction: System prompt for the agent
    - server_names: MCP servers to connect to
    - provider: Optional LLM provider (defaults to settings)
    - model: Optional model override (uses ModelSelector if not set)

    The @with_agent decorator creates the agent from this spec
    for this test only, not affecting other tests.
    """
    response = await agent.generate_str("What is at https://httpbin.org/json?")
    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_tool_used")
    await session.assert_that(
        Expect.content.contains("json", case_sensitive=False),
        response=response,
        name="mentions_json",
    )


# =============================================================================
# Method 3: Programmatic Agent instance
# =============================================================================


@task("Test with Agent instance")
@with_agent(
    lambda: Agent(
        name="DirectAgent",
        instruction="You fetch web content and analyze it thoroughly.",
        server_names=["fetch"],
        # context=None is the default, TestSession will set it properly
        # Note: Agent doesn't take provider/model directly,
        # those are configured via settings or when attaching LLM
    )
)
async def test_agent_instance(agent: TestAgent, session: TestSession):
    """
    Create an Agent instance directly for this test.

    The Agent class is the core runtime object that:
    - Manages connections to MCP servers
    - Handles tool execution
    - Maintains conversation context

    Note: Agent doesn't take provider/model directly.
    Those are configured via settings or when attaching an LLM.

    The @with_agent decorator ensures this agent is used
    only for this specific test.
    """
    response = await agent.generate_str("Analyze the structure of https://example.com")

    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_executed")
    await session.assert_that(
        Expect.judge.llm(
            "Response provides structural analysis of the webpage", min_score=0.7
        ),
        response=response,
        name="quality_check",
    )


# =============================================================================
# Method 4: Programmatic AugmentedLLM
# =============================================================================


@task("Test with factory-created AugmentedLLM")
@with_agent(
    lambda: create_llm(
        agent_name="FactoryLLM",
        instruction="You are a helpful web content fetcher.",
        server_names=["fetch"],
        provider="anthropic",
        # model not specified - will use ModelSelector
    )
)
async def test_augmented_llm(agent: TestAgent, session: TestSession):
    """Uses the factory-created AugmentedLLM."""
    response = await agent.generate_str(
        "Get me the content from https://httpbin.org/html"
    )

    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_used")
    await session.assert_that(
        Expect.content.contains("html", case_sensitive=False),
        response=response,
        name="html_mentioned",
    )


# =============================================================================
# Method 5: Agent Factory (for parallel test safety)
# =============================================================================


def create_test_agent():
    """
    Factory function to create a fresh agent for each test session.

    This pattern is crucial for parallel test execution:
    - Each test gets its own agent instance
    - No shared state between parallel tests
    - Thread-safe test execution

    Two ways to use factories:
    1. Global default: use_agent_factory(create_test_agent)
    2. Per-test: @with_agent(create_test_agent())

    Note: The TestSession will set the proper context when the agent is used.
    """
    return Agent(
        name="FactoryAgent",
        instruction="Fetch and summarize web content efficiently.",
        server_names=["fetch"],
        # context=None by default, TestSession will set it
    )


# Example 5a: Using factory for a specific test
@task("Test with agent factory (per-test)")
@with_agent(create_test_agent)  # Creates new instance for this test
async def test_agent_factory(agent: TestAgent, session: TestSession):
    """
    This test gets a fresh agent instance from the factory.

    The factory is called when the test starts, ensuring
    a clean agent instance even in parallel execution.
    """
    await agent.generate_str("Summarize https://example.com in one sentence")

    await session.assert_that(
        Expect.tools.was_called("fetch"), name="fetch_called"
    )
    await session.assert_that(
        Expect.performance.max_iterations(2), name="efficient_execution"
    )


# Example 5b: Using factory as global default
def setup_global_factory():
    """
    Set up a global factory for all tests that don't specify an agent.

    This would typically be called once at the start of your test suite:
    ```python
    from mcp_eval.config import use_agent_factory

    def my_agent_factory():
        return Agent(name="GlobalFactory", ...)

    use_agent_factory(my_agent_factory)
    ```

    Then all tests without @with_agent would use this factory.
    """
    # Note: We're not actually calling this in the example
    # to avoid affecting other tests
    use_agent_factory(create_test_agent)


# =============================================================================
# Method 6: Per-test agent override using @with_agent
# =============================================================================


@task("Test with per-test agent override")
@with_agent(
    # Using a lambda factory to defer Agent creation
    lambda: Agent(
        name="CustomTestAgent",
        instruction="You are specialized for this specific test.",
        server_names=["fetch"],
    )
)
async def test_with_custom_agent(agent: TestAgent, session: TestSession):
    """This test uses its own custom agent, overriding any global setting."""
    await agent.generate_str("Check what's at https://httpbin.org/json")

    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_used")
    # Verify the agent name to confirm override worked
    assert agent.agent.name == "CustomTestAgent"


# =============================================================================
# Method 7: Using test_session context manager directly
# =============================================================================


async def test_with_session_context():
    """Direct use of test_session with inline agent definition."""

    # Create agent inline
    custom_agent = Agent(
        name="InlineAgent",
        instruction="Fetch URLs and be very detailed.",
        server_names=["fetch"],
        # context will be set by test_session when it's used
    )

    async with test_session("inline_test", agent=custom_agent) as agent:
        response = await agent.generate_str("Tell me about https://example.com")

        # Can use session through agent.session
        await agent.session.assert_that(
            Expect.tools.was_called("fetch"), name="fetch_executed"
        )
        await agent.session.assert_that(
            Expect.content.contains("Example Domain"),
            response=response,
            name="has_content",
        )


# =============================================================================
# Method 8: Default fallback (no explicit configuration)
# =============================================================================


@task("Test with default fallback agent")
async def test_default_fallback(agent: TestAgent, session: TestSession):
    """
    If no agent is configured, mcp-eval creates a minimal default.
    This requires server_names to be set in default_servers config.
    """
    # This would use the fallback agent with minimal configuration
    await agent.generate_str("Hello, can you fetch https://example.com?")

    # Even the default agent should work if servers are configured
    await session.assert_that(Expect.tools.was_called("fetchIMadeUpTheToolName"), name="fetch_attempted")


# =============================================================================
# Method 9: Loading AgentSpec from file programmatically
# =============================================================================


async def test_load_agent_from_file():
    """Load an AgentSpec from a YAML/JSON/Markdown file."""
    from mcp_agent.workflows.factory import load_agent_specs_from_file

    # Create a sample agent spec file
    agent_yaml = """
name: FileLoadedAgent
instruction: |
  You are an expert at fetching and analyzing web content.
  Always provide clear, structured responses.
server_names:
  - fetch
"""

    # Write to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(agent_yaml)
        spec_file = f.name

    try:
        # Load the spec
        specs = load_agent_specs_from_file(spec_file)
        if specs:
            agent_spec = specs[0]

            # Use it in a test session
            async with test_session("file_loaded_test", agent=agent_spec) as agent:
                await agent.generate_str("Fetch https://example.com")

                await agent.session.assert_that(
                    Expect.tools.was_called("fetch"), name="fetch_called"
                )
    finally:
        # Cleanup
        Path(spec_file).unlink(missing_ok=True)


# =============================================================================
# Method 10: Using agent_from_spec with context
# =============================================================================


async def test_agent_from_spec_with_context():
    """Create an agent from spec using the factory with explicit context."""

    # Create a spec
    spec = AgentSpec(
        name="ContextualAgent",
        instruction="Fetch and analyze with context awareness.",
        server_names=["fetch"],
    )

    # In a real scenario, you might have a context with special configuration
    # For this example, we'll use the test session's context
    async with test_session("contextual_test") as test_agent:
        # The session already has a context we can reference
        context = test_agent.session.app.context if test_agent.session.app else None

        if context:
            # Create agent with explicit context
            contextual_agent = agent_from_spec(spec, context=context)

            # Note: This is more advanced usage - typically you'd just use
            # the simpler methods above
            await contextual_agent.generate_str("Analyze https://example.com")

            await test_agent.session.assert_that(
                Expect.tools.was_called("fetch"), name="fetch_with_context"
            )


# =============================================================================
# Run examples
# =============================================================================


async def run_all_examples():
    """Run all the example tests to demonstrate different agent definition methods."""
    print("Running agent definition examples...\n")

    # Note: In practice, you'd use mcp-eval CLI or pytest to run these
    # This is just for demonstration

    examples = [
        test_config_agent,
        test_agent_spec,
        test_agent_instance,
        test_augmented_llm,
        test_agent_factory,
        test_with_custom_agent,
        test_with_session_context,
        test_default_fallback,
        test_load_agent_from_file,
        test_agent_from_spec_with_context,
    ]

    for example in examples:
        try:
            print(f"Running: {example.__name__}")
            if asyncio.iscoroutinefunction(example):
                await example()
            else:
                # For decorated tests, we need to handle them differently
                # In real usage, the test runner handles this
                pass
            print("  ✓ Completed\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")


if __name__ == "__main__":
    # This would normally be run via: mcp-eval run agent_definition_examples.py
    print("Note: Run these examples using 'mcp-eval run' command")
    print("Example: mcp-eval run examples/agent_definition_examples.py")
