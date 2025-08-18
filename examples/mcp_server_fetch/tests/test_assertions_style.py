"""Legacy assertions-style tests with explicit session passing."""

from mcp_eval import task, setup, Expect
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_assertions_tests():
    """Prefer defining server_names on Agent/AgentSpec via config; no per-test server selection needed."""


@task("Test basic fetch with legacy assertions")
async def test_basic_fetch_assertions(agent, session):
    """Test basic URL fetching using legacy assertion style."""
    response = await agent.generate_str("Fetch https://example.com")

    # Prefer modern Expect-based assertions
    await session.assert_that(Expect.tools.was_called("fetch"))
    await session.assert_that(
        Expect.content.contains("Example Domain"), response=response
    )
    await session.assert_that(
        Expect.tools.success_rate(min_rate=1.0, tool_name="fetch")
    )


@task("Test fetch error handling with assertions")
async def test_fetch_error_assertions(agent, session):
    """Test error handling using legacy assertions."""
    response = await agent.generate_str("Fetch https://invalid-domain-xyz-123.com")

    # Tool should be called but might fail
    await session.assert_that(Expect.tools.was_called("fetch"))
    await session.assert_that(
        Expect.content.contains("error", case_sensitive=False), response=response
    )


@task("Test response time requirements")
async def test_fetch_performance_assertions(agent, session):
    """Test performance requirements using legacy assertions."""
    await agent.generate_str("Quickly fetch https://httpbin.org/json")

    await session.assert_that(Expect.tools.was_called("fetch"))
    await session.assert_that(Expect.performance.response_time_under(10000))
    await session.assert_that(Expect.performance.max_iterations(3))


@task("Test multiple fetch calls")
async def test_multiple_fetch_assertions(agent, session):
    """Test multiple URL fetching."""
    await agent.generate_str(
        "Fetch content from both https://example.com and https://httpbin.org/html"
    )

    await session.assert_that(Expect.tools.count("fetch", 2))
    await session.assert_that(Expect.tools.success_rate(0.8, tool_name="fetch"))


@task("Test content format detection")
async def test_content_format_assertions(agent, session):
    """Test content format handling."""
    response = await agent.generate_str(
        "Fetch https://httpbin.org/json and tell me what format it's in"
    )

    await session.assert_that(Expect.tools.was_called("fetch"))
    await session.assert_that(
        Expect.content.contains("json", case_sensitive=False), response=response
    )
    await session.assert_that(
        Expect.judge.llm(
            "Response correctly identifies the content as JSON format", min_score=0.8
        ),
        response=response,
    )


@task("Test tool output return text content")
async def test_tool_output_assetion(agent, session):
    """Test tool output"""
    await agent.generate_str(
        "Print the first line of the paragraph in https://example.com"
    )
    await session.assert_that(
        Expect.tools.output_matches(
            tool_name="fetch",
            expected_output={"content": {"0": {"type": "text"}}},
            match_type="partial",
        )
    )


@task("Test path efficiency with assertions")
async def test_path_efficiency_assertions(agent: TestAgent, session: TestSession):
    """Test efficient path using legacy assertions."""
    await agent.generate_str("Fetch https://httpbin.org/json and extract the data")

    # Test path efficiency using modern Expect
    await session.assert_that(
        Expect.path.efficiency(
            expected_tool_sequence=["fetch"],
            allow_extra_steps=1,
            tool_usage_limits={"fetch": 1},
            penalize_backtracking=True,
        )
    )
