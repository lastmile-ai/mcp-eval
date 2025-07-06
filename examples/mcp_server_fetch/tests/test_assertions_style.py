"""Legacy assertions-style tests with explicit session passing."""

import mcp_eval
from mcp_eval import task, setup
from mcp_eval import assertions


@setup
def configure_assertions_tests():
    """Configure mcp-eval for assertions-style tests."""
    mcp_eval.use_server("fetch")


@task("Test basic fetch with legacy assertions")
async def test_basic_fetch_assertions(agent, session):
    """Test basic URL fetching using legacy assertion style."""
    response = await agent.generate_str("Fetch https://example.com")

    # Legacy assertions with explicit session passing
    assertions.tool_was_called(session, "fetch")
    assertions.contains(session, response, "Example Domain")
    assertions.tool_call_succeeded(session, "fetch")


@task("Test fetch error handling with assertions")
async def test_fetch_error_assertions(agent, session):
    """Test error handling using legacy assertions."""
    response = await agent.generate_str("Fetch https://invalid-domain-xyz-123.com")

    # Tool should be called but might fail
    assertions.tool_was_called(session, "fetch")
    assertions.contains(session, response, "error", case_sensitive=False)


@task("Test response time requirements")
async def test_fetch_performance_assertions(agent, session):
    """Test performance requirements using legacy assertions."""
    await agent.generate_str("Quickly fetch https://httpbin.org/json")

    assertions.tool_was_called(session, "fetch")
    assertions.response_time_under(session, 10000)  # 10 seconds max
    assertions.completed_within(session, 3)  # 3 iterations max


@task("Test multiple fetch calls")
async def test_multiple_fetch_assertions(agent, session):
    """Test multiple URL fetching."""
    await agent.generate_str(
        "Fetch content from both https://example.com and https://httpbin.org/html"
    )

    assertions.tool_call_count(session, "fetch", 2)
    assertions.tool_success_rate(session, 0.8, "fetch")  # 80% success rate minimum


@task("Test content format detection")
async def test_content_format_assertions(agent, session):
    """Test content format handling."""
    response = await agent.generate_str(
        "Fetch https://httpbin.org/json and tell me what format it's in"
    )

    assertions.tool_was_called(session, "fetch")
    assertions.contains(session, response, "json", case_sensitive=False)

    # Use LLM judge for complex evaluation
    await assertions.judge(
        session,
        response,
        "Response correctly identifies the content as JSON format",
        min_score=0.8,
    )
