"""Decorator-style tests using modern evaluators."""

import mcp_eval
from mcp_eval import Expect
from mcp_eval import task, setup, teardown, parametrize
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_decorator_tests():
    """Configure mcp-eval for decorator-style tests."""
    mcp_eval.use_server("fetch")


@teardown
def cleanup_decorator_tests():
    """Cleanup after decorator tests."""
    print("Decorator tests completed")


@task("Test basic URL fetching functionality")
async def test_basic_fetch_decorator(agent: TestAgent, session: TestSession):
    """Test basic fetch functionality with modern evaluators."""
    response = await agent.generate_str("Fetch the content from https://example.com")

    # Modern evaluator approach - unified assertion API
    session.assert_that(Expect.tools.was_called("fetch"), name="fetch_tool_called", response=response)

    session.assert_that(
        Expect.content.contains("Example Domain"), name="contains_domain_text", response=response
    )

    # Deferred evaluation for tool success
    session.assert_that(Expect.tools.success_rate(min_rate=1.0, tool_name="fetch"), name="fetch_success_rate")


@task("Test tool output")
async def test_fetch_tool_output(agent: TestAgent, session: TestSession):
    """Test tool output"""
    await agent.generate_str(
        "Print the first line of the paragraph from https://example.com"
    )

    session.assert_that(
        Expect.tools.output_matches(
            tool_name="fetch",
            expected_output=r"use.*examples",
            match_type="regex",
            case_sensitive=False,
            field_path="content.0.text",
        ),
        name="fetch_output_match",
    )


@task("Test content extraction quality")
async def test_content_extraction_decorator(agent: TestAgent, session: TestSession):
    """Test quality of content extraction."""
    response = await agent.generate_str(
        "Fetch https://httpbin.org/html and summarize the main content"
    )

    # Tool usage check
    session.assert_that(Expect.tools.was_called("fetch"), name="fetch_called_for_extraction")

    # LLM judge for extraction quality
    extraction_judge = Expect.judge.llm(
        rubric="Response should demonstrate successful content extraction and provide a meaningful summary",
        min_score=0.8,
        include_input=True,
        require_reasoning=True,
    )

    session.assert_that(extraction_judge, name="extraction_quality_assessment", response=response)


@task("Test efficiency and iteration limits")
async def test_efficiency_decorator(agent: TestAgent, session: TestSession):
    """Test that fetch operations are efficient."""
    await agent.generate_str(
        "Fetch https://httpbin.org/json and extract the main information"
    )

    # Should complete efficiently
    session.assert_that(Expect.performance.max_iterations(max_iterations=3), name="efficiency_check")

    session.assert_that(Expect.tools.was_called("fetch"), name="fetch_completed")


@task("Test handling different content types")
@parametrize(
    "url,content_type,expected_indicator",
    [
        ("https://httpbin.org/json", "JSON", "json"),
        ("https://httpbin.org/html", "HTML", "html"),
        ("https://httpbin.org/xml", "XML", "xml"),
    ],
)
async def test_content_types_decorator(
    agent: TestAgent,
    session: TestSession,
    url: str,
    content_type: str,
    expected_indicator: str,
):
    """Test handling of different content types."""
    response = await agent.generate_str(
        f"Fetch {url} and identify what type of content it contains",
    )

    session.assert_that(Expect.tools.was_called("fetch"), name=f"fetch_called_for_{content_type.lower()}")

    session.assert_that(
        Expect.content.contains(expected_indicator, case_sensitive=False),
        name=f"identifies_{content_type.lower()}_content",
        response=response,
    )


@task("Test error recovery mechanisms")
async def test_error_recovery_decorator(agent: TestAgent, session: TestSession):
    """Test agent's ability to recover from fetch errors."""
    response = await agent.generate_str(
        "Try to fetch https://nonexistent-domain-12345.invalid and "
        "if that fails, fetch https://example.com instead"
    )

    # Should attempt multiple fetches
    session.assert_that(
        Expect.tools.was_called("fetch", min_times=1),  # At least one fetch attempt
        name="fetch_attempts_made",
    )

    # Should demonstrate recovery
    recovery_judge = Expect.judge.llm(
        rubric="Response should show attempt to fetch the invalid URL, recognize the error, and successfully fetch the fallback URL",
        min_score=0.8,
    )

    session.assert_that(recovery_judge, name="error_recovery_demonstration", response=response)


@task("Test path efficiency")
async def test_path_efficiency_decorator(agent: TestAgent, session: TestSession):
    """Test that agent takes an efficient path for simple fetch tasks."""
    await agent.generate_str("Fetch https://example.com and summarize the content")

    # Test basic efficiency - should complete in optimal steps
    session.assert_that(
        Expect.path.efficiency(
            expected_tool_sequence=["fetch"],
            allow_extra_steps=1,  # TODO: jerron - fix iteration count logic
            tool_usage_limits={"fetch": 1},
        ),
        name="fetch_path_efficiency",
    )
