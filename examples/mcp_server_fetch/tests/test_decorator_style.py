"""Decorator-style tests using modern evaluators."""

import mcp_eval
from mcp_eval import task, setup, teardown, parametrize
from mcp_eval import (
    ToolWasCalled,
    ResponseContains,
    LLMJudge,
    ToolSuccessRate,
    MaxIterations,
)


@setup
def configure_decorator_tests():
    """Configure mcp-eval for decorator-style tests."""
    mcp_eval.use_server("fetch")


@teardown
def cleanup_decorator_tests():
    """Cleanup after decorator tests."""
    print("Decorator tests completed")


@task("Test basic URL fetching functionality")
async def test_basic_fetch_decorator(agent, session):
    """Test basic fetch functionality with modern evaluators."""
    response = await agent.generate_str("Fetch the content from https://example.com")

    # Modern evaluator approach - immediate evaluation
    session.evaluate_now(ToolWasCalled("fetch"), response, "fetch_tool_called")

    session.evaluate_now(
        ResponseContains("Example Domain"), response, "contains_domain_text"
    )

    # Deferred evaluation for tool success
    session.add_deferred_evaluator(
        ToolSuccessRate(min_rate=1.0, tool_name="fetch"), "fetch_success_rate"
    )


@task("Test content extraction quality")
async def test_content_extraction_decorator(agent, session):
    """Test quality of content extraction."""
    response = await agent.generate_str(
        "Fetch https://httpbin.org/html and summarize the main content"
    )

    # Tool usage check
    session.add_deferred_evaluator(
        ToolWasCalled("fetch"), "fetch_called_for_extraction"
    )

    # LLM judge for extraction quality
    extraction_judge = LLMJudge(
        rubric="Response should demonstrate successful content extraction and provide a meaningful summary",
        min_score=0.8,
        include_input=True,
        require_reasoning=True,
    )

    await session.evaluate_now_async(
        extraction_judge, response, "extraction_quality_assessment"
    )


@task("Test efficiency and iteration limits")
async def test_efficiency_decorator(agent, session):
    """Test that fetch operations are efficient."""
    await agent.generate_str(
        "Fetch https://httpbin.org/json and extract the main information"
    )

    # Should complete efficiently
    session.add_deferred_evaluator(MaxIterations(max_iterations=3), "efficiency_check")

    session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_completed")


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
    agent, session, url, content_type, expected_indicator
):
    """Test handling of different content types."""
    response = await agent.generate_str(
        f"Fetch {url} and identify what type of content it contains"
    )

    session.add_deferred_evaluator(
        ToolWasCalled("fetch"), f"fetch_called_for_{content_type.lower()}"
    )

    session.evaluate_now(
        ResponseContains(expected_indicator, case_sensitive=False),
        response,
        f"identifies_{content_type.lower()}_content",
    )


@task("Test error recovery mechanisms")
async def test_error_recovery_decorator(agent, session):
    """Test agent's ability to recover from fetch errors."""
    response = await agent.generate_str(
        "Try to fetch https://nonexistent-domain-12345.invalid and "
        "if that fails, fetch https://example.com instead"
    )

    # Should attempt multiple fetches
    session.add_deferred_evaluator(
        ToolWasCalled("fetch", min_times=1),  # At least one fetch attempt
        "fetch_attempts_made",
    )

    # Should demonstrate recovery
    recovery_judge = LLMJudge(
        rubric="Response should show attempt to fetch the invalid URL, recognize the error, and successfully fetch the fallback URL",
        min_score=0.8,
    )

    await session.evaluate_now_async(
        recovery_judge, response, "error_recovery_demonstration"
    )
