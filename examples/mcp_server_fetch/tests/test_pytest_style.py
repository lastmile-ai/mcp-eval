"""Pytest-style tests for MCP fetch server using mcp-eval fixtures."""

import pytest
import mcp_eval
from mcp_eval import ToolWasCalled, ResponseContains, LLMJudge


@mcp_eval.setup
def configure_for_pytest():
    """Configure mcp-eval for pytest integration."""
    mcp_eval.use_server("fetch")


@pytest.mark.asyncio
@pytest.mark.network
async def test_basic_fetch_with_pytest(mcp_agent):
    """Test basic URL fetching using pytest fixture."""
    response = await mcp_agent.generate_str(
        "Fetch the content from https://example.com"
    )

    # Modern evaluator approach
    mcp_agent.session.evaluate_now(
        ToolWasCalled("fetch"), response, "fetch_tool_called"
    )
    mcp_agent.session.evaluate_now(
        ResponseContains("Example Domain"), response, "contains_example_domain"
    )


@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_with_markdown_conversion(mcp_agent):
    """Test that HTML is properly converted to markdown."""
    response = await mcp_agent.generate_str(
        "Fetch https://example.com and tell me about the content format"
    )

    # Check tool usage
    mcp_agent.session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_called")

    # Use LLM judge to evaluate markdown conversion
    markdown_judge = LLMJudge(
        rubric="Response should indicate that content was converted to markdown format",
        min_score=0.7,
        include_input=True,
    )
    await mcp_agent.session.evaluate_now_async(
        markdown_judge, response, "markdown_conversion_check"
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    "url,expected_content",
    [
        ("https://example.com", "Example Domain"),
        ("https://httpbin.org/html", "Herman Melville"),
        ("https://httpbin.org/json", "slideshow"),
    ],
)
async def test_fetch_multiple_urls(mcp_agent, url, expected_content):
    """Parametrized test for multiple URLs."""
    response = await mcp_agent.generate_str(f"Fetch content from {url}")

    mcp_agent.session.evaluate_now(
        ToolWasCalled("fetch"),
        response,
        f"fetch_called_for_{url.split('//')[1].replace('.', '_')}",
    )

    mcp_agent.session.evaluate_now(
        ResponseContains(expected_content), response, "contains_expected_content"
    )


@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_error_handling(mcp_agent):
    """Test error handling for invalid URLs."""
    response = await mcp_agent.generate_str(
        "Try to fetch content from https://this-domain-should-not-exist-12345.com"
    )

    # Should still call the fetch tool
    mcp_agent.session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_attempted")

    # Should handle the error gracefully
    error_handling_judge = LLMJudge(
        rubric="Response should acknowledge the fetch failed and explain the error appropriately",
        min_score=0.8,
    )
    await mcp_agent.session.evaluate_now_async(
        error_handling_judge, response, "error_handling_quality"
    )


@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_with_raw_content(mcp_agent):
    """Test fetching raw HTML content."""
    response = await mcp_agent.generate_str(
        "Fetch the raw HTML content from https://example.com without markdown conversion"
    )

    # Check that fetch was called
    mcp_agent.session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_raw_called")

    # Check for HTML tags in response
    mcp_agent.session.evaluate_now(
        ResponseContains("<html", case_sensitive=False), response, "contains_html_tags"
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.slow
async def test_large_content_chunking(mcp_agent):
    """Test fetching large content with chunking."""
    response = await mcp_agent.generate_str(
        "Fetch content from https://httpbin.org/json and if it's truncated, "
        "continue fetching until you have the complete content"
    )

    # Should call fetch tool (possibly multiple times for chunking)
    mcp_agent.session.add_deferred_evaluator(
        ToolWasCalled("fetch", min_times=1), "fetch_called_for_chunking"
    )

    # Should get complete content
    completeness_judge = LLMJudge(
        rubric="Response should contain complete JSON data or acknowledge if chunking was needed",
        min_score=0.8,
    )
    await mcp_agent.session.evaluate_now_async(
        completeness_judge, response, "content_completeness"
    )
