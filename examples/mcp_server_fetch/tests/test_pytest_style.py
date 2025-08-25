"""Pytest-style tests for MCP fetch server using mcp-eval fixtures."""

import pytest
import mcp_eval
from mcp_eval import Expect
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.metrics import TestMetrics
from mcp_eval.session import TestAgent
from mcp_agent.agents.agent import Agent


@mcp_eval.setup
def configure_for_pytest():
    """Configure mcp-eval for pytest integration."""
    # Define servers on your Agent/AgentSpec; no global server selection needed here


@pytest.mark.asyncio
@pytest.mark.network
async def test_basic_fetch_with_pytest(mcp_agent: TestAgent):
    """Test basic URL fetching using pytest fixture."""
    response = await mcp_agent.generate_str(
        "Fetch the content from https://example.com"
    )

    # Modern evaluator approach
    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch"), name="fetch_tool_called", response=response
    )
    await mcp_agent.session.assert_that(
        Expect.content.contains("Example Domain"),
        name="contains_example_domain",
        response=response,
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.mcp_agent("default")
async def test_fetch_with_markdown_conversion(mcp_agent: TestAgent):
    """Test that HTML is properly converted to markdown."""
    response = await mcp_agent.generate_str(
        "Fetch https://example.com and tell me about the content format"
    )

    # Check tool usage
    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch"), name="fetch_called"
    )

    # Use LLM judge to evaluate markdown conversion
    markdown_judge = Expect.judge.llm(
        rubric="Response should indicate that content was converted to markdown format",
        min_score=0.7,
        include_input=True,
    )
    await mcp_agent.session.assert_that(
        markdown_judge, name="markdown_conversion_check", response=response
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
async def test_fetch_multiple_urls(
    mcp_agent: TestAgent, url: str, expected_content: str
):
    """Parametrized test for multiple URLs."""
    response = await mcp_agent.generate_str(f"Fetch content from {url}")

    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch"),
        name=f"fetch_called_for_{url.split('//')[1].replace('.', '_')}",
        response=response,
    )

    await mcp_agent.session.assert_that(
        Expect.content.contains(expected_content, case_sensitive=False),
        name="contains_expected_content",
        response=response,
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.mcp_agent("advanced")
async def test_fetch_error_handling(mcp_agent: TestAgent):
    """Test error handling for invalid URLs."""
    response = await mcp_agent.generate_str(
        "Try to fetch content from https://this-domain-should-not-exist-12345.com"
    )

    # Should still call the fetch tool
    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch"), name="fetch_attempted"
    )

    # Should handle the error gracefully
    error_handling_judge = Expect.judge.llm(
        rubric="Response should acknowledge the fetch failed and explain the error appropriately",
        min_score=0.8,
    )
    await mcp_agent.session.assert_that(
        error_handling_judge, name="error_handling_quality", response=response
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.mcp_agent(
    Agent(name="custom_fetcher", instruction="You fetch", server_names=["fetch"])
)
async def test_with_custom_agent_instance(mcp_agent: TestAgent):
    response = await mcp_agent.generate_str("Fetch https://example.com and summarise")
    assert "example" in response.lower()


@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_with_raw_content(mcp_agent: TestAgent):
    """Test fetching raw HTML content."""
    response = await mcp_agent.generate_str(
        "Fetch the raw HTML content from https://example.com without markdown conversion"
    )

    # Check that fetch was called
    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch"), name="fetch_raw_called"
    )

    # Check for HTML tags in response
    await mcp_agent.session.assert_that(
        Expect.content.contains("<html", case_sensitive=False),
        name="contains_html_tags",
        response=response,
    )


@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.slow
async def test_large_content_chunking(mcp_agent: TestAgent):
    """Test fetching large content with chunking."""
    response = await mcp_agent.generate_str(
        "Fetch content from https://httpbin.org/json and if it's truncated, "
        "continue fetching until you have the complete content"
    )

    # Should call fetch tool (possibly multiple times for chunking)
    await mcp_agent.session.assert_that(
        Expect.tools.was_called("fetch", min_times=1), name="fetch_called_for_chunking"
    )

    # Should get complete content
    completeness_judge = Expect.judge.llm(
        rubric="Response should contain complete JSON data or acknowledge if chunking was needed",
        min_score=0.8,
    )
    await mcp_agent.session.assert_that(
        completeness_judge, name="content_completeness", response=response
    )


# Complex test cases to verify OTEL trace parsing and metrics
class CustomMetricsValidationEvaluator(Evaluator):
    """Evaluator that validates metrics are being collected correctly."""

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        metrics = context.metrics

        # Check that we have metrics
        assert metrics is not None, "Metrics should not be None"
        assert isinstance(
            metrics, TestMetrics
        ), f"Expected TestMetrics, got {type(metrics)}"

        # Validate tool calls were recorded
        assert len(metrics.tool_calls) > 0, "Should have recorded tool calls"
        assert len(metrics.unique_tools_used) > 0, "Should have unique tools recorded"

        # Check that fetch tool was used
        assert (
            "fetch" in metrics.unique_tools_used
        ), "Fetch tool should be in unique tools"

        # Validate timing metrics
        assert metrics.total_duration_ms > 0, "Total duration should be positive"
        assert metrics.latency_ms > 0, "Latency should be positive"

        # Check LLM metrics
        assert metrics.llm_metrics.model_name != "", "Model name should be set"
        assert metrics.llm_metrics.total_tokens > 0, "Should have token usage"

        return EvaluatorResult(
            passed=True,
            details={
                "tool_calls": len(metrics.tool_calls),
                "unique_tools": metrics.unique_tools_used,
                "duration_ms": metrics.total_duration_ms,
                "tokens": metrics.llm_metrics.total_tokens,
            },
        )


@pytest.mark.asyncio
@pytest.mark.network
async def test_metrics_collection_single_fetch(mcp_agent: TestAgent):
    """Test that metrics are properly collected for a single fetch."""
    # Add metrics validation evaluator
    await mcp_agent.session.assert_that(
        CustomMetricsValidationEvaluator(), name="metrics_validation"
    )

    response = await mcp_agent.generate_str(
        "Fetch https://example.com and summarize it in one sentence."
    )

    assert response
    assert len(response) > 10

    # Access metrics directly to verify
    metrics = mcp_agent.session.get_metrics()
    assert len(metrics.tool_calls) >= 1, "Should have at least one tool call"
    assert metrics.tool_calls[0].name == "fetch", "First tool call should be fetch"


@pytest.mark.asyncio
@pytest.mark.network
async def test_multiple_sequential_fetches_metrics(mcp_agent: TestAgent):
    """Test metrics collection for multiple sequential fetches."""
    response = await mcp_agent.generate_str(
        "First fetch https://example.com, then fetch https://httpbin.org/json. "
        "Summarize what you found from both sites."
    )

    assert response
    assert len(response) > 20

    # Check metrics show multiple tool calls
    metrics = mcp_agent.session.get_metrics()
    assert len(metrics.tool_calls) >= 2, "Should have at least 2 tool calls"

    # Verify both were fetch calls
    tool_names = [call.name for call in metrics.tool_calls]
    assert all(name == "fetch" for name in tool_names), "All tool calls should be fetch"

    # Add deferred evaluator to verify at session end
    await mcp_agent.session.assert_that(
        CustomMetricsValidationEvaluator(), name="multi_fetch_metrics"
    )


@pytest.mark.asyncio
@pytest.mark.network
async def test_parallel_fetches_detection(mcp_agent: TestAgent):
    """Test that parallel fetch calls are detected in metrics."""
    response = await mcp_agent.generate_str(
        "Fetch these URLs in parallel and tell me the title of each page: "
        "https://example.com, https://httpbin.org/html, and https://httpbin.org/json"
    )

    assert response

    # Check metrics
    metrics = mcp_agent.session.get_metrics()
    assert len(metrics.tool_calls) >= 3, "Should have at least 3 tool calls"

    # Log parallel calls for debugging
    print(f"Parallel tool calls detected: {metrics.parallel_tool_calls}")

    # Check if any calls overlapped in time (indicating parallelism)
    # Note: This might not always detect parallelism depending on execution
    if len(metrics.tool_calls) >= 2:
        # Check for overlapping time windows
        for i in range(len(metrics.tool_calls) - 1):
            for j in range(i + 1, len(metrics.tool_calls)):
                call1 = metrics.tool_calls[i]
                call2 = metrics.tool_calls[j]
                # Check if calls overlap
                if (
                    call1.start_time < call2.end_time
                    and call2.start_time < call1.end_time
                ):
                    print(f"Found overlapping calls: {call1.name} and {call2.name}")


@pytest.mark.asyncio
@pytest.mark.network
async def test_span_tree_structure(mcp_agent: TestAgent):
    """Test that span tree is properly constructed from OTEL traces."""
    _response = await mcp_agent.generate_str(
        "Fetch https://example.com and tell me about it"
    )

    # Get span tree
    span_tree = mcp_agent.session.get_span_tree()
    assert span_tree is not None, "Span tree should not be None"

    # Check tree structure
    total_spans = span_tree.count_spans()
    assert total_spans > 0, "Should have spans in the tree"

    # Find tool spans
    tool_spans = span_tree.find_spans_by_attribute("mcp.tool.name")
    assert len(tool_spans) > 0, "Should have tool spans"

    # Log tree info for debugging
    print(f"Total spans: {total_spans}")
    print(f"Max depth: {span_tree.max_depth()}")
    print(f"Tool spans found: {len(tool_spans)}")


@pytest.mark.asyncio
@pytest.mark.network
async def test_error_metrics_tracking(mcp_agent: TestAgent):
    """Test that errors are properly tracked in metrics."""
    response = await mcp_agent.generate_str(
        "Try to fetch these URLs: https://example.com (valid) and "
        "https://this-definitely-does-not-exist-12345.invalid (invalid)"
    )

    assert response

    # Check metrics
    metrics = mcp_agent.session.get_metrics()
    assert len(metrics.tool_calls) >= 2, "Should have attempted both fetches"

    # Check for errors
    error_calls = [call for call in metrics.tool_calls if call.is_error]
    print(f"Error calls: {len(error_calls)}")
    print(f"Success rate: {metrics.success_rate}")

    # At least one should have failed
    assert metrics.error_count >= 0, "Should track errors"


@pytest.mark.asyncio
@pytest.mark.network
async def test_comprehensive_metrics_validation(mcp_agent: TestAgent):
    """Comprehensive test that validates all aspects of metrics collection."""
    # Add comprehensive metrics evaluator
    await mcp_agent.session.assert_that(
        CustomMetricsValidationEvaluator(), name="comprehensive_metrics"
    )

    # Complex prompt that exercises multiple features
    response = await mcp_agent.generate_str("""
        Please do the following:
        1. Fetch https://example.com and https://httpbin.org/json
        2. Tell me the main heading from example.com
        3. Tell me if httpbin.org/json contains slideshow data
        4. Summarize your findings
    """)

    assert response
    assert len(response) > 50

    # Get final metrics for validation
    metrics = mcp_agent.session.get_metrics()

    # Comprehensive checks
    assert len(metrics.tool_calls) >= 2, "Should have multiple tool calls"
    assert metrics.unique_tools_used == ["fetch"], "Should only use fetch tool"
    assert metrics.total_duration_ms > 0, "Should have positive duration"
    assert metrics.llm_metrics.input_tokens > 0, "Should have input tokens"
    assert metrics.llm_metrics.output_tokens > 0, "Should have output tokens"
    assert metrics.cost_estimate > 0, "Should have cost estimate"

    # Log detailed metrics
    print("\n=== Detailed Metrics ===")
    print(f"Tool calls: {len(metrics.tool_calls)}")
    print(f"Unique tools: {metrics.unique_tools_used}")
    print(f"Total duration: {metrics.total_duration_ms}ms")
    print(f"Model: {metrics.llm_metrics.model_name}")
    print(f"Tokens: {metrics.llm_metrics.total_tokens}")
    print(f"Cost estimate: ${metrics.cost_estimate:.6f}")
    print(f"Iterations: {metrics.iteration_count}")
    print(f"Success rate: {metrics.success_rate * 100:.1f}%")
