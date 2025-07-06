"""Advanced feature tests demonstrating deep analysis capabilities."""

import mcp_eval
from mcp_eval import task, setup
from mcp_eval import ToolWasCalled, LLMJudge, ToolSequence


@setup
def configure_advanced_tests():
    """Configure for advanced feature testing."""
    mcp_eval.use_server("fetch")
    mcp_eval.use_agent(
        {
            "name": "advanced_fetch_agent",
            "instruction": "You are an expert web content agent with advanced capabilities.",
            "llm_factory": "AnthropicAugmentedLLM",
            "model": "claude-3-sonnet-20240229",
            "max_iterations": 10,
        }
    )


@task("Test span tree analysis for fetch operations")
async def test_span_tree_analysis(agent, session):
    """Test advanced span tree analysis capabilities."""
    await agent.generate_str(
        "Fetch content from https://example.com, then fetch https://httpbin.org/json, "
        "and compare what you found"
    )

    # Expected tool sequence
    session.add_deferred_evaluator(
        ToolSequence(["fetch", "fetch"], allow_other_calls=True),
        "correct_fetch_sequence",
    )

    # Wait for completion and analyze span tree
    span_tree = session.get_span_tree()
    if span_tree:
        # Check for performance issues
        rephrasing_loops = span_tree.get_llm_rephrasing_loops()
        if rephrasing_loops:
            session._record_evaluation_result(
                "no_rephrasing_loops",
                False,
                f"Found {len(rephrasing_loops)} LLM rephrasing loops",
            )
        else:
            session._record_evaluation_result("no_rephrasing_loops", True, None)

        # Analyze tool path efficiency
        golden_paths = {"multi_fetch": ["fetch", "fetch"]}
        path_analyses = span_tree.get_inefficient_tool_paths(golden_paths)
        for analysis in path_analyses:
            efficiency_passed = analysis.efficiency_score >= 0.8
            session._record_evaluation_result(
                "path_efficiency",
                efficiency_passed,
                f"Tool path efficiency: {analysis.efficiency_score:.2f}",
            )

        # Check error recovery
        recovery_sequences = span_tree.get_error_recovery_sequences()
        if recovery_sequences:
            successful_recoveries = sum(
                1 for seq in recovery_sequences if seq.recovery_successful
            )
            total_recoveries = len(recovery_sequences)
            session._record_evaluation_result(
                "error_recovery",
                successful_recoveries == total_recoveries,
                f"Error recovery: {successful_recoveries}/{total_recoveries} successful",
            )


@task("Test enhanced LLM judge with structured output")
async def test_enhanced_llm_judge(agent, session):
    """Test the enhanced LLM judge with structured JSON output."""
    response = await agent.generate_str(
        "Fetch https://httpbin.org/html and provide a detailed analysis of the content structure"
    )

    # Basic tool check
    session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_called_for_analysis")

    # Enhanced LLM judge with structured output
    enhanced_judge = LLMJudge(
        rubric="""
        Evaluate the response based on these criteria:
        1. Successfully fetched the HTML content
        2. Provided structural analysis of the content
        3. Demonstrated understanding of HTML elements
        4. Gave specific details about what was found
        """,
        min_score=0.85,
        include_input=True,
        require_reasoning=True,
    )

    await session.evaluate_now_async(
        enhanced_judge, response, "detailed_content_analysis"
    )


@task("Test fetch server capabilities under load")
async def test_fetch_performance_analysis(agent, session):
    """Test fetch server performance characteristics."""
    response = await agent.generate_str(
        "Fetch content from these URLs in sequence: "
        "https://example.com, https://httpbin.org/json, https://httpbin.org/html. "
        "Provide a summary of each."
    )

    # Should make multiple fetch calls
    session.add_deferred_evaluator(
        ToolWasCalled("fetch", min_times=3), "multiple_fetch_calls"
    )

    # Performance evaluation
    performance_judge = LLMJudge(
        rubric="Response should demonstrate efficient fetching of multiple URLs with appropriate summaries for each",
        min_score=0.8,
    )

    await session.evaluate_now_async(
        performance_judge, response, "multi_url_performance"
    )

    # Check final metrics
    metrics = session.get_metrics()

    # Custom performance checks
    total_duration = metrics.total_duration_ms
    if total_duration < 30000:  # Less than 30 seconds
        session._record_evaluation_result("reasonable_duration", True, None)
    else:
        session._record_evaluation_result(
            "reasonable_duration", False, f"Duration too long: {total_duration:.0f}ms"
        )

    # Check tool call efficiency
    tool_calls = len(metrics.tool_calls)
    if tool_calls >= 3:  # Expected number of fetch calls
        session._record_evaluation_result("sufficient_tool_calls", True, None)
    else:
        session._record_evaluation_result(
            "sufficient_tool_calls", False, f"Only {tool_calls} tool calls made"
        )


@task("Test fetch server error recovery patterns")
async def test_comprehensive_error_recovery(agent, session):
    """Test comprehensive error recovery patterns."""
    response = await agent.generate_str(
        "Try to fetch these URLs in order, and for any that fail, "
        "try an alternative approach: "
        "1. https://invalid-domain-12345.com "
        "2. https://example.com "
        "3. https://httpbin.org/status/404 "
        "4. https://httpbin.org/json"
    )

    # Should make multiple fetch attempts
    session.add_deferred_evaluator(
        ToolWasCalled("fetch", min_times=3), "multiple_fetch_attempts"
    )

    # Comprehensive error handling evaluation
    error_recovery_judge = LLMJudge(
        rubric="""
        Evaluate the agent's error recovery capabilities:
        1. Attempts to fetch invalid URLs and recognizes failures
        2. Successfully fetches valid URLs
        3. Handles HTTP errors appropriately
        4. Provides clear status for each URL attempt
        5. Demonstrates resilience and appropriate fallback behavior
        """,
        min_score=0.8,
        include_input=True,
    )

    await session.evaluate_now_async(
        error_recovery_judge, response, "comprehensive_error_recovery"
    )
