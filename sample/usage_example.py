import asyncio
import mcp_eval
from mcp_eval import task, setup, test_session, ToolWasCalled, LLMJudge, Case, Dataset
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM


@setup
def configure_tests():
    mcp_eval.use_server("fetch")
    mcp_eval.use_agent(
        {
            "name": "enhanced_tester",
            "instruction": "You can fetch web content and analyze it thoroughly.",
            "llm_factory": AnthropicAugmentedLLM,
        }
    )


@task("Test with enhanced LLM judge")
async def test_enhanced_judge(agent, session):
    """Test using the enhanced LLM judge with structured output."""
    response = await agent.generate_str(
        "Fetch https://example.com and explain what it is"
    )

    # Enhanced LLM judge with structured output
    enhanced_judge = LLMJudge(
        rubric="Response should fetch the website and provide a clear explanation of what example.com is",
        min_score=0.8,
        include_input=True,
        require_reasoning=True,
    )

    await session.evaluate_now_async(enhanced_judge, response, "enhanced_judge_test")
    session.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_called")


@task("Test with span tree analysis")
async def test_span_analysis(agent, session):
    """Test that demonstrates span tree analysis capabilities."""
    await agent.generate_str("Fetch multiple URLs: example.com and github.com")

    # Wait for execution to complete, then analyze span tree
    span_tree = session.get_span_tree()
    if span_tree:
        # Check for potential issues
        rephrasing_loops = span_tree.get_llm_rephrasing_loops()
        if rephrasing_loops:
            session._record_evaluation_result(
                "no_rephrasing_loops",
                False,
                f"Found {len(rephrasing_loops)} rephrasing loops",
            )
        else:
            session._record_evaluation_result("no_rephrasing_loops", True, None)

        # Analyze tool path efficiency
        golden_paths = {
            "fetch_multiple": ["fetch", "fetch"]  # Expected: two fetch calls
        }
        path_analyses = span_tree.get_inefficient_tool_paths(golden_paths)
        for analysis in path_analyses:
            session._record_evaluation_result(
                "path_efficiency",
                analysis.efficiency_score > 0.8,
                f"Efficiency: {analysis.efficiency_score:.2f}",
            )


async def dataset_with_enhanced_features():
    """Dataset evaluation using enhanced features."""

    # Enhanced test cases
    cases = [
        Case(
            name="fetch_with_structured_judge",
            inputs="Fetch https://example.com and summarize its purpose",
            evaluators=[
                ToolWasCalled("fetch"),
                LLMJudge(
                    rubric="Response should include both website content and a clear summary",
                    min_score=0.85,
                    include_input=True,
                    require_reasoning=True,
                ),
            ],
        ),
        Case(
            name="multi_step_task",
            inputs="Fetch both example.com and github.com, then compare them",
            evaluators=[
                ToolWasCalled("fetch", min_times=2),
                LLMJudge(
                    rubric="Response should demonstrate comparison between the two websites",
                    min_score=0.8,
                ),
            ],
        ),
    ]

    dataset = Dataset(
        name="Enhanced Fetch Tests",
        cases=cases,
        server_name="fetch",
        agent_config={
            "llm_factory": AnthropicAugmentedLLM,
            "max_iterations": 10,
        },
    )

    async def enhanced_fetch_task(inputs: str) -> str:
        async with test_session("fetch", "enhanced_task") as agent:
            return await agent.generate_str(inputs)

    # Run evaluation
    report = await dataset.evaluate(enhanced_fetch_task, max_concurrency=2)
    report.print(include_input=True, include_output=True, include_scores=True)

    # Analyze results
    for result in report.results:
        if result.span_tree:
            # Check for performance issues
            loops = result.span_tree.get_llm_rephrasing_loops()
            if loops:
                print(
                    f"Warning: Found {len(loops)} rephrasing loops in {result.case_name}"
                )

            recovery_sequences = result.span_tree.get_error_recovery_sequences()
            if recovery_sequences:
                successful_recoveries = sum(
                    1 for seq in recovery_sequences if seq.recovery_successful
                )
                print(
                    f"Error recovery: {successful_recoveries}/{len(recovery_sequences)} successful"
                )


if __name__ == "__main__":
    asyncio.run(dataset_with_enhanced_features())
