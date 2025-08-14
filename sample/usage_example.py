import asyncio
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


import mcp_eval

from mcp_eval import task, setup, ToolWasCalled, LLMJudge, Case, Dataset
from mcp_eval.core import with_agent
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_tests():
    # Prefer consolidated mcp-agent config; optional lightweight overrides
    mcp_eval.use_server("fetch")


@task("Test with enhanced LLM judge")
async def test_enhanced_judge(agent: TestAgent, session: TestSession):
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

    await session.assert_that(
        enhanced_judge,
        name="enhanced_judge_test",
        response=response,
    )

    await session.assert_that(
        ToolWasCalled("fetch"),
        name="fetch_called",
    )


@task("Test with span tree analysis")
async def test_span_analysis(agent: TestAgent, session: TestSession):
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
                EvaluatorResult(passed=False),
                f"Found {len(rephrasing_loops)} rephrasing loops",
            )
        else:
            session._record_evaluation_result(
                "no_rephrasing_loops", EvaluatorResult(passed=True), None
            )

        # Analyze tool path efficiency
        golden_paths = {
            "fetch_multiple": ["fetch", "fetch"]  # Expected: two fetch calls
        }
        path_analyses = span_tree.get_inefficient_tool_paths(golden_paths)
        for analysis in path_analyses:
            session._record_evaluation_result(
                "path_efficiency",
                EvaluatorResult(passed=analysis.efficiency_score > 0.8),
                f"Efficiency: {analysis.efficiency_score:.2f}",
            )


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
        "llm_factory": "AnthropicAugmentedLLM",
        "max_iterations": 10,
    },
)


async def dataset_with_enhanced_features():
    """Dataset evaluation using enhanced features."""

    async def enhanced_fetch_task(inputs: str) -> str:
        # Example A: Programmatic Agent + LLM via with_agent per-test override
        prog_agent = Agent(
            name="prog",
            instruction="You can fetch and summarize.",
            server_names=["fetch"],
        )
        prog_llm = OpenAIAugmentedLLM(agent=prog_agent)

        @with_agent(prog_llm)
        @task("enhanced_task_prog")
        async def run(agent: TestAgent, _session: TestSession):
            return await agent.generate_str(inputs)

        result = await run()
        return result.output if hasattr(result, "output") else ""

    # Run evaluation
    report = await dataset.evaluate(enhanced_fetch_task, max_concurrency=2)
    report.print(include_input=True, include_output=True, include_scores=True)

    print(
        f"Results: {report.passed_cases}/{report.total_cases} cases passed ({report.success_rate:.1%})"
    )
    print(f"Average duration: {report.average_duration_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(dataset_with_enhanced_features())
