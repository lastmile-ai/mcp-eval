import asyncio
from mcp_agent.agents.agent import Agent

import mcp_eval
from mcp_eval import task, Case, Dataset, Expect
from mcp_eval.session import TestAgent, TestSession


@task("Test with enhanced LLM judge")
async def test_enhanced_judge(agent: TestAgent, session: TestSession):
    """Test using the enhanced LLM judge with structured output."""
    response = await agent.generate_str(
        "Fetch https://example.com and explain what it is"
    )

    # Use the new Expect catalog API for cleaner assertions
    await session.assert_that(
        Expect.judge.llm(
            rubric="Response should fetch the website and provide a clear explanation of what example.com is",
            min_score=0.8,
            include_input=True,
            require_reasoning=True,
        ),
        name="enhanced_judge_test",
        response=response,
    )

    await session.assert_that(
        Expect.tools.was_called("fetch"),
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
        _rephrasing_loops = span_tree.get_llm_rephrasing_loops()
        # Use modern Expect API for better assertions
        await session.assert_that(
            Expect.judge.llm(
                "Agent should avoid unnecessary rephrasing loops", min_score=0.5
            ),
            name="avoid_rephrasing",
        )

        # Use path efficiency evaluator
        await session.assert_that(
            Expect.path.efficiency(
                expected_tool_sequence=["fetch", "fetch"],
                allow_extra_steps=1,
            ),
            name="path_efficiency",
        )


# Enhanced test cases using catalog-based evaluators
cases = [
    Case(
        name="fetch_with_structured_judge",
        inputs="Fetch https://example.com and summarize its purpose",
        evaluators=[
            Expect.tools.was_called("fetch"),
            Expect.judge.llm(
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
            Expect.tools.was_called("fetch", min_times=2),
            Expect.judge.llm(
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
    # LLM provider/model now configured globally in mcpeval.yaml
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

        async with mcp_eval.test_session(
            "enhanced_task_prog", agent=prog_agent
        ) as test_agent:
            return await test_agent.generate_str(inputs)

    # Run evaluation
    report = await dataset.evaluate(enhanced_fetch_task, max_concurrency=2)
    report.print(include_input=True, include_output=True, include_scores=True)

    print(
        f"Results: {report.passed_cases}/{report.total_cases} cases passed ({report.success_rate:.1%})"
    )
    print(f"Average duration: {report.average_duration_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(dataset_with_enhanced_features())
