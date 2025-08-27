import asyncio
from mcp_agent.agents.agent_spec import AgentSpec

import mcp_eval
from mcp_eval import task, setup, Case, Dataset, Expect
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_fetch_agent():
    """Configure agent with fetch server for tests."""
    spec = AgentSpec(
        name="FetchTester",
        instruction="You are a helpful assistant that can fetch and analyze web content.",
        server_names=["fetch"],
    )
    mcp_eval.use_agent(spec)


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
    response = await agent.generate_str(
        "Fetch multiple URLs: example.com and github.com"
    )

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
            response=response,  # Pass the response to the judge
        )

        # Use path efficiency evaluator
        await session.assert_that(
            Expect.path.efficiency(
                expected_tool_sequence=["fetch", "fetch"],
                allow_extra_steps=1,
                tool_usage_limits={"fetch": 2},  # Allow up to 2 fetch calls
                penalize_repeated_tools=False,  # Don't penalize using fetch twice
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
    agent_spec=AgentSpec(
        name="DatasetFetchTester",
        instruction="You are a helpful assistant that can fetch and analyze web content.",
        server_names=["fetch"],
    ),
    # LLM provider/model now configured globally in mcpeval.yaml
)


async def dataset_with_enhanced_features():
    """Dataset evaluation using enhanced features."""

    async def enhanced_fetch_task(
        inputs: str, agent: TestAgent, session: TestSession
    ) -> str:
        # The Dataset.evaluate method passes the agent and session
        # We should use them instead of creating our own
        print(f"DEBUG: Agent name: {agent.agent.name}")
        print(f"DEBUG: Agent has LLM: {agent._llm is not None}")
        print(f"DEBUG: Input: {inputs}")
        response = await agent.generate_str(inputs)
        print(f"DEBUG: Response: {response[:200] if len(response) > 200 else response}")
        return response

    # Run evaluation
    report = await dataset.evaluate(enhanced_fetch_task, max_concurrency=2)

    # Print results manually since there's no print method
    print("\n" + ("=" * 60))
    print(f"Evaluation Report: {report.dataset_name}")
    print(f"Task: {report.task_name}")
    print("=" * 60)

    for result in report.results:
        print(f"\nCase: {result.case_name}")
        print(f"  Input: {result.inputs}")
        print(
            f"  Output: {result.output[:100]}..."
            if result.output and len(str(result.output)) > 100
            else f"  Output: {result.output}"
        )
        print(f"  Passed: {result.passed}")
        for eval_result in result.evaluation_results:
            print(f"    - {eval_result.name}: {'✓' if eval_result.passed else '✗'}")
            if eval_result.error:
                print(f"      Error: {eval_result.error}")

    print("\n" + ("=" * 60))
    print(
        f"Results: {report.passed_cases}/{report.total_cases} cases passed ({report.success_rate:.1%})"
    )
    print(f"Average duration: {report.average_duration_ms:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(dataset_with_enhanced_features())
