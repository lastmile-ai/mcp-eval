from mcp_eval import task, setup, Expect
from mcp_eval.session import TestAgent, TestSession
from mcp_agent.agents.agent_spec import AgentSpec
import mcp_eval


@setup
def configure_sample_server():
    """Configure agent for sample server tests."""
    # Define an agent that can use our sample server tools
    spec = AgentSpec(
        name="SampleServerTester",
        instruction="You help users by using the available time and text tools.",
        server_names=[
            "sample_server"
        ],  # Assumes sample_server is configured in mcp-agent.config.yaml
    )
    mcp_eval.use_agent(spec)


@task(description="A simple success case for getting the time in a major city.")
async def test_get_time_in_london(agent: TestAgent, session: TestSession):
    """
    Tests the basic functionality of the get_current_time tool and objective success.
    """
    objective = "Can you tell me the current time in London, UK?"
    response = await agent.generate_str(objective)

    # Content checks and tool usage
    await session.assert_that(
        Expect.content.contains("london", case_sensitive=False), response=response
    )
    await session.assert_that(
        Expect.content.contains("current time"), response=response
    )
    await session.assert_that(Expect.tools.was_called("get_current_time"))
    await session.assert_that(
        Expect.tools.called_with("get_current_time", {"timezone": "Europe/London"})
    )

    # Confirm the overall goal was met
    await session.assert_that(
        Expect.judge.llm(
            rubric="The response should provide the current time in London with appropriate timezone information",
            min_score=0.8,
        ),
        name="evaluate_objective",
        response=response,
        inputs=objective,
    )


@task(description="A test designed to FAIL by checking summarization quality.")
async def test_summarization_quality_fails(agent: TestAgent, session: TestSession):
    """
    This test exposes the weakness of the naive summarization tool.
    The LLM Judge will score the truncated, incoherent summary poorly, causing the test to fail.
    """
    objective = "Please summarize this text for me, make it about 15 words: 'Artificial intelligence is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.'"
    response = await agent.generate_str(objective)

    # This assertion will fail because the summary is just a blunt truncation.
    await session.assert_that(
        Expect.judge.llm(
            rubric="The summary must be coherent, grammatically correct, and capture the main idea of the original text. It should not be abruptly cut off.",
            min_score=0.8,
        ),
        name="evaluate_coherent_summary",
        response=response,
        inputs=objective,
    )


@task(
    description="Tests if the agent can chain tools to achieve a multi-step objective."
)
async def test_chained_tool_use(agent: TestAgent, session: TestSession):
    """
    This test requires the agent to first get the time and then summarize the result,
    testing its ability to perform multi-step reasoning.
    """
    objective = "First, find out the current time in Tokyo, then write a short, one-sentence summary of that information."
    response = await agent.generate_str(objective)

    # Content and path checks
    await session.assert_that(
        Expect.content.contains("tokyo", case_sensitive=False), response=response
    )
    await session.assert_that(Expect.content.contains("time"), response=response)
    await session.assert_that(Expect.tools.was_called("get_current_time"))
    await session.assert_that(Expect.tools.was_called("summarize_text"))
    await session.assert_that(
        Expect.path.efficiency(
            expected_tool_sequence=["get_current_time", "summarize_text"],
            allow_extra_steps=1,
        ),
        name="efficient_path",
    )


@task(description="Tests how the agent handles a known error from a tool.")
async def test_invalid_timezone_error_handling(agent: TestAgent, session: TestSession):
    """
    This test checks if the agent correctly handles an error from the get_current_time tool
    and clearly communicates the failure to the user.
    """
    objective = "What time is it in the made-up city of Atlantis?"
    response = await agent.generate_str(objective)

    # The agent should respond that it can't find the timezone.
    await session.assert_that(
        Expect.content.contains("Unknown timezone"), response=response
    )
    # Check error handling quality
    await session.assert_that(
        Expect.judge.llm(
            rubric="The response should clearly explain that the timezone/location cannot be found and handle the error gracefully",
            min_score=0.7,
        ),
        name="error_handling_quality",
        response=response,
        inputs=objective,
    )
