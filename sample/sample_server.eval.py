from mcp_eval import assertions, task
from mcp_eval.evaluators import LLMJudge
from mcp_eval.session import TestAgent, TestSession


@task(
    description="A simple success case for getting the time in a major city.",
    server="sample_server",
)
async def test_get_time_in_london(agent: TestAgent, session: TestSession):
    """
    Tests the basic functionality of the get_current_time tool and objective success.
    """
    objective = "Can you tell me the current time in London, UK?"
    response = await agent.generate_str(objective)

    # # Check for keywords in the response
    assertions.contains(session, response.lower(), "london")
    assertions.contains(session, response, "current time")

    # Verify the correct tool was used with the correct arguments
    assertions.tool_was_called(session, "get_current_time")

    assertions.tool_was_called_with(
        session, "get_current_time", {"timezone": "Europe/London"}
    )

    # # Confirm the overall goal was met
    judge = LLMJudge(rubric="Check if objective was fulfilled")
    await session.evaluate_now_async(judge, response, "evaluate_objective", objective)


@task(
    description="A test designed to FAIL by checking summarization quality.",
    server="sample_server",
)
async def test_summarization_quality_fails(agent: TestAgent, session: TestSession):
    """
    This test exposes the weakness of the naive summarization tool.
    The LLM Judge will score the truncated, incoherent summary poorly, causing the test to fail.
    """
    objective = "Please summarize this text for me, make it about 15 words: 'Artificial intelligence is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.'"
    response = await agent.generate_str(objective)

    # This assertion will fail because the summary is just a blunt truncation.
    judge = LLMJudge(
        rubric="The summary must be coherent, grammatically correct, and capture the main idea of the original text. It should not be abruptly cut off."
    )

    await session.evaluate_now_async(
        judge, response, "evaluate_coherent_summary", objective
    )


@task(
    description="Tests if the agent can chain tools to achieve a multi-step objective.",
    server="sample_server",
)
async def test_chained_tool_use(agent: TestAgent, session: TestSession):
    """
    This test requires the agent to first get the time and then summarize the result,
    testing its ability to perform multi-step reasoning.
    """
    objective = "First, find out the current time in Tokyo, then write a short, one-sentence summary of that information."
    response = await agent.generate_str(objective)

    # Check that the final response contains the key information
    assertions.contains(session, response, "tokyo")
    assertions.contains(session, response, "time")

    # Verify that both tools were called in the process
    assertions.tool_was_called(session, "get_current_time")
    assertions.tool_was_called(session, "summarize_text")

    # Check that the agent's plan was efficient
    assertions.path_efficiency(session)


@task(
    description="Tests how the agent handles a known error from a tool.",
    server="sample_server",
)
async def test_invalid_timezone_error_handling(agent: TestAgent, session: TestSession):
    """
    This test checks if the agent correctly handles an error from the get_current_time tool
    and clearly communicates the failure to the user.
    """
    objective = "What time is it in the made-up city of Atlantis?"
    response = await agent.generate_str(objective)

    # The agent should respond that it can't find the timezone.
    assertions.contains(session, response, "Unknown timezone")

    # The overall objective should fail, as the agent couldn't fulfill the core request.
    judge = LLMJudge(rubric="Check if objective was fulfilled")
    await session.evaluate_now_async(judge, response, "evaluate_objective", objective)
