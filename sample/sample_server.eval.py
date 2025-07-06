from mcp_eval import task
from mcp_eval.assertions import (
    contains,
    tool_was_called,
    objective_succeeded,
    plan_is_efficient,
    llm_judge,
    tool_arguments_match,
)


@task(
    description="A simple success case for getting the time in a major city.",
    server="sample_server",
)
async def test_get_time_in_london(agent, session):
    """
    Tests the basic functionality of the get_current_time tool and objective success.
    """
    objective = "Can you tell me the current time in London, UK?"
    response = await agent.llm.generate_str(objective)

    # Check for keywords in the response
    contains("london", response.lower(), session=session)
    contains("current time", response.lower(), session=session)

    # Verify the correct tool was used with the correct arguments
    tool_was_called("get_current_time", session=session)
    tool_arguments_match(
        "get_current_time", {"timezone": "Europe/London"}, session=session
    )

    # Confirm the overall goal was met
    objective_succeeded(objective, response, session=session)


@task(
    description="A test designed to FAIL by checking summarization quality.",
    server="sample_server",
)
async def test_summarization_quality_fails(agent, session):
    """
    This test exposes the weakness of the naive summarization tool.
    The LLM Judge will score the truncated, incoherent summary poorly, causing the test to fail.
    """
    objective = "Please summarize this text for me, make it about 15 words: 'Artificial intelligence is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.'"
    response = await agent.llm.generate_str(objective)

    # This assertion will fail because the summary is just a blunt truncation.
    llm_judge(
        response_to_judge=response,
        rubric="The summary must be coherent, grammatically correct, and capture the main idea of the original text. It should not be abruptly cut off.",
        min_score=0.8,
        session=session,
    )


@task(
    description="Tests if the agent can chain tools to achieve a multi-step objective.",
    server="sample_server",
)
async def test_chained_tool_use(agent, session):
    """
    This test requires the agent to first get the time and then summarize the result,
    testing its ability to perform multi-step reasoning.
    """
    objective = "First, find out the current time in Tokyo, then write a short, one-sentence summary of that information."
    response = await agent.llm.generate_str(objective)

    # Check that the final response contains the key information
    contains("tokyo", response.lower(), session=session)
    contains("time", response.lower(), session=session)

    # Verify that both tools were called in the process
    tool_was_called("get_current_time", session=session)
    tool_was_called("summarize_text", session=session)

    # Check that the agent's plan was efficient
    plan_is_efficient(objective, session=session)


@task(
    description="Tests how the agent handles a known error from a tool.",
    server="sample_server",
)
async def test_invalid_timezone_error_handling(agent, session):
    """
    This test checks if the agent correctly handles an error from the get_current_time tool
    and clearly communicates the failure to the user.
    """
    objective = "What time is it in the made-up city of Atlantis?"
    response = await agent.llm.generate_str(objective)

    # The agent should respond that it can't find the timezone.
    contains("Unknown timezone", response, session=session)

    # The overall objective should fail, as the agent couldn't fulfill the core request.
    # We expect this assertion to fail, which means the test case correctly identifies the objective failure.
    objective_succeeded(objective, response, session=session)
