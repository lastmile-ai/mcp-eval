import pytest
import mcp_eval
from mcp_eval import ToolWasCalled, ResponseContains


# Configure mcp-eval for pytest
@mcp_eval.setup
def configure_for_pytest():
    mcp_eval.use_server("fetch")


# Standard pytest test with mcp-eval fixtures
@pytest.mark.asyncio
async def test_fetch_with_pytest_agent(mcp_agent):
    """Test using the mcp_agent fixture."""
    response = await mcp_agent.generate_str("Fetch https://example.com")

    # Use session through agent
    mcp_agent.session.evaluate_now(
        ResponseContains("Example Domain"), response, "contains_domain"
    )


@pytest.mark.asyncio
async def test_fetch_with_pytest_session(mcp_session):
    """Test using the full mcp_session fixture."""
    response = await mcp_session.agent.generate_str("Fetch https://github.com")

    # Direct session access
    mcp_session.session.add_deferred_evaluator(
        ToolWasCalled("fetch"), "fetch_was_called"
    )
    mcp_session.session.evaluate_now(
        ResponseContains("GitHub"), response, "contains_github"
    )


# Parametrized pytest test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com", "Example Domain"),
        ("https://github.com", "GitHub"),
    ],
)
async def test_fetch_parametrized(mcp_agent, url, expected):
    """Parametrized test combining pytest and mcp-eval."""
    response = await mcp_agent.generate_str(f"Fetch {url}")

    mcp_agent.session.evaluate_now(
        ResponseContains(expected),
        response,
        f"contains_{expected.lower().replace(' ', '_')}",
    )
