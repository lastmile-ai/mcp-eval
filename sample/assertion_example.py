import mcp_eval
from mcp_eval.assertions import contains, tool_was_called, judge
from mcp_eval import task, setup


@setup
def configure_tests():
    mcp_eval.use_server("fetch")


@task("Legacy assertion style")
async def test_legacy_style(agent):
    """Using legacy assertions for backwards compatibility."""
    response = await agent.generate_str("Fetch https://example.com")

    # Legacy assertions still work
    contains(response, "Example Domain")
    tool_was_called("fetch")
    await judge(response, "Response contains website content")
