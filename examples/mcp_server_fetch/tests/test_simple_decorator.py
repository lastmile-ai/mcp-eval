from mcp_eval import task, setup, Expect
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_decorator_tests():
    pass


@task("basic_website_fetch")
async def basic_website_fetch(agent: TestAgent, session: TestSession):
    await agent.generate_str(
        "Please fetch the content from https://example.com and tell me what you find"
    )
    await session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await session.assert_that(
        Expect.tools.called_with("fetch", {"url": "https://example.com"})
    )
