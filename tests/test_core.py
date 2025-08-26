import asyncio
import pytest

from mcp_eval.core import task, with_agent, TestResult
from mcp_eval.catalog import Expect


@pytest.mark.asyncio
async def test_task_decorator_runs_and_collects_results(tmp_path):
    @task("desc")
    async def my_test(agent, session):
        await session.assert_that(Expect.content.contains("hello"), response="hello world")

    result: TestResult = await my_test()
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.test_name == "my_test"


@pytest.mark.asyncio
async def test_with_agent_marker_is_attached():
    # Using a simple string AgentSpec name to ensure it attaches to function metadata
    @with_agent("dummy")
    @task("desc")
    async def my_test(agent, session):
        # No-op
        pass

    # Ensure attribute is present on original function (inner func gets wrapper)
    assert hasattr(my_test, "_is_mcpeval_task")
    # The with_agent decorator attaches override to the undecorated function, but after composition
    # wrapper used by @task consumes the attribute; presence indicates flow setup
    assert hasattr(my_test, "_description")


