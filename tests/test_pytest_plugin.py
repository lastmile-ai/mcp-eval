import asyncio
import pytest

from mcp_eval.evaluators.response_contains import ResponseContains
from mcp_eval.pytest_plugin import MCPEvalPytestSession


@pytest.mark.asyncio
async def test_mcp_session_fixture_basic(mcp_session):
    # Fixture should provide agent and session, and allow immediate assertions
    assert mcp_session.agent is not None
    await mcp_session.session.assert_that(ResponseContains("ok"), response="ok")


@pytest.mark.asyncio
async def test_agent_alias_fixture(agent):
    # The alias fixture should also provide the agent
    assert agent is not None


@pytest.mark.asyncio
async def test_mcpevalpytestsess_appends_results(monkeypatch):
    # Import plugin module-level results list to check append
    import mcp_eval.pytest_plugin as plugin

    initial_len = len(plugin._pytest_results)
    async with MCPEvalPytestSession(test_name="plugin_unit", verbose=False) as sess:
        await sess.session.assert_that(ResponseContains("ok"), response="ok")

    assert len(plugin._pytest_results) == initial_len + 1
    last = plugin._pytest_results[-1]
    assert last.test_name == "plugin_unit"
    assert last.passed is True


def test_terminal_summary_renders(monkeypatch):
    # Prepare at least one result to render
    import mcp_eval.pytest_plugin as plugin
    from mcp_eval.core import TestResult

    plugin._pytest_results.clear()
    plugin._pytest_results.append(
        TestResult(
            id="id",
            test_name="t",
            description="d",
            server_name="",
            servers=[],
            agent_name="",
            parameters={},
            passed=True,
            evaluation_results=[],
            metrics=None,
            duration_ms=0.0,
            file="f",
        )
    )

    class _Cfg:
        def getoption(self, name):
            return 0

    class _TR:
        config = _Cfg()

    # Should not raise
    plugin.pytest_terminal_summary(_TR())


def test_event_loop_fixture(event_loop):
    # The plugin's session-scoped event loop should be usable
    assert hasattr(event_loop, "call_soon")
    called = {"v": False}
    def _set():
        called["v"] = True
    event_loop.call_soon(_set)
    event_loop.run_until_complete(asyncio.sleep(0))
    assert called["v"] is True


