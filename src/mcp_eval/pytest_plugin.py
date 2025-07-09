"""Pytest plugin for mcp-eval framework.

This plugin enables seamless integration between mcp-eval and pytest,
allowing users to write mcp-eval tests that run natively in pytest.
"""

import asyncio
import inspect
from typing import AsyncGenerator
import pytest

from mcp_eval import TestSession, TestAgent
from mcp_eval.config import get_current_config


class MCPEvalPytestSession:
    """Pytest-compatible wrapper around TestSession."""

    def __init__(
        self,
        server_name: str,
        test_name: str,
        agent_config: dict | None = None,
        verbose: bool = False,
    ):
        self._session = TestSession(server_name, test_name, agent_config, verbose)
        self._agent: TestAgent | None = None

    async def __aenter__(self):
        self._agent = await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.__aexit__(exc_type, exc_val, exc_tb)
        # Note: cleanup is already handled by the test_session context manager

    @property
    def agent(self) -> TestAgent | None:
        return self._agent

    @property
    def session(self) -> TestSession:
        return self._session


@pytest.fixture
async def mcp_session(request) -> AsyncGenerator[MCPEvalPytestSession, None]:
    """Pytest fixture that provides an MCP test session.

    Usage:
        async def test_my_mcp_function(mcp_session):
            response = await mcp_session.agent.generate_str("Hello")
            mcp_session.session.evaluate_now(ResponseContains("hello"), response, "greeting")
    """
    # Get test configuration
    config = get_current_config()
    server_name = config.get("default_server", "default")
    agent_config = config.get("agent_config", {})

    test_name = request.node.name

    # Check if pytest is running in verbose mode
    verbose = request.config.getoption("verbose") > 0

    # Create and yield session
    pytest_session_wrapper = MCPEvalPytestSession(
        server_name, test_name, agent_config, verbose
    )
    async with pytest_session_wrapper:
        yield pytest_session_wrapper
    # Cleanup happens after the context manager exits
    pytest_session_wrapper.session.cleanup()


@pytest.fixture
async def mcp_agent(mcp_session: MCPEvalPytestSession) -> TestAgent | None:
    """Convenience fixture that provides just the agent.

    Usage:
        async def test_my_function(mcp_agent):
            response = await mcp_agent.generate_str("Hello")
            mcp_agent.session.evaluate_now(ResponseContains("hello"), response, "greeting")
    """
    return mcp_session.agent


def pytest_configure(config):
    """Configure pytest to work with mcp-eval."""
    config.addinivalue_line("markers", "mcp-eval: mark test as an mcp-eval test")

    # Suppress Pydantic serialization warnings from MCP library
    # These warnings are due to MCP's internal union type handling and are not user-actionable
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Pydantic serializer warnings.*",
        category=UserWarning,
        module="pydantic.main",
    )

    # Track if we need to cleanup OTEL at the end
    config._mcp_eval_needs_otel_cleanup = False


def pytest_collection_modifyitems(config, items):
    """Automatically mark async tests that use mcp fixtures as mcp-eval tests."""
    for item in items:
        if hasattr(item, "function"):
            # Check if test function uses mcp fixtures
            sig = inspect.signature(item.function)
            if any(param in sig.parameters for param in ["mcp_session", "mcp_agent"]):
                item.add_marker(pytest.mark.mcpeval)


def pytest_runtest_setup(item):
    """Setup for mcp-eval tests."""
    if (
        "mcpeval" in item.keywords
        or "mcp-eval" in item.keywords
        or "mcp_eval" in item.keywords
    ):
        # Mark that we're using mcp-eval and will need cleanup
        item.config._mcp_eval_needs_otel_cleanup = True

        # Run any mcp-eval setup functions
        from mcp_eval.core import _setup_functions

        for setup_func in _setup_functions:
            if not asyncio.iscoroutinefunction(setup_func):
                setup_func()


def pytest_runtest_teardown(item):
    """Teardown for mcp-eval tests."""
    if (
        "mcpeval" in item.keywords
        or "mcp-eval" in item.keywords
        or "mcp_eval" in item.keywords
    ):
        # Run any mcp-eval teardown functions
        from mcp_eval.core import _teardown_functions

        for teardown_func in _teardown_functions:
            if not asyncio.iscoroutinefunction(teardown_func):
                teardown_func()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


