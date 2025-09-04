"""Pytest plugin for mcp-eval framework.

This plugin enables seamless integration between mcp-eval and pytest,
allowing users to write mcp-eval tests that run natively in pytest.
"""

import asyncio
import gc
import inspect
from typing import AsyncGenerator
from pathlib import Path
import pytest
import pytest_asyncio
from mcp_eval import TestSession, TestAgent
from mcp_eval.config import get_current_config
from mcp_eval.core import (
    TestResult,
    generate_test_id,
    _setup_functions,
    _teardown_functions,
)
from mcp_eval.core import _metrics_to_dict  # reuse consistent metrics shaping
from mcp_eval.report_generation.console import generate_failure_message
from mcp_eval.report_generation.summary import generate_combined_summary
from rich.console import Console


class MCPEvalPytestSession:
    """Pytest-compatible wrapper around TestSession."""

    def __init__(
        self,
        test_name: str,
        verbose: bool = False,
        *,
        agent_override=None,
        test_file: str = "pytest",
    ):
        self._session = TestSession(
            test_name=test_name,
            verbose=verbose,
            agent_override=agent_override,
        )
        self._agent: TestAgent | None = None
        self._test_file = test_file

    async def __aenter__(self):
        self._agent = await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.__aexit__(exc_type, exc_val, exc_tb)

        # Check if all evaluations passed - if not, fail the test
        # Always collect a structured result for terminal summary
        evaluation_results = self._session.get_results()
        server_names_str = ""
        try:
            if self._session.agent and getattr(
                self._session.agent, "server_names", None
            ):
                server_names_str = ",".join(self._session.agent.server_names)
        except Exception:
            server_names_str = ""

        test_id = generate_test_id(self._test_file, self._session.test_name)
        collected_result = TestResult(
            id=test_id,
            test_name=self._session.test_name,
            description=f"Pytest test: {self._session.test_name}",
            server_name=server_names_str,
            servers=(self._session.agent.server_names if self._session.agent else []),
            agent_name=(self._session.agent.name if self._session.agent else ""),
            parameters={},
            passed=self._session.all_passed(),
            evaluation_results=evaluation_results,
            metrics=_metrics_to_dict(self._session.get_metrics()),
            duration_ms=self._session.get_duration_ms(),
            file=self._test_file,
        )
        _pytest_results.append(collected_result)

        if not self._session.all_passed():
            # Create a TestResult object from session results for compatibility with generate_failure_message
            evaluation_results = self._session.get_results()
            test_result = TestResult(
                id=test_id,
                test_name=self._session.test_name,
                description=f"Pytest test: {self._session.test_name}",
                server_name=server_names_str,
                servers=(
                    self._session.agent.server_names if self._session.agent else []
                ),
                agent_name=(self._session.agent.name if self._session.agent else ""),
                parameters={},
                passed=False,
                evaluation_results=evaluation_results,
                metrics=None,
                duration_ms=self._session.get_duration_ms(),
                file=self._test_file,
            )
            failure_message = test_result.error or generate_failure_message(
                test_result.evaluation_results
            )
            pytest.fail(failure_message, pytrace=False)

    @property
    def agent(self) -> TestAgent | None:
        return self._agent

    @property
    def session(self) -> TestSession:
        return self._session


@pytest_asyncio.fixture
async def mcp_session(request) -> AsyncGenerator[MCPEvalPytestSession, None]:
    """Pytest fixture that provides an MCP test session.

    Usage:
        async def test_my_mcp_function(mcp_session):
            response = await mcp_session.agent.generate_str("Hello")
            mcp_session.session.evaluate_now(ResponseContains("hello"), response, "greeting")
    """
    # Touch configuration (ensures settings are loaded)
    _ = get_current_config()

    test_name = request.node.name

    # Get the test file name
    test_file = (
        Path(request.node.fspath).name if hasattr(request.node, "fspath") else "pytest"
    )

    # Check if pytest is running in verbose mode
    verbose = request.config.getoption("verbose") > 0

    # Create and yield session
    # Allow per-test markers for agents and servers
    agent_marker = request.node.get_closest_marker("mcp_agent")
    _servers_marker = request.node.get_closest_marker("mcp_servers")

    # Build session – allow agent override from marker
    agent_override = (
        agent_marker.args[0] if agent_marker and agent_marker.args else None
    )
    pytest_session_wrapper = MCPEvalPytestSession(
        test_name=test_name,
        verbose=verbose,
        agent_override=agent_override,
        test_file=test_file,
    )
    async with pytest_session_wrapper:
        yield pytest_session_wrapper
    # Cleanup happens after the context manager exits
    pytest_session_wrapper.session.cleanup()


@pytest_asyncio.fixture
async def mcp_agent(mcp_session: MCPEvalPytestSession) -> TestAgent | None:
    """Convenience fixture that provides just the agent.

    Usage:
        async def test_my_function(mcp_agent):
            response = await mcp_agent.generate_str("Hello")
            mcp_agent.session.evaluate_now(ResponseContains("hello"), response, "greeting")
    """
    return mcp_session.agent


# Alias fixture: prefer simple name `agent` to avoid confusion with mcp_agent package
@pytest_asyncio.fixture
async def agent(mcp_session: MCPEvalPytestSession) -> TestAgent | None:
    return mcp_session.agent


def pytest_configure(config):
    """Configure pytest to work with mcp-eval."""
    config.addinivalue_line("markers", "mcp-eval: mark test as an mcp-eval test")
    # Also register common alias spellings used by this plugin
    config.addinivalue_line("markers", "mcpeval: mark test as an mcp-eval test")
    config.addinivalue_line("markers", "mcp_eval: mark test as an mcp-eval test")
    config.addinivalue_line(
        "markers", "mcp_agent(name_or_object): override agent for this test"
    )
    # No per-test servers override; define servers on the agent instead

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

    # Optionally suppress pytest's short test summary if requested
    # Users can enable via: pytest --mcp-eval-summary-only
    if getattr(config.option, "mcp_eval_summary_only", False):
        # Disable extra summary sections (equivalent to -r with no chars)
        try:
            config.option.reportchars = ""
        except Exception:
            pass


def pytest_collection_modifyitems(config, items):
    """Automatically mark async tests that use mcp fixtures as mcp-eval tests."""
    for item in items:
        if hasattr(item, "function"):
            # Check if test function uses mcp fixtures
            sig = inspect.signature(item.function)
            if any(
                param in sig.parameters
                for param in ["mcp_session", "mcp_agent", "agent"]
            ):
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

        # Run any mcp-eval setup functions from the same file as the test
        # Get the test's source file
        test_file = (
            str(Path(item.fspath).resolve()) if hasattr(item, "fspath") else None
        )

        if test_file and test_file in _setup_functions:
            for setup_func in _setup_functions[test_file]:
                if not asyncio.iscoroutinefunction(setup_func):
                    setup_func()


def pytest_runtest_teardown(item):
    """Teardown for mcp-eval tests."""
    if (
        "mcpeval" in item.keywords
        or "mcp-eval" in item.keywords
        or "mcp_eval" in item.keywords
    ):
        # Run any mcp-eval teardown functions from the same file as the test
        # Get the test's source file
        test_file = (
            str(Path(item.fspath).resolve()) if hasattr(item, "fspath") else None
        )

        if test_file and test_file in _teardown_functions:
            for teardown_func in _teardown_functions[test_file]:
                if not asyncio.iscoroutinefunction(teardown_func):
                    teardown_func()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        asyncio.set_event_loop(loop)
    except Exception:
        pass
    yield loop
    # Graceful shutdown to avoid 'Event loop is closed' during async client cleanup
    # Ensure transports' __del__ run while loop is still open (avoids unraisable warnings)
    try:
        gc.collect()
    except Exception:
        pass
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    # Give any pending callbacks scheduled by transport finalizers a chance to run
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass
    # Shut down default executor threads where supported (Py3.9+)
    try:
        loop.run_until_complete(loop.shutdown_default_executor())  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass
    try:
        loop.close()
    except Exception:
        pass
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass


# Accumulate results for a richer terminal summary
_pytest_results: list[TestResult] = []


def pytest_terminal_summary(terminalreporter):
    """Render a combined MCP‑Eval summary at the end of the pytest run."""
    if not _pytest_results:
        return
    try:
        console = Console(force_terminal=True)
        # Verbose if -v was used
        verbose = terminalreporter.config.getoption("verbose") > 0
        generate_combined_summary(
            test_results=_pytest_results,
            dataset_reports=[],
            console=console,
            verbose=verbose,
        )
    except Exception:
        # Best-effort; don't break pytest summary
        pass


def pytest_addoption(parser):
    """Add CLI options for mcp-eval pytest integration."""
    group = parser.getgroup("mcp-eval")
    group.addoption(
        "--mcp-eval-summary-only",
        action="store_true",
        help="Show only the MCP-Eval combined summary and suppress pytest's short test summary info.",
    )
