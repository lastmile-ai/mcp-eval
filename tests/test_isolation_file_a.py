"""Test file A to verify setup/teardown isolation between files."""

import pytest
from mcp_eval import setup, teardown, task
from mcp_eval.core import _setup_functions, _teardown_functions
from pathlib import Path
import inspect

# Global flags to track if setup/teardown ran
setup_a_executed = False
teardown_a_executed = False


@setup
def setup_for_file_a():
    """Setup that should only run for tests in file A."""
    global setup_a_executed
    setup_a_executed = True
    print("Setup A executed")


@teardown
def teardown_for_file_a():
    """Teardown that should only run for tests in file A."""
    global teardown_a_executed
    teardown_a_executed = True
    print("Teardown A executed")


@pytest.mark.asyncio
async def test_setup_registration_file_a():
    """Test that setup is registered under this file's path."""
    source_file = str(Path(__file__).resolve())

    # Verify our setup is registered under this file
    assert source_file in _setup_functions
    assert setup_for_file_a in _setup_functions[source_file]

    # Verify our teardown is registered under this file
    assert source_file in _teardown_functions
    assert teardown_for_file_a in _teardown_functions[source_file]


@pytest.mark.asyncio
async def test_task_runs_correct_setup_a():
    """Test that only file A's setup runs for file A's tasks."""
    global setup_a_executed, teardown_a_executed

    # Reset flags
    setup_a_executed = False
    teardown_a_executed = False

    @task("Test in file A")
    async def test_task_a(agent, session):
        # Verify setup A ran
        assert setup_a_executed, "Setup A should have run"
        session.all_passed = lambda: True
        session.get_results = lambda: []
        session.get_metrics = lambda: None

    # Manually set the source file for the task
    test_task_a._source_file = str(Path(__file__).resolve())

    # Mock the session
    from unittest.mock import MagicMock, AsyncMock, patch

    with patch("mcp_eval.core.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.all_passed = lambda: True
        mock_session.get_results = lambda: []
        mock_session.get_metrics = lambda: None
        mock_session.agent = mock_agent
        mock_agent.server_names = []
        mock_agent.name = "test"
        MockSession.return_value = mock_session

        # Run the task
        result = await test_task_a()

        # Verify setup and teardown ran
        assert setup_a_executed, "Setup A should have executed"
        assert teardown_a_executed, "Teardown A should have executed"
        assert result.passed


def test_no_cross_file_contamination():
    """Verify that setups from other files are not in our list."""
    source_file = str(Path(__file__).resolve())

    # Get all registered setup functions for this file
    if source_file in _setup_functions:
        setups_for_this_file = _setup_functions[source_file]

        # All setup functions should be from this file
        for setup_func in setups_for_this_file:
            func_file = str(Path(inspect.getfile(setup_func)).resolve())
            assert (
                func_file == source_file
            ), f"Found setup from different file: {func_file}"
