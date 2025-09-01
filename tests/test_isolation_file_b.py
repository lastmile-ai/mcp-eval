"""Test file B to verify setup/teardown isolation between files."""

import pytest
from mcp_eval import setup, teardown, task
from mcp_eval.core import _setup_functions, _teardown_functions
from pathlib import Path
import inspect

# Global flags to track if setup/teardown ran
setup_b_executed = False
teardown_b_executed = False

# This flag should never be set if isolation works
setup_a_was_detected = False


@setup
def setup_for_file_b():
    """Setup that should only run for tests in file B."""
    global setup_b_executed, setup_a_was_detected
    setup_b_executed = True
    print("Setup B executed")

    # Check if setup A is trying to run (it shouldn't)
    # We detect this by checking if any setup from file A is in our execution
    source_file = str(Path(__file__).resolve())
    for file_path, funcs in _setup_functions.items():
        if file_path != source_file and "test_isolation_file_a" in file_path:
            # If we find file A's setups registered, that's expected
            # But they shouldn't run for our tests
            pass


@teardown
def teardown_for_file_b():
    """Teardown that should only run for tests in file B."""
    global teardown_b_executed
    teardown_b_executed = True
    print("Teardown B executed")


@pytest.mark.asyncio
async def test_setup_registration_file_b():
    """Test that setup is registered under this file's path."""
    source_file = str(Path(__file__).resolve())

    # Verify our setup is registered under this file
    assert source_file in _setup_functions
    assert setup_for_file_b in _setup_functions[source_file]

    # Verify our teardown is registered under this file
    assert source_file in _teardown_functions
    assert teardown_for_file_b in _teardown_functions[source_file]


@pytest.mark.asyncio
async def test_task_runs_correct_setup_b():
    """Test that only file B's setup runs for file B's tasks."""
    global setup_b_executed, teardown_b_executed

    # Reset flags
    setup_b_executed = False
    teardown_b_executed = False

    @task("Test in file B")
    async def test_task_b(agent, session):
        # Verify setup B ran
        assert setup_b_executed, "Setup B should have run"
        session.all_passed = lambda: True
        session.get_results = lambda: []
        session.get_metrics = lambda: None

    # Manually set the source file for the task
    test_task_b._source_file = str(Path(__file__).resolve())

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
        result = await test_task_b()

        # Verify setup and teardown ran
        assert setup_b_executed, "Setup B should have executed"
        assert teardown_b_executed, "Teardown B should have executed"
        assert result.passed


def test_file_b_not_affected_by_file_a():
    """Test that file A's setups don't affect file B."""
    source_file = str(Path(__file__).resolve())

    # Check our setup list only contains functions from this file
    if source_file in _setup_functions:
        for setup_func in _setup_functions[source_file]:
            func_file = str(Path(inspect.getfile(setup_func)).resolve())
            assert (
                func_file == source_file
            ), f"Found setup from different file: {func_file}"
            assert (
                "test_isolation_file_a" not in func_file
            ), "File A's setup found in File B's list"


def test_both_files_have_separate_registrations():
    """Verify both files can have their own setup/teardown without interference."""
    # Both files should be able to register their own functions
    file_a_found = False
    file_b_found = False

    for file_path in _setup_functions.keys():
        if "test_isolation_file_a" in file_path:
            file_a_found = True
        if "test_isolation_file_b" in file_path:
            file_b_found = True

    # After both files are imported, both should have registrations
    # But they should be separate
    if file_a_found and file_b_found:
        # Verify they have different setup functions
        for file_a_path, file_a_funcs in _setup_functions.items():
            if "test_isolation_file_a" in file_a_path:
                for file_b_path, file_b_funcs in _setup_functions.items():
                    if "test_isolation_file_b" in file_b_path:
                        # The function lists should be completely different
                        assert set(file_a_funcs).isdisjoint(
                            set(file_b_funcs)
                        ), "File A and File B share setup functions!"
