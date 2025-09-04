"""State isolation utilities for setup/teardown functions."""

from contextlib import contextmanager
from typing import Any, Dict
from mcp_eval.config import (
    ProgrammaticDefaults,
)
import copy


class IsolatedState:
    """Manages isolated state for test files."""

    def __init__(self):
        self.saved_states: Dict[str, Dict[str, Any]] = {}

    def save_state(self, file_path: str):
        """Save current global state for a file."""
        from mcp_eval.config import _current_settings

        self.saved_states[file_path] = {
            "agent": ProgrammaticDefaults.get_default_agent(),
            "agent_factory": ProgrammaticDefaults.get_default_agent_factory(),
            "settings": copy.deepcopy(_current_settings) if _current_settings else None,
        }

    def restore_state(self, file_path: str):
        """Restore saved state for a file."""
        if file_path not in self.saved_states:
            return

        state = self.saved_states[file_path]
        ProgrammaticDefaults.set_default_agent(state["agent"])
        ProgrammaticDefaults.set_default_agent_factory(state["agent_factory"])

        # Restore settings
        import mcp_eval.config

        if state["settings"]:
            mcp_eval.config._current_settings = state["settings"]


# Global instance
_state_manager = IsolatedState()


@contextmanager
def isolated_test_state(file_path: str):
    """Context manager to isolate state changes for a test file.

    Usage:
        with isolated_test_state(__file__):
            # Run setup
            # Run tests
            # Run teardown
        # State is restored
    """
    # Save current state
    _state_manager.save_state("__global__")

    try:
        # Load any file-specific saved state
        if file_path in _state_manager.saved_states:
            _state_manager.restore_state(file_path)

        yield

        # Save this file's state for potential reuse
        _state_manager.save_state(file_path)

    finally:
        # Restore original global state
        _state_manager.restore_state("__global__")


def get_isolated_state_manager():
    """Get the global state manager instance."""
    return _state_manager
