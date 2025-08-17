"""Core decorators and task management using unified session."""

import asyncio
import inspect
import traceback
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from pathlib import Path

from mcp_eval.session import TestSession
from mcp_eval.config import get_current_config

if TYPE_CHECKING:
    from mcp_eval.evaluators import EvaluationRecord


@dataclass
class TestResult:
    """Result of a single test execution."""

    id: str
    test_name: str
    description: str
    server_name: str
    parameters: Dict[str, Any]
    passed: bool
    evaluation_results: List["EvaluationRecord"]
    metrics: Optional[Dict[str, Any]]
    duration_ms: float
    file: str
    error: Optional[str] = None


# Global test configuration state
_setup_functions: List[Callable] = []
_teardown_functions: List[Callable] = []


def generate_test_id(file: str, test_name: str) -> str:
    """Generate a unique test ID from file and test name."""
    # Generate 20-char hash for file
    file_hash = hashlib.sha256(file.encode()).hexdigest()[:20]
    # Generate 20-char hash for test_name
    name_hash = hashlib.sha256(test_name.encode()).hexdigest()[:20]
    return f"{file_hash}-{name_hash}"


def setup(func: Callable):
    """Register a setup function."""
    _setup_functions.append(func)
    return func


def teardown(func: Callable):
    """Register a teardown function."""
    _teardown_functions.append(func)
    return func


def parametrize(param_names: str, values: List[Any]):
    """Parametrize a test function.

    Args:
        param_names: Comma-separated parameter names
        values: List of tuples/values for each parameter combination
    """

    def decorator(func):
        # Store parameter combinations for pytest-style parametrization
        func._mcpeval_param_combinations = []

        # Parse parameter names
        names = [name.strip() for name in param_names.split(",")]

        # Create parameter combinations
        for value in values:
            if len(names) == 1:
                # Single parameter case
                func._mcpeval_param_combinations.append({names[0]: value})
            else:
                # Multiple parameters case - unpack tuple
                if isinstance(value, (tuple, list)) and len(value) == len(names):
                    func._mcpeval_param_combinations.append(dict(zip(names, value)))
                else:
                    raise ValueError(
                        f"Parameter count mismatch: expected {len(names)} values, got {len(value) if isinstance(value, (tuple, list)) else 1}"
                    )

        return func

    return decorator


def task(description: str = "", server: str = None):
    """Mark a function as an MCP evaluation task.

    The decorated function will receive (agent: TestAgent, session: TestSession)
    as arguments, making all dependencies explicit.

    Args:
        description (str, optional): The description of the evaluation task. Defaults to "".
        server (str, optional): Name of the MCP server. Defaults to None.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Run setup functions
            for setup_func in _setup_functions:
                if asyncio.iscoroutinefunction(setup_func):
                    await setup_func()
                else:
                    setup_func()

            # Get file name from the wrapper function (set during discovery)
            source_file = getattr(wrapper, "_source_file", None)
            file_name = Path(source_file).name if source_file else "unknown"
            test_id = generate_test_id(file_name, func.__name__)

            try:
                # Get configuration
                config = get_current_config()
                server_name = server or config.get("default_server", "default")
                agent_config = config.get("agent_config", {})

                # Create unified session
                session = TestSession(
                    server_name=server_name,
                    test_name=func.__name__,
                    agent_config=agent_config,
                )

                start_time = asyncio.get_event_loop().time()

                async with session as test_agent:
                    # Call the test function with explicit arguments
                    sig = inspect.signature(func)
                    if "session" in sig.parameters and "agent" in sig.parameters:
                        await func(test_agent, session, **kwargs)
                    elif "agent" in sig.parameters:
                        await func(test_agent, **kwargs)
                    elif "session" in sig.parameters:
                        await func(session, **kwargs)
                    else:
                        await func(**kwargs)

                end_time = asyncio.get_event_loop().time()
                duration_ms = (end_time - start_time) * 1000

                # Create result from session
                return TestResult(
                    id=test_id,
                    test_name=func.__name__,
                    description=description,
                    server_name=server_name,
                    parameters=kwargs,
                    passed=session.all_passed(),
                    evaluation_results=session.get_results(),
                    metrics=session.get_metrics().__dict__,
                    duration_ms=duration_ms,
                    file=file_name,
                )

            except Exception:
                return TestResult(
                    id=test_id,
                    test_name=func.__name__,
                    description=description,
                    server_name=server_name,
                    parameters=kwargs,
                    passed=False,
                    evaluation_results=session.get_results() if session else [],
                    metrics=None,
                    duration_ms=0,
                    file=file_name,
                    error=traceback.format_exc(),
                )

            finally:
                # Run teardown functions
                for teardown_func in _teardown_functions:
                    if asyncio.iscoroutinefunction(teardown_func):
                        await teardown_func()
                    else:
                        teardown_func()

        # Mark as MCP eval task
        wrapper._is_mcpeval_task = True
        wrapper._description = description
        wrapper._server = server

        return wrapper

    return decorator
