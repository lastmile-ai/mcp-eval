"""Core decorators and task management using unified session."""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass

from mcp_eval.session import TestSession
from mcp_eval.config import get_current_config


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    description: str
    server_name: str
    parameters: Dict[str, Any]
    passed: bool
    evaluation_results: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    duration_ms: float
    error: Optional[str] = None


# Global test configuration state
_setup_functions: List[Callable] = []
_teardown_functions: List[Callable] = []


def setup(func: Callable):
    """Register a setup function."""
    _setup_functions.append(func)
    return func


def teardown(func: Callable):
    """Register a teardown function."""
    _teardown_functions.append(func)
    return func


def parametrize(param_name: str, values: List[Any]):
    """Parametrize a test function."""

    def decorator(func):
        if not hasattr(func, "_mcpeval_parameters"):
            func._mcpeval_parameters = {}
        func._mcpeval_parameters[param_name] = values
        return func

    return decorator


def task(description: str = "", server: str = None):
    """Mark a function as an MCP evaluation task.

    The decorated function will receive (agent: TestAgent, session: TestSession)
    as arguments, making all dependencies explicit.
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
                    test_name=func.__name__,
                    description=description,
                    server_name=server_name,
                    parameters=kwargs,
                    passed=session.all_passed(),
                    evaluation_results=session.get_results(),
                    metrics=session.get_metrics().__dict__,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                return TestResult(
                    test_name=func.__name__,
                    description=description,
                    server_name=server_name,
                    parameters=kwargs,
                    passed=False,
                    evaluation_results=[],
                    metrics=None,
                    duration_ms=0,
                    error=str(e),
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
