"""Core decorators and task management using unified session."""

import asyncio
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass

from mcp_eval.session import TestSession
from mcp_eval.config import get_current_config
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

if TYPE_CHECKING:
    from mcp_eval.evaluators import EvaluationRecord


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    description: str
    server_name: str
    parameters: Dict[str, Any]
    passed: bool
    evaluation_results: List["EvaluationRecord"]
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


def task(description: str = ""):
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
                agent_config = config.get("agent_config", {})

                # Create unified session – servers come from config defaults unless overridden via with_agent
                session = TestSession(
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
                    server_name=",".join(session.agent.agent.server_names)
                    if session.agent
                    and getattr(session.agent, "agent", None)
                    and getattr(session.agent.agent, "server_names", None)
                    else "",
                    parameters=kwargs,
                    passed=session.all_passed(),
                    evaluation_results=session.get_results(),
                    metrics=session.get_metrics().__dict__,
                    duration_ms=duration_ms,
                )

            except Exception:
                return TestResult(
                    test_name=func.__name__,
                    description=description,
                    server_name=",".join(session.agent.agent.server_names)
                    if session
                    and getattr(session, "agent", None)
                    and getattr(session.agent, "agent", None)
                    and getattr(session.agent.agent, "server_names", None)
                    else "",
                    parameters=kwargs,
                    passed=False,
                    evaluation_results=session.get_results() if session else [],
                    metrics=None,
                    duration_ms=0,
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

        return wrapper

    return decorator


def with_agent(agent: Agent | AugmentedLLM | AgentSpec | str):
    """Per-test override for the agent.

    Accepts:
    - Agent instance
    - AugmentedLLM instance (its agent is used)
    - AgentSpec instance
    - AgentSpec name (string)
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build a session using the override agent/LLM/spec
            config = get_current_config()
            session = TestSession(
                test_name=func.__name__,
                agent_config=config.get("agent_config", {}),
                agent_override=agent,  # type: ignore[arg-type]
            )

            async with session as test_agent:
                sig = inspect.signature(func)
                if "session" in sig.parameters and "agent" in sig.parameters:
                    return await func(test_agent, session, **kwargs)
                if "agent" in sig.parameters:
                    return await func(test_agent, **kwargs)
                if "session" in sig.parameters:
                    return await func(session, **kwargs)
                return await func(**kwargs)

        return wrapper

    return decorator
