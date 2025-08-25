"""Core decorators and task management using unified session."""

import asyncio
import inspect
import traceback
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass, asdict
from pathlib import Path

from mcp_eval.session import TestSession
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

if TYPE_CHECKING:
    from mcp_eval.evaluators import EvaluationRecord


@dataclass
class TestResult:
    """Result of a single test execution."""

    id: str
    test_name: str
    description: str
    # Back-compat single server field
    server_name: str
    # New: full server list used by the Agent
    servers: List[str]
    # New: agent identity
    agent_name: str
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


def _metrics_to_dict(metrics):
    """Convert TestMetrics object to dict, handling nested objects."""
    if not metrics:
        return None

    result = metrics.__dict__.copy()

    # Convert tool_calls list - they might already be dicts or might be ToolCall objects
    if "tool_calls" in result:
        tool_calls_list = []
        for call in result["tool_calls"]:
            if isinstance(call, dict):
                # Already a dict, just use it
                tool_calls_list.append(call)
            else:
                # It's a ToolCall dataclass, convert it
                tool_calls_list.append(asdict(call))
        result["tool_calls"] = tool_calls_list

    # Convert tool_coverage dict of ToolCoverage objects to dict
    if "tool_coverage" in result:
        coverage_dict = {}
        for server_name, coverage in result["tool_coverage"].items():
            coverage_dict[server_name] = asdict(coverage)
        result["tool_coverage"] = coverage_dict

    # Convert llm_metrics if it's an object
    if "llm_metrics" in result and hasattr(result["llm_metrics"], "__dict__"):
        result["llm_metrics"] = asdict(result["llm_metrics"])

    return result


def task(description: str = ""):
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
                # Check if function has been decorated with with_agent
                agent_override = getattr(func, "_mcpeval_agent_override", None)

                # Create unified session with optional agent override
                session = TestSession(
                    test_name=func.__name__, agent_override=agent_override
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
                result = TestResult(
                    id=test_id,
                    test_name=func.__name__,
                    description=description,
                    server_name=",".join(session.agent.server_names)
                    if session and session.agent and session.agent.server_names
                    else "",
                    servers=(
                        session.agent.server_names if session and session.agent else []
                    ),
                    agent_name=(
                        session.agent.name if session and session.agent else ""
                    ),
                    parameters=kwargs,
                    passed=session.all_passed(),
                    evaluation_results=session.get_results(),
                    metrics=_metrics_to_dict(session.get_metrics()),
                    duration_ms=duration_ms,
                    file=file_name,
                )

                # Add agent details for verbose mode
                if session and session.agent:
                    agent_details = {}
                    # Try to get agent instruction
                    if hasattr(session.agent, "instruction"):
                        agent_details["instruction"] = session.agent.instruction
                    elif hasattr(session.agent, "_instruction"):
                        agent_details["instruction"] = session.agent._instruction

                    # Get provider/model from session settings
                    if hasattr(session, "app") and session.app:
                        try:
                            from mcp_eval.config import get_settings

                            settings = get_settings()
                            if settings.provider:
                                agent_details["provider"] = settings.provider
                            if settings.model:
                                agent_details["model"] = settings.model
                        except Exception:
                            pass

                    if agent_details:
                        result._agent_details = agent_details

                return result

            except Exception:
                return TestResult(
                    id=test_id,
                    test_name=func.__name__,
                    description=description,
                    server_name=",".join(session.agent.server_names)
                    if session and session.agent and session.agent.server_names
                    else "",
                    servers=(
                        session.agent.server_names if session and session.agent else []
                    ),
                    agent_name=(
                        session.agent.name if session and session.agent else ""
                    ),
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

        return wrapper

    return decorator


def with_agent(
    agent: Agent | AugmentedLLM | AgentSpec | str | Callable[[], Agent | AugmentedLLM],
):
    """Per-test override for the agent.

    This decorator is a pure marker: it attaches the override to the function so
    that the @task decorator (or the test runner) can construct the correct
    Agent/AugmentedLLM. It deliberately does NOT create its own TestSession to
    avoid nested sessions and lost assertions when combined with @task.

    Accepts:
    - Agent instance
    - AugmentedLLM instance (its agent is used)
    - AgentSpec instance
    - AgentSpec name (string)
    - Callable that returns Agent or AugmentedLLM (factory function)
    """

    def decorator(func: Callable):
        # Attach override for the outer @task wrapper to consume
        setattr(func, "_mcpeval_agent_override", agent)
        return func

    return decorator
