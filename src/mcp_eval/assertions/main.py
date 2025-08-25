from typing import Literal, Optional, Union, Any, Dict, List, Pattern

from mcp_eval.evaluators import (
    ExactToolCount,
    NotContains,
    ResponseTimeCheck,
    ToolCalledWith,
    ToolFailed,
    ToolWasCalled,
    ResponseContains,
    ToolSuccessRate,
    LLMJudge,
    MaxIterations,
)
from mcp_eval.evaluators.tool_output_matches import ToolOutputMatches
from mcp_eval.evaluators.path_efficiency import PathEfficiency

from mcp_eval.session import TestSession

# Thread-local session context (legacy approach)
import threading

_local = threading.local()


def _get_session():
    """Get current test session from thread-local storage."""
    if not hasattr(_local, "session"):
        raise RuntimeError(
            "No active test session. Use @task decorator or with test_session()."
        )
    return _local.session


def _set_session(session):
    """Set current test session in thread-local storage."""
    _local.session = session


def contains(
    session: TestSession, response: str, text: str, case_sensitive: bool = False
):
    """Assert that response contains text."""
    session = _get_session() if session is None else session
    evaluator = ResponseContains(text=text, case_sensitive=case_sensitive)
    session.evaluate_now(evaluator, response, name=f"contains_{text}")


def not_contains(
    session: TestSession, response: str, text: str, case_sensitive: bool = False
):
    """Assert that response does not contain text."""
    session = _get_session() if session is None else session
    evaluator = NotContains(text=text, case_sensitive=case_sensitive)
    session.evaluate_now(evaluator, response, name=f"not_contains_{text}")


def matches_regex(session: TestSession, response: str, pattern: str):
    """Assert that response matches regex pattern."""
    session = _get_session() if session is None else session
    evaluator = ResponseContains(text=pattern, regex=True)
    session.evaluate_now(evaluator, response, name="matches_regex")


def tool_was_called(session: TestSession, tool_name: str, min_times: int = 1):
    """Assert that a tool was called."""
    session = _get_session() if session is None else session
    evaluator = ToolWasCalled(tool_name=tool_name, min_times=min_times)
    session.add_deferred_evaluator(evaluator, name=f"tool_called_{tool_name}")


def tool_was_called_with(session: TestSession, tool_name: str, arguments: dict):
    """Assert that a tool was called with specific arguments."""
    session = _get_session() if session is None else session
    evaluator = ToolCalledWith(tool_name, arguments)
    session.add_deferred_evaluator(evaluator, name=f"tool_called_with_{tool_name}")


def tool_call_count(session: TestSession, tool_name: str, expected_count: int):
    """Assert exact tool call count."""
    session = _get_session() if session is None else session
    evaluator = ExactToolCount(tool_name, expected_count)
    session.add_deferred_evaluator(
        evaluator, name=f"tool_count_{tool_name}_{expected_count}"
    )


def tool_call_succeeded(session: TestSession, tool_name: str):
    """Assert that tool calls succeeded."""
    session = _get_session() if session is None else session
    evaluator = ToolSuccessRate(min_rate=1.0, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, name=f"tool_succeeded_{tool_name}")


def tool_call_failed(session: TestSession, tool_name: str):
    """Assert that tool calls failed."""
    session = _get_session() if session is None else session
    evaluator = ToolFailed(min_rate=0.0, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, name=f"tool_failed_{tool_name}")


def tool_success_rate(
    session: TestSession, min_rate: float, tool_name: str | None = None
):
    """Assert minimum tool success rate."""
    session = _get_session() if session is None else session
    evaluator = ToolSuccessRate(min_rate=min_rate, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, name=f"success_rate_{min_rate}")


def completed_within(session: TestSession, max_iterations: int):
    """Assert task completed within max iterations - explicit session passing."""
    evaluator = MaxIterations(max_iterations=max_iterations)
    session.add_deferred_evaluator(evaluator, name=f"max_iterations_{max_iterations}")


def response_time_under(session: TestSession, max_ms: float):
    """Assert response time is under threshold."""
    session = _get_session() if session is None else session
    evaluator = ResponseTimeCheck(max_ms)
    session.add_deferred_evaluator(evaluator, name=f"response_time_under_{max_ms}")


async def judge(
    session: TestSession, response: str, rubric: str, min_score: float = 0.8
):
    """Use LLM to judge response quality."""
    session = _get_session() if session is None else session
    evaluator = LLMJudge(rubric=rubric, min_score=min_score)
    # Use unified API; it will schedule async eval now and await at session end.
    session.evaluate_now(evaluator, response, name=f"judge_{rubric[:20]}")


def tool_output_matches(
    session: TestSession,
    tool_name: str,
    expected_output: Union[Dict[str, Any], str, Pattern, int, float, List[Any]],
    field_path: str | None = None,
    match_type: Literal["exact", "contains", "regex", "partial"] = "exact",
    case_sensitive: bool = True,
    call_index: int = -1,
):
    """Assert that tool output matches expected pattern.

    Args:
        session: Test session to use for evaluation
        tool_name: Name of the tool to check
        expected_output: Expected output value or pattern
        field_path: Optional path to extract nested field (e.g., "content.text", "[0].message")
        match_type: Type of matching - "exact", "contains", "regex", or "partial"
        case_sensitive: Whether string matching should be case sensitive
        call_index: Which tool call to check (-1 for last, 0 for first)

    Examples:
        ```
        # Exact match on full output
        tool_output_matches(session, "read_file", "Hello world")

        # Match substring in output
        tool_output_matches(session, "search", "found", match_type="contains")

        # Regex match
        tool_output_matches(session, "validate", r"\\d+", match_type="regex")

        # Extract nested field and match
        tool_output_matches(session, "api_call", "success", field_path="result.status")

        # Partial dict matching
        tool_output_matches(session, "get_config", {"debug": True}, match_type="partial")
        ```
    """
    session = _get_session() if session is None else session
    evaluator = ToolOutputMatches(
        tool_name=tool_name,
        expected_output=expected_output,
        field_path=field_path,
        match_type=match_type,
        case_sensitive=case_sensitive,
        call_index=call_index,
    )
    session.add_deferred_evaluator(evaluator, name=f"tool_output_{tool_name}")


def path_efficiency(
    session: TestSession,
    optimal_steps: int | None = None,
    expected_tool_sequence: List[str] | None = None,
    golden_path: List[str] | None = None,
    allow_extra_steps: int = 0,
    penalize_backtracking: bool = True,
    penalize_repeated_tools: bool = True,
    tool_usage_limits: Dict[str, int] | None = None,
    default_tool_limit: int = 1,
):
    """Assert that agent took an efficient path to complete the task.

    Args:
        session: Test session to use for evaluation
        optimal_steps: Expected optimal number of steps (auto-calculated if None)
        expected_tool_sequence: Expected sequence of tool calls
        allow_extra_steps: Tolerance for additional steps beyond optimal
        penalize_backtracking: Whether to penalize returning to previous tools
        penalize_repeated_tools: Whether to penalize excessive tool repetition
        tool_usage_limits: Custom limits per tool (e.g., {"read": 2, "write": 1})
        default_tool_limit: Default limit for tools not in tool_usage_limits

    Examples:
        ```
        # Basic efficiency check with auto-calculated optimal steps
        path_efficiency(session)

        # Check with specific expected sequence
        path_efficiency(session, expected_tool_sequence=["read", "analyze", "write"])

        # Allow some tolerance
        path_efficiency(session, optimal_steps=5, allow_extra_steps=2)

        # Disable backtracking penalty for exploratory tasks
        path_efficiency(session, penalize_backtracking=False)

        # Custom tool usage limits
        path_efficiency(session, tool_usage_limits={"read": 3, "write": 1}, default_tool_limit=2)
        ```
    """
    session = _get_session() if session is None else session
    evaluator = PathEfficiency(
        optimal_steps=optimal_steps,
        expected_tool_sequence=expected_tool_sequence,
        golden_path=golden_path,
        allow_extra_steps=allow_extra_steps,
        penalize_backtracking=penalize_backtracking,
        penalize_repeated_tools=penalize_repeated_tools,
        tool_usage_limits=tool_usage_limits,
        default_tool_limit=default_tool_limit,
    )
    session.add_deferred_evaluator(evaluator, name="path_efficiency")
