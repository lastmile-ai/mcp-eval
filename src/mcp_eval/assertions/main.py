from typing import Optional

from mcp_eval.evaluators.builtin import (
    ToolWasCalled,
    ResponseContains,
    ToolSuccessRate,
    LLMJudge,
    MaxIterations,
)

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
    session.evaluate_now(evaluator, response, f"contains_{text}")


def not_contains(
    session: TestSession, response: str, text: str, case_sensitive: bool = False
):
    """Assert that response does not contain text."""
    session = _get_session() if session is None else session

    # Custom evaluator for not_contains
    class NotContains(ResponseContains):
        def evaluate_sync(self, ctx):
            return not super().evaluate_sync(ctx)

    evaluator = NotContains(text=text, case_sensitive=case_sensitive)
    session.evaluate_now(evaluator, response, f"not_contains_{text}")


def matches_regex(session: TestSession, response: str, pattern: str):
    """Assert that response matches regex pattern."""
    session = _get_session() if session is None else session
    evaluator = ResponseContains(text=pattern, regex=True)
    session.evaluate_now(evaluator, response, "matches_regex")


def tool_was_called(session: TestSession, tool_name: str, min_times: int = 1):
    """Assert that a tool was called."""
    session = _get_session() if session is None else session
    evaluator = ToolWasCalled(tool_name=tool_name, min_times=min_times)
    session.add_deferred_evaluator(evaluator, f"tool_called_{tool_name}")


def tool_was_called_with(session: TestSession, tool_name: str, arguments: dict):
    """Assert that a tool was called with specific arguments."""
    session = _get_session() if session is None else session

    # Custom evaluator for argument checking
    class ToolCalledWith(ToolWasCalled):
        def __init__(self, tool_name: str, expected_args: dict):
            super().__init__(tool_name)
            self.expected_args = expected_args

        def evaluate_sync(self, ctx):
            tool_calls = [
                call for call in ctx.tool_calls if call.name == self.tool_name
            ]
            return any(
                all(call.arguments.get(k) == v for k, v in self.expected_args.items())
                for call in tool_calls
            )

    evaluator = ToolCalledWith(tool_name, arguments)
    session.add_deferred_evaluator(evaluator, f"tool_called_with_{tool_name}")


def tool_call_count(session: TestSession, tool_name: str, expected_count: int):
    """Assert exact tool call count."""
    session = _get_session() if session is None else session

    class ExactToolCount(ToolWasCalled):
        def __init__(self, tool_name: str, expected_count: int):
            super().__init__(tool_name)
            self.expected_count = expected_count

        def evaluate_sync(self, ctx):
            tool_calls = [
                call for call in ctx.tool_calls if call.name == self.tool_name
            ]
            return len(tool_calls) == self.expected_count

    evaluator = ExactToolCount(tool_name, expected_count)
    session.add_deferred_evaluator(
        evaluator, f"tool_count_{tool_name}_{expected_count}"
    )


def tool_call_succeeded(session: TestSession, tool_name: str):
    """Assert that tool calls succeeded."""
    session = _get_session() if session is None else session
    evaluator = ToolSuccessRate(min_rate=1.0, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, f"tool_succeeded_{tool_name}")


def tool_call_failed(session: TestSession, tool_name: str):
    """Assert that tool calls failed."""
    session = _get_session() if session is None else session

    class ToolFailed(ToolSuccessRate):
        def evaluate_sync(self, ctx):
            result = super().evaluate_sync(ctx)
            return result["rate"] == 0.0  # Invert success rate

    evaluator = ToolFailed(min_rate=0.0, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, f"tool_failed_{tool_name}")


def tool_success_rate(
    session: TestSession, min_rate: float, tool_name: Optional[str] = None
):
    """Assert minimum tool success rate."""
    session = _get_session() if session is None else session
    evaluator = ToolSuccessRate(min_rate=min_rate, tool_name=tool_name)
    session.add_deferred_evaluator(evaluator, f"success_rate_{min_rate}")


def completed_within(session: TestSession, max_iterations: int):
    """Assert task completed within max iterations - explicit session passing."""
    evaluator = MaxIterations(max_iterations=max_iterations)
    session.add_deferred_evaluator(evaluator, f"max_iterations_{max_iterations}")


def response_time_under(session: TestSession, max_ms: float):
    """Assert response time is under threshold."""
    session = _get_session() if session is None else session

    class ResponseTimeCheck(MaxIterations):
        def __init__(self, max_ms: float):
            self.max_ms = max_ms

        def evaluate_sync(self, ctx):
            return ctx.metrics.latency_ms <= self.max_ms

    evaluator = ResponseTimeCheck(max_ms)
    session.add_deferred_evaluator(evaluator, f"response_time_under_{max_ms}")


async def judge(
    session: TestSession, response: str, rubric: str, min_score: float = 0.8
):
    """Use LLM to judge response quality."""
    session = _get_session() if session is None else session
    evaluator = LLMJudge(rubric=rubric, min_score=min_score)
    await session.evaluate_now_async(evaluator, response, f"judge_{rubric[:20]}")
