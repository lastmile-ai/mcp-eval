from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.response_contains import ResponseContains
from mcp_eval.evaluators.equals_expected import EqualsExpected
from mcp_eval.evaluators.tool_called_with import ToolCalledWith
from mcp_eval.evaluators.tool_output_matches import ToolOutputMatches
from mcp_eval.evaluators.path_efficiency import PathEfficiency
from mcp_eval.metrics import TestMetrics, ToolCall


def _ctx_with_tool_calls(*calls: ToolCall):
    m = TestMetrics()
    m.tool_calls = list(calls)
    return EvaluatorContext(inputs="", output="", expected_output=None, metadata={}, metrics=m, span_tree=None)


def test_response_contains_and_equals():
    m = TestMetrics()
    ctx = EvaluatorContext(inputs="i", output="Hello World", expected_output="Hello World", metadata={}, metrics=m, span_tree=None)

    rc = ResponseContains("hello")
    assert rc.evaluate_sync(ctx).passed is True

    eq = EqualsExpected()
    assert eq.evaluate_sync(ctx).passed is True


def test_tool_called_with_and_output_matches_exact_and_contains():
    call = ToolCall(name="fetch", arguments={"url": "https://example.com"}, result={"text": "Example Domain"}, start_time=0, end_time=1)
    ctx = _ctx_with_tool_calls(call)

    tcw = ToolCalledWith("fetch", {"url": "https://example.com"})
    assert tcw.evaluate_sync(ctx).passed is True

    tom = ToolOutputMatches("fetch", expected_output="Example Domain", match_type="contains")
    assert tom.evaluate_sync(ctx).passed is True


def test_tool_output_matches_field_path_and_partial():
    result = {"result": {"status": "ok", "items": [{"name": "a"}, {"name": "b"}]}}
    call = ToolCall(name="proc", arguments={}, result=result, start_time=0, end_time=2)
    ctx = _ctx_with_tool_calls(call)

    # Field path
    tom = ToolOutputMatches("proc", expected_output="ok", field_path="result.status", match_type="exact")
    assert tom.evaluate_sync(ctx).passed is True

    # Partial list of dicts
    tom2 = ToolOutputMatches("proc", expected_output=[{"name": "b"}], field_path="result.items", match_type="partial")
    assert tom2.evaluate_sync(ctx).passed is True


def test_path_efficiency_basic_scoring():
    calls = [
        ToolCall(name="search", arguments={}, result={}, start_time=0, end_time=1),
        ToolCall(name="fetch", arguments={}, result={}, start_time=1, end_time=2),
    ]
    ctx = _ctx_with_tool_calls(*calls)
    pe = PathEfficiency(optimal_steps=2)
    res = pe.evaluate_sync(ctx)
    assert res.passed is True
    assert 0.0 <= (res.score or 0.0) <= 1.0


