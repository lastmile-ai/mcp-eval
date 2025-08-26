from datetime import datetime, timedelta

from mcp_eval.otel.span_tree import SpanNode, SpanTree


def _mk_span(id: str, name: str, start: datetime, dur_ms: int, attrs=None, events=None, parent_id=None):
    return SpanNode(
        span_id=id,
        name=name,
        start_time=start,
        end_time=start + timedelta(milliseconds=dur_ms),
        attributes=attrs or {},
        events=events or [],
        parent_id=parent_id,
        children=[],
    )


def test_span_tree_find_and_metrics():
    t0 = datetime.utcnow()
    root = _mk_span("r", "root", t0, 100)
    a = _mk_span("a", "call_tool read", t0, 30, attrs={"status.code": "OK"}, parent_id="r")
    b = _mk_span("b", "llm.generate", t0 + timedelta(milliseconds=10), 40, attrs={"gen_ai.system": "x"}, parent_id="r")
    c = _mk_span("c", "call_tool write", t0 + timedelta(milliseconds=20), 50, attrs={"result.isError": True}, parent_id="r")
    root.children = [a, b, c]
    tree = SpanTree(root)

    # Queries
    assert tree.any({"name_contains": "call_tool"})
    errs = tree.find({"has_error": True})
    assert len(errs) == 1 and errs[0].span_id == "c"

    perf = tree.analyze_performance()
    assert perf["tool_call_count"] >= 2
    assert perf["llm_call_count"] == 1
    assert perf["error_count"] == 1


