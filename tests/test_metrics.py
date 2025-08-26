from mcp_eval.metrics import (
    TraceSpan,
    process_spans,
)


def _span(
    name: str, start: int, end: int, attrs=None, events=None, context=None, parent=None
):
    return TraceSpan(
        name=name,
        context=context or {"span_id": name, "trace_id": "t"},
        parent=parent or None,
        start_time=start,
        end_time=end,
        attributes=attrs or {},
        events=events or [],
    )


def test_process_spans_extracts_tool_and_llm_metrics():
    # Tool call with arguments.* and result.* attributes
    tool_attrs = {
        "parsed_tool_name": "fetch",
        "parsed_server_name": "serverA",
        "arguments.url": "https://example.com",
        "arguments.max_length": 100,
        "result.text": "Example Domain",
        "result.isError": False,
    }
    tool = _span("MCPAggregator.call_tool", 0, 1_000_000, attrs=tool_attrs)

    # LLM call spans with usage tokens
    llm1 = _span(
        "llm.generate",
        2_000_000,
        3_000_000,
        attrs={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "test-model",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
        },
    )
    llm2 = _span(
        "llm.generate",
        3_000_000,
        3_500_000,
        attrs={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "test-model",
            "gen_ai.usage.input_tokens": 7,
            "gen_ai.usage.output_tokens": 3,
        },
    )

    metrics = process_spans([tool, llm1, llm2])

    # Tool calls extracted
    assert len(metrics.tool_calls) == 1
    call = metrics.tool_calls[0]
    assert call.name == "fetch"
    assert call.arguments.get("url") == "https://example.com"
    assert call.result.get("text") == "Example Domain"
    assert call.server_name == "serverA"
    assert metrics.unique_tools_used == ["fetch"]
    assert metrics.unique_servers_used == ["serverA"]

    # LLM metrics aggregated
    assert metrics.llm_metrics.total_tokens == 25
    assert metrics.iteration_count >= 2
    assert metrics.llm_time_ms > 0
