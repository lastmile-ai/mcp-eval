"""Metrics collection and processing from OTEL traces."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """Represents a single tool call."""

    name: str
    arguments: Dict[str, Any]
    result: Any
    start_time: float
    end_time: float
    is_error: bool = False
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class LLMMetrics:
    """LLM usage metrics."""

    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    latency_ms: float = 0.0


@dataclass
class TestMetrics:
    """Comprehensive test metrics derived from OTEL traces."""

    # Tool usage
    tool_calls: List[ToolCall] = field(default_factory=list)
    unique_tools_used: List[str] = field(default_factory=list)

    # Execution metrics
    iteration_count: int = 0
    total_duration_ms: float = 0.0
    latency_ms: float = 0.0

    # LLM metrics
    llm_metrics: LLMMetrics = field(default_factory=LLMMetrics)

    # Performance metrics
    parallel_tool_calls: int = 0
    error_count: int = 0
    success_rate: float = 1.0

    # Cost estimation
    cost_estimate: float = 0.0


@dataclass
class TraceSpan:
    """Represents a single OTEL span from trace file."""

    name: str
    context: Dict[str, str]
    parent: Optional[Dict[str, str]]
    start_time: int  # nanoseconds since epoch
    end_time: int  # nanoseconds since epoch
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_line: str) -> "TraceSpan":
        """Create TraceSpan from JSONL line."""
        from datetime import datetime

        data = json.loads(json_line)

        # Helper to parse timestamps
        def parse_timestamp(ts):
            if isinstance(ts, (int, float)):
                return int(ts)
            elif isinstance(ts, str):
                # Parse ISO format timestamp to nanoseconds
                if "T" in ts and ts.endswith("Z"):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return int(dt.timestamp() * 1e9)
                else:
                    # Try to parse as a number string
                    return int(float(ts))
            return 0

        # Handle both standard OTEL export format and Jaeger format
        if "name" in data:
            # Standard OTEL format
            return cls(
                name=data.get("name", ""),
                context=data.get("context", {}),
                parent=data.get("parent"),
                start_time=parse_timestamp(data.get("start_time", 0)),
                end_time=parse_timestamp(data.get("end_time", 0)),
                attributes=data.get("attributes", {}),
                events=data.get("events", []),
            )
        else:
            # Jaeger format fallback (from original implementation)
            return cls(
                name=data.get("operationName", ""),
                context={
                    "span_id": data.get("spanID", ""),
                    "trace_id": data.get("traceID", ""),
                },
                parent=data.get("references", [{}])[0]
                if data.get("references")
                else None,
                start_time=parse_timestamp(data.get("startTime", 0)),
                end_time=parse_timestamp(data.get("startTime", 0))
                + data.get("duration", 0),
                attributes=data.get("tags", {}),
                events=data.get("logs", []),
            )


def process_spans(spans: List[TraceSpan]) -> TestMetrics:
    """Process OTEL spans into comprehensive metrics."""
    metrics = TestMetrics()

    if not spans:
        return metrics

    # Calculate total duration
    if spans:
        start_times = [span.start_time for span in spans]
        end_times = [span.end_time for span in spans]
        metrics.total_duration_ms = (max(end_times) - min(start_times)) / 1e6

    # Process tool calls
    tool_calls = []
    for span in spans:
        if _is_tool_call_span(span):
            tool_call = _extract_tool_call(span)
            if tool_call:
                tool_calls.append(tool_call)

    metrics.tool_calls = tool_calls
    metrics.unique_tools_used = list(set(call.name for call in tool_calls))

    # Calculate error metrics
    error_calls = [call for call in tool_calls if call.is_error]
    metrics.error_count = len(error_calls)
    metrics.success_rate = (
        1.0 - (len(error_calls) / len(tool_calls)) if tool_calls else 1.0
    )

    # Process LLM metrics
    llm_spans = [span for span in spans if _is_llm_span(span)]
    if llm_spans:
        metrics.llm_metrics = _extract_llm_metrics(llm_spans)

    # Calculate iteration count (number of agent turns)
    agent_spans = [span for span in spans if "agent" in span.name.lower()]
    metrics.iteration_count = len(agent_spans)

    # Calculate parallel tool calls
    metrics.parallel_tool_calls = _calculate_parallel_calls(tool_calls)

    # Aggregate latency
    if tool_calls:
        metrics.latency_ms = sum(call.duration_ms for call in tool_calls)

    # Cost estimation
    metrics.cost_estimate = _estimate_cost(metrics.llm_metrics)

    return metrics


def _is_tool_call_span(span: TraceSpan) -> bool:
    """Determine if span represents a tool call."""
    return (
        "tool" in span.name.lower()
        or "call_tool" in span.name
        or span.attributes.get("mcp.tool.name") is not None
    )


def _is_llm_span(span: TraceSpan) -> bool:
    """Determine if span represents an LLM call."""
    return (
        span.attributes.get("gen_ai.system") is not None
        or "llm" in span.name.lower()
        or "generate" in span.name.lower()
    )


def _extract_tool_call(span: TraceSpan) -> Optional[ToolCall]:
    """Extract tool call information from span."""
    try:
        tool_name = (
            span.attributes.get("mcp.tool.name")
            or span.attributes.get("tool.name")
            or span.name.replace("call_tool_", "").replace("tool_", "")
        )

        arguments = span.attributes.get("mcp.tool.arguments", {})
        result = span.attributes.get("mcp.tool.result")

        is_error = (
            span.attributes.get("error") is not None
            or span.attributes.get("mcp.tool.error") is not None
            or any(event.get("level") == "error" for event in span.events)
        )

        error_message = span.attributes.get("error.message")

        return ToolCall(
            name=tool_name,
            arguments=arguments,
            result=result,
            start_time=span.start_time / 1e9,
            end_time=span.end_time / 1e9,
            is_error=is_error,
            error_message=error_message,
        )
    except Exception:
        return None


def _extract_llm_metrics(llm_spans: List[TraceSpan]) -> LLMMetrics:
    """Extract LLM metrics from spans."""
    metrics = LLMMetrics()

    for span in llm_spans:
        attrs = span.attributes

        # Model information
        if not metrics.model_name:
            metrics.model_name = attrs.get("gen_ai.request.model", "")

        # Token usage
        metrics.input_tokens += attrs.get("gen_ai.usage.input_tokens", 0)
        metrics.output_tokens += attrs.get("gen_ai.usage.output_tokens", 0)

        # Latency
        duration_ms = (span.end_time - span.start_time) / 1e6
        metrics.latency_ms += duration_ms

    metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
    return metrics


def _calculate_parallel_calls(tool_calls: List[ToolCall]) -> int:
    """Calculate maximum number of parallel tool calls."""
    if len(tool_calls) <= 1:
        return 0

    events = []
    for call in tool_calls:
        events.append(("start", call.start_time))
        events.append(("end", call.end_time))

    events.sort(key=lambda x: x[1])

    max_concurrent = 0
    current_concurrent = 0

    for event_type, _ in events:
        if event_type == "start":
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        else:
            current_concurrent -= 1

    return max_concurrent - 1


def _estimate_cost(llm_metrics: LLMMetrics) -> float:
    """Estimate cost based on token usage."""
    # Simple cost estimation - would be configurable in real implementation
    cost_per_input_token = 0.000001
    cost_per_output_token = 0.000003

    return (
        llm_metrics.input_tokens * cost_per_input_token
        + llm_metrics.output_tokens * cost_per_output_token
    )


# Metric registration for extensibility
_custom_metrics: Dict[str, callable] = {}


def register_metric(name: str, processor: callable):
    """Register a custom metric processor."""
    _custom_metrics[name] = processor
