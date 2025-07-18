"""Metrics collection and processing from OTEL traces."""

import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field


def unflatten_attributes(attributes: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Unflatten values from span attributes with dot notation support.

    Args:
        attributes: Span attributes dictionary
        prefix: Prefix to look for in attribute keys

    Returns:
        Nested dictionary with unflattened values
    """
    arguments = {}
    for full_key, value in attributes.items():
        if not full_key.startswith(prefix):
            continue

        # strip prefix and split into path components
        keys = full_key[len(prefix) :].split(".")
        current = arguments

        # for each key except the last, descend (or create) a dict
        for part in keys[:-1]:
            current = current.setdefault(part, {})
        # assign the final value
        current[keys[-1]] = value

    return arguments


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
class MessageInfo:
    """Information about a specific message or communication."""
    
    content: str
    sender: str  # who sent/outputted the message
    recipient: Optional[str] = None  # who received the message
    timestamp: float = 0.0  # timestamp for ordering
    end_timestamp: Optional[float] = None  # end timestamp if applicable
    message_type: str = ""  # e.g., "user_task", "server_init", "llm_request", "llm_response"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceInformation:
    """Comprehensive information extracted from trace spans."""
    
    # Server and initialization information
    server_initialization: List[MessageInfo] = field(default_factory=list)
    
    # User tasks and requests
    user_tasks: List[MessageInfo] = field(default_factory=list)
    
    # Agent information and activities
    agent_conversations: List[MessageInfo] = field(default_factory=list)
    
    # LLM interactions
    llm_requests: List[MessageInfo] = field(default_factory=list)
    llm_responses: List[MessageInfo] = field(default_factory=list)
    
    # Tool calls and results
    tool_interactions: List[MessageInfo] = field(default_factory=list)
    
    # System events and logs
    system_events: List[MessageInfo] = field(default_factory=list)
    
    # Error messages and warnings
    errors_warnings: List[MessageInfo] = field(default_factory=list)
    
    # General span information
    all_spans: List[TraceSpan] = field(default_factory=list)


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
        span.attributes.get("mcp.tool.name") is not None
        # # TODO: jerron - This leads to duplicates from *.call_tool
        # # but neccessary for non-MCP tools
        # or span.attributes.get("gen_ai.tool.name") is not None
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

        # Extract arguments using the unflatten utility
        # TODO: jerron - Unflattened result.content is a dict instead of list
        arguments = unflatten_attributes(span.attributes, "mcp.request.argument.")
        result = unflatten_attributes(span.attributes, "result.")

        is_error = span.attributes.get("result.isError", False)

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


def extract_comprehensive_trace_information(spans: List[TraceSpan]) -> TraceInformation:
    """Extract comprehensive information from trace spans including messages, conversations, and system events.
    
    Args:
        spans: List of TraceSpan objects extracted from trace jsonl file
        
    Returns:
        TraceInformation object containing categorized information with timestamps
    """
    trace_info = TraceInformation()
    trace_info.all_spans = spans
    
    for span in spans:
        timestamp = span.start_time / 1e9  # Convert nanoseconds to seconds
        end_timestamp = span.end_time / 1e9
        
        # Extract sender/recipient information from span attributes
        sender = _extract_sender_info(span)
        recipient = _extract_recipient_info(span)
        
        # Server initialization information
        if _is_server_init_span(span):
            content = _extract_server_init_content(span)
            trace_info.server_initialization.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="server_init",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # User tasks and requests
        elif _is_user_task_span(span):
            content = _extract_user_task_content(span)
            trace_info.user_tasks.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="user_task",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # Agent conversations
        elif _is_agent_conversation_span(span):
            content = _extract_agent_conversation_content(span)
            trace_info.agent_conversations.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="agent_conversation",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # LLM requests
        elif _is_llm_request_span(span):
            content = _extract_llm_request_content(span)
            trace_info.llm_requests.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="llm_request",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # LLM responses
        elif _is_llm_response_span(span):
            content = _extract_llm_response_content(span)
            trace_info.llm_responses.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="llm_response",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # Tool interactions
        elif _is_tool_call_span(span):
            content = _extract_tool_interaction_content(span)
            trace_info.tool_interactions.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="tool_interaction",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # System events
        elif _is_system_event_span(span):
            content = _extract_system_event_content(span)
            trace_info.system_events.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="system_event",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
        
        # Errors and warnings
        if _has_error_or_warning(span):
            content = _extract_error_warning_content(span)
            trace_info.errors_warnings.append(MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type="error_warning",
                metadata={"span_name": span.name, "attributes": span.attributes}
            ))
    
    # Sort all lists by timestamp for chronological order
    _sort_messages_by_timestamp(trace_info)
    
    return trace_info


def _extract_sender_info(span: TraceSpan) -> str:
    """Extract sender information from span attributes."""
    # Look for common sender attributes
    sender_attrs = [
        "mcp.client.name", "client.name", "agent.name", "service.name",
        "user.name", "system.name", "llm.provider", "tool.executor"
    ]
    
    for attr in sender_attrs:
        if attr in span.attributes:
            return str(span.attributes[attr])
    
    # Fall back to span name analysis
    if "agent" in span.name.lower():
        return "agent"
    elif "llm" in span.name.lower() or "generate" in span.name.lower():
        return "llm"
    elif "tool" in span.name.lower():
        return "tool_executor"
    elif "server" in span.name.lower():
        return "server"
    elif "user" in span.name.lower():
        return "user"
    
    return "unknown"


def _extract_recipient_info(span: TraceSpan) -> Optional[str]:
    """Extract recipient information from span attributes."""
    recipient_attrs = [
        "mcp.server.name", "server.name", "target.name", "destination.name"
    ]
    
    for attr in recipient_attrs:
        if attr in span.attributes:
            return str(span.attributes[attr])
    
    return None


def _is_server_init_span(span: TraceSpan) -> bool:
    """Check if span represents server initialization."""
    return (
        "init" in span.name.lower() or
        "start" in span.name.lower() or
        "setup" in span.name.lower() or
        span.attributes.get("mcp.server.status") == "initializing"
    )


def _is_user_task_span(span: TraceSpan) -> bool:
    """Check if span represents a user task or request."""
    return (
        "user" in span.name.lower() or
        "task" in span.name.lower() or
        "request" in span.name.lower() or
        span.attributes.get("message.type") == "user_input"
    )


def _is_agent_conversation_span(span: TraceSpan) -> bool:
    """Check if span represents agent conversation."""
    return (
        "agent" in span.name.lower() or
        "conversation" in span.name.lower() or
        span.attributes.get("message.sender") == "agent"
    )


def _is_llm_request_span(span: TraceSpan) -> bool:
    """Check if span represents an LLM request."""
    return (
        span.attributes.get("gen_ai.operation.name") == "chat" or
        span.attributes.get("gen_ai.request.model") is not None or
        ("llm" in span.name.lower() and "request" in span.name.lower())
    )


def _is_llm_response_span(span: TraceSpan) -> bool:
    """Check if span represents an LLM response."""
    return (
        span.attributes.get("gen_ai.response.model") is not None or
        ("llm" in span.name.lower() and "response" in span.name.lower()) or
        span.attributes.get("gen_ai.usage.output_tokens") is not None
    )


def _is_system_event_span(span: TraceSpan) -> bool:
    """Check if span represents a system event."""
    return (
        "system" in span.name.lower() or
        "event" in span.name.lower() or
        "log" in span.name.lower() or
        span.attributes.get("event.type") is not None
    )


def _has_error_or_warning(span: TraceSpan) -> bool:
    """Check if span contains error or warning information."""
    return (
        span.attributes.get("error") is not None or
        span.attributes.get("error.message") is not None or
        span.attributes.get("warning") is not None or
        span.attributes.get("result.isError") is True or
        "error" in span.name.lower() or
        "warning" in span.name.lower()
    )


def _extract_server_init_content(span: TraceSpan) -> str:
    """Extract server initialization content from span."""
    content_parts = []
    
    if span.attributes.get("mcp.server.name"):
        content_parts.append(f"Server: {span.attributes['mcp.server.name']}")
    
    if span.attributes.get("server.version"):
        content_parts.append(f"Version: {span.attributes['server.version']}")
    
    if span.attributes.get("server.capabilities"):
        content_parts.append(f"Capabilities: {span.attributes['server.capabilities']}")
    
    # Check events for initialization messages
    for event in span.events:
        if event.get("name") == "log" and event.get("attributes", {}).get("message"):
            content_parts.append(event["attributes"]["message"])
    
    return " | ".join(content_parts) if content_parts else span.name


def _extract_user_task_content(span: TraceSpan) -> str:
    """Extract user task content from span."""
    # Look for user message or task description
    content_attrs = [
        "message.content", "user.message", "task.description", "request.content"
    ]
    
    for attr in content_attrs:
        if attr in span.attributes:
            return str(span.attributes[attr])
    
    # Check events for user messages
    for event in span.events:
        if event.get("name") == "user_input" and event.get("attributes", {}).get("message"):
            return event["attributes"]["message"]
    
    return span.name


def _extract_agent_conversation_content(span: TraceSpan) -> str:
    """Extract agent conversation content from span."""
    content_attrs = [
        "agent.message", "conversation.content", "agent.response", "message.content"
    ]
    
    for attr in content_attrs:
        if attr in span.attributes:
            return str(span.attributes[attr])
    
    # Check events for agent messages
    for event in span.events:
        if event.get("name") == "agent_message" and event.get("attributes", {}).get("content"):
            return event["attributes"]["content"]
    
    return span.name


def _extract_llm_request_content(span: TraceSpan) -> str:
    """Extract LLM request content from span."""
    content_parts = []
    
    if span.attributes.get("gen_ai.request.model"):
        content_parts.append(f"Model: {span.attributes['gen_ai.request.model']}")
    
    if span.attributes.get("gen_ai.prompt"):
        content_parts.append(f"Prompt: {span.attributes['gen_ai.prompt']}")
    
    # Look for message content in various formats
    message_attrs = ["message.content", "gen_ai.request.messages", "prompt.content"]
    for attr in message_attrs:
        if attr in span.attributes:
            content_parts.append(str(span.attributes[attr]))
    
    return " | ".join(content_parts) if content_parts else span.name


def _extract_llm_response_content(span: TraceSpan) -> str:
    """Extract LLM response content from span."""
    content_parts = []
    
    if span.attributes.get("gen_ai.response.model"):
        content_parts.append(f"Model: {span.attributes['gen_ai.response.model']}")
    
    if span.attributes.get("gen_ai.completion"):
        content_parts.append(f"Response: {span.attributes['gen_ai.completion']}")
    
    if span.attributes.get("gen_ai.usage.output_tokens"):
        content_parts.append(f"Output tokens: {span.attributes['gen_ai.usage.output_tokens']}")
    
    # Look for response content
    response_attrs = ["response.content", "gen_ai.response.text", "completion.text"]
    for attr in response_attrs:
        if attr in span.attributes:
            content_parts.append(str(span.attributes[attr]))
    
    return " | ".join(content_parts) if content_parts else span.name


def _extract_tool_interaction_content(span: TraceSpan) -> str:
    """Extract tool interaction content from span."""
    content_parts = []
    
    tool_name = span.attributes.get("mcp.tool.name") or span.attributes.get("tool.name")
    if tool_name:
        content_parts.append(f"Tool: {tool_name}")
    
    # Extract arguments
    arguments = unflatten_attributes(span.attributes, "mcp.request.argument.")
    if arguments:
        content_parts.append(f"Arguments: {arguments}")
    
    # Extract result
    result = unflatten_attributes(span.attributes, "result.")
    if result:
        content_parts.append(f"Result: {result}")
    
    return " | ".join(content_parts) if content_parts else span.name


def _extract_system_event_content(span: TraceSpan) -> str:
    """Extract system event content from span."""
    content_attrs = [
        "event.message", "log.message", "system.message", "event.description"
    ]
    
    for attr in content_attrs:
        if attr in span.attributes:
            return str(span.attributes[attr])
    
    # Check events for system messages
    for event in span.events:
        if event.get("name") == "log" and event.get("attributes", {}).get("message"):
            return event["attributes"]["message"]
    
    return span.name


def _extract_error_warning_content(span: TraceSpan) -> str:
    """Extract error or warning content from span."""
    content_parts = []
    
    if span.attributes.get("error.message"):
        content_parts.append(f"Error: {span.attributes['error.message']}")
    
    if span.attributes.get("error.type"):
        content_parts.append(f"Type: {span.attributes['error.type']}")
    
    if span.attributes.get("warning"):
        content_parts.append(f"Warning: {span.attributes['warning']}")
    
    # Check events for error/warning messages
    for event in span.events:
        if event.get("name") in ["error", "warning"] and event.get("attributes", {}).get("message"):
            content_parts.append(event["attributes"]["message"])
    
    return " | ".join(content_parts) if content_parts else span.name


def _sort_messages_by_timestamp(trace_info: TraceInformation) -> None:
    """Sort all message lists by timestamp for chronological order."""
    trace_info.server_initialization.sort(key=lambda x: x.timestamp)
    trace_info.user_tasks.sort(key=lambda x: x.timestamp)
    trace_info.agent_conversations.sort(key=lambda x: x.timestamp)
    trace_info.llm_requests.sort(key=lambda x: x.timestamp)
    trace_info.llm_responses.sort(key=lambda x: x.timestamp)
    trace_info.tool_interactions.sort(key=lambda x: x.timestamp)
    trace_info.system_events.sort(key=lambda x: x.timestamp)
    trace_info.errors_warnings.sort(key=lambda x: x.timestamp)
