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


@dataclass
class TraceInformation:
    """Comprehensive information extracted from a single trace span."""
    
    # Associated message information
    message_info: MessageInfo
    
    # Span attributes and metadata
    span_attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Original span for reference
    original_span: Optional[TraceSpan] = None


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


def extract_comprehensive_trace_information(spans: List[TraceSpan]) -> List[TraceInformation]:
    """Extract comprehensive information from trace spans that contain actual text/message content.
    
    Args:
        spans: List of TraceSpan objects extracted from trace jsonl file
        
    Returns:
        List of TraceInformation objects, one for each span that contains meaningful text/message content
    """
    trace_info_list = []
    
    for span in spans:
        # Check if this span contains any meaningful text/message content
        if not _span_contains_text_message(span):
            continue
            
        timestamp = span.start_time / 1e9  # Convert nanoseconds to seconds
        end_timestamp = span.end_time / 1e9
        
        # Extract sender/recipient information from span attributes
        sender = _extract_sender_info(span)
        recipient = _extract_recipient_info(span)
        
        # Determine message type and extract content
        message_type = _determine_message_type(span)
        content = _extract_text_content(span)
        
        # Only create TraceInformation if we found meaningful content
        if content and content.strip():
            # Create MessageInfo for this span
            message_info = MessageInfo(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                end_timestamp=end_timestamp,
                message_type=message_type,
                metadata={"span_name": span.name, "attributes": span.attributes}
            )
            
            # Create TraceInformation for this span
            trace_info = TraceInformation(
                message_info=message_info,
                span_attributes=span.attributes.copy(),
                original_span=span
            )
            
            trace_info_list.append(trace_info)
    
    # Sort by timestamp for chronological order
    trace_info_list.sort(key=lambda x: x.message_info.timestamp)
    
    return trace_info_list


def _span_contains_text_message(span: TraceSpan) -> bool:
    """Check if a span contains meaningful text/message content."""
    # Check common attributes that might contain text content
    text_attributes = [
        # User messages and tasks
        "message.content", "user.message", "task.description", "request.content",
        # LLM prompts and responses
        "gen_ai.prompt", "gen_ai.completion", "gen_ai.response.text", 
        "gen_ai.prompt.1.content", "gen_ai.prompt.2.content", "gen_ai.prompt.3.content",
        "gen_ai.response.1.content", "gen_ai.response.2.content",
        # Agent conversations
        "agent.message", "conversation.content", "agent.response",
        # System messages and logs
        "log.message", "system.message", "event.message", "event.description",
        # Error messages
        "error.message", "warning", "error.description",
        # Server initialization messages
        "server.message", "init.message", "startup.message",
        # Tool interaction content that might have text
        "tool.description", "tool.result.content"
    ]
    
    # Check span attributes for text content
    for attr in text_attributes:
        if attr in span.attributes and span.attributes[attr]:
            value = str(span.attributes[attr]).strip()
            if value and len(value) > 0:
                return True
    
    # Check events for text content
    for event in span.events:
        if event.get("name") in ["log", "message", "user_input", "agent_message"]:
            event_attrs = event.get("attributes", {})
            for key, value in event_attrs.items():
                if "message" in key.lower() or "content" in key.lower():
                    if value and str(value).strip():
                        return True
    
    # Check for any attribute that might contain meaningful text (broader check)
    for key, value in span.attributes.items():
        if isinstance(value, str) and len(value.strip()) > 10:  # Reasonable text length
            # Skip technical IDs and short codes
            if not (key.endswith("_id") or key.endswith(".id") or key.endswith("_key") or 
                   key.startswith("trace.") or key.startswith("span.") or
                   value.replace("-", "").replace("_", "").isalnum()):
                return True
    
    return False


def _extract_text_content(span: TraceSpan) -> str:
    """Extract all meaningful text content from a span."""
    content_parts = []
    
    # Priority order for extracting content
    priority_attributes = [
        # User content first
        "message.content", "user.message", "task.description", "request.content",
        # LLM content
        "gen_ai.prompt.1.content", "gen_ai.prompt.2.content", "gen_ai.prompt.3.content",
        "gen_ai.response.1.content", "gen_ai.response.2.content",
        "gen_ai.prompt", "gen_ai.completion", "gen_ai.response.text",
        # Agent content
        "agent.message", "conversation.content", "agent.response",
        # System content
        "log.message", "system.message", "event.message", "event.description",
        # Error content
        "error.message", "warning", "error.description",
        # Server content
        "server.message", "init.message", "startup.message"
    ]
    
    # Extract content in priority order
    for attr in priority_attributes:
        if attr in span.attributes and span.attributes[attr]:
            value = str(span.attributes[attr]).strip()
            if value:
                content_parts.append(value)
    
    # Check events for additional content
    for event in span.events:
        if event.get("name") in ["log", "message", "user_input", "agent_message"]:
            event_attrs = event.get("attributes", {})
            for key, value in event_attrs.items():
                if ("message" in key.lower() or "content" in key.lower()) and value:
                    content_parts.append(str(value).strip())
    
    # If no priority content found, look for any meaningful text
    if not content_parts:
        for key, value in span.attributes.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                # Skip technical IDs and short codes
                if not (key.endswith("_id") or key.endswith(".id") or key.endswith("_key") or 
                       key.startswith("trace.") or key.startswith("span.") or
                       value.replace("-", "").replace("_", "").isalnum()):
                    content_parts.append(value.strip())
    
    # Return combined content, removing duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in content_parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    
    return " | ".join(unique_parts) if unique_parts else ""


def _determine_message_type(span: TraceSpan) -> str:
    """Determine the message type for a span."""
    if _is_server_init_span(span):
        return "server_init"
    elif _is_user_task_span(span):
        return "user_task"
    elif _is_agent_conversation_span(span):
        return "agent_conversation"
    elif _is_llm_request_span(span):
        return "llm_request"
    elif _is_llm_response_span(span):
        return "llm_response"
    elif _is_tool_call_span(span):
        return "tool_interaction"
    elif _is_system_event_span(span):
        return "system_event"
    elif _has_error_or_warning(span):
        return "error_warning"
    else:
        return "general"


def _extract_content_by_type(span: TraceSpan, message_type: str) -> str:
    """Extract content based on message type."""
    if message_type == "server_init":
        return _extract_server_init_content(span)
    elif message_type == "user_task":
        return _extract_user_task_content(span)
    elif message_type == "agent_conversation":
        return _extract_agent_conversation_content(span)
    elif message_type == "llm_request":
        return _extract_llm_request_content(span)
    elif message_type == "llm_response":
        return _extract_llm_response_content(span)
    elif message_type == "tool_interaction":
        return _extract_tool_interaction_content(span)
    elif message_type == "system_event":
        return _extract_system_event_content(span)
    elif message_type == "error_warning":
        return _extract_error_warning_content(span)
    else:
        return span.name


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


