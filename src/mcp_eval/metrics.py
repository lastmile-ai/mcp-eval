import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SimpleSpan:
    name: str
    context: dict
    parent: Optional[dict]
    start_time: int
    end_time: int
    attributes: dict = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)

    @staticmethod
    def from_json(json_str: str) -> 'SimpleSpan':
        data = json.loads(json_str)
        return SimpleSpan(
            name=data.get('name'),
            context=data.get('context'),
            parent=data.get('parent'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            attributes=data.get('attributes', {}),
            events=data.get('events', [])
        )

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Any
    is_error: bool
    latency_ms: float

@dataclass
class LLMMetrics:
    model_name: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost: float = 0.0
    call_count: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

@dataclass
class TestMetrics:
    success: bool = True
    latency_ms: float = 0.0
    turns: int = 0
    llm_metrics: Dict[str, LLMMetrics] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def tool_metrics(self) -> Dict[str, Any]:
        """Aggregates metrics for each tool."""
        metrics = {}
        for tc in self.tool_calls:
            if tc.name not in metrics:
                metrics[tc.name] = {"call_count": 0, "success_count": 0, "error_count": 0}
            metrics[tc.name]["call_count"] += 1
            if tc.is_error:
                metrics[tc.name]["error_count"] += 1
            else:
                metrics[tc.name]["success_count"] += 1
        return metrics


def process_spans(spans: List[SimpleSpan]) -> TestMetrics:
    """Processes a list of SimpleSpan objects to compute test metrics."""
    metrics = TestMetrics()
    if not spans:
        return metrics

    root_span = next((s for s in spans if s.parent is None), spans[0])
    metrics.latency_ms = (root_span.end_time - root_span.start_time) / 1e6

    for span in spans:
        latency_ms = (span.end_time - span.start_time) / 1e6
        
        if span.attributes.get('gen_ai.system'):
            model = span.attributes.get('gen_ai.request.model', 'unknown')
            if model not in metrics.llm_metrics:
                metrics.llm_metrics[model] = LLMMetrics(model_name=model)
            
            llm_metric = metrics.llm_metrics[model]
            llm_metric.call_count += 1
            llm_metric.total_latency_ms += latency_ms
            llm_metric.total_input_tokens += span.attributes.get('gen_ai.usage.input_tokens', 0)
            llm_metric.total_output_tokens += span.attributes.get('gen_ai.usage.output_tokens', 0)
            metrics.turns += 1

        if span.name.endswith(".call_tool"):
            tool_name = span.attributes.get("gen_ai.tool.name", "unknown")
            is_error = span.attributes.get("result.isError", False)
            
            arguments = {}
            prefix = "request.params.arguments."
            for key, value in span.attributes.items():
                if key.startswith(prefix):
                    arg_name = key[len(prefix):]
                    arguments[arg_name] = value

            result_content = []
            for key, value in span.attributes.items():
                if key.startswith("result.content.") and key.endswith(".text"):
                    result_content.append(value)
            
            metrics.tool_calls.append(ToolCall(
                name=tool_name,
                arguments=arguments,
                result=" ".join(result_content),
                is_error=is_error,
                latency_ms=latency_ms
            ))

    return metrics