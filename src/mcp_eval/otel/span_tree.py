"""OpenTelemetry span tree analysis utilities."""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SpanNode:
    """Represents a single span in the execution tree."""
    
    span_id: str
    name: str
    start_time: datetime
    end_time: datetime
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    parent_id: Optional[str] = None
    children: List['SpanNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def duration(self) -> timedelta:
        """Duration of this span."""
        return self.end_time - self.start_time
    
    def has_error(self) -> bool:
        """Check if this span indicates an error."""
        return (
            self.attributes.get('status.code') == 'ERROR' or
            self.attributes.get('result.isError', False) or
            any(event.get('name') == 'exception' for event in self.events)
        )


SpanQuery = Union[
    Callable[[SpanNode], bool],
    Dict[str, Any]  # Query dict with keys like 'name_contains', 'has_attribute', etc.
]


class SpanTree:
    """Tree structure for analyzing OpenTelemetry spans."""
    
    def __init__(self, root: SpanNode):
        self.root = root
        self._all_spans = {}
        self._build_index()
    
    def _build_index(self):
        """Build an index of all spans for fast lookup."""
        def _index_span(span: SpanNode):
            self._all_spans[span.span_id] = span
            for child in span.children:
                _index_span(child)
        
        _index_span(self.root)
    
    def find(self, query: SpanQuery) -> List[SpanNode]:
        """Find all spans matching the query."""
        matches = []
        
        def _check_span(span: SpanNode):
            if self._matches_query(span, query):
                matches.append(span)
            for child in span.children:
                _check_span(child)
        
        _check_span(self.root)
        return matches
    
    def find_first(self, query: SpanQuery) -> Optional[SpanNode]:
        """Find the first span matching the query."""
        matches = self.find(query)
        return matches[0] if matches else None
    
    def any(self, query: SpanQuery) -> bool:
        """Check if any span matches the query."""
        return self.find_first(query) is not None
    
    def all_spans(self) -> List[SpanNode]:
        """Get all spans in the tree."""
        return list(self._all_spans.values())
    
    def get_tool_calls(self) -> List[SpanNode]:
        """Get all tool call spans."""
        return self.find(lambda span: 'tool' in span.name.lower() or 'call_tool' in span.name)
    
    def get_llm_calls(self) -> List[SpanNode]:
        """Get all LLM call spans."""
        return self.find(lambda span: span.attributes.get('gen_ai.system') is not None)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics of the execution."""
        tool_calls = self.get_tool_calls()
        llm_calls = self.get_llm_calls()
        
        return {
            'total_duration': self.root.duration.total_seconds(),
            'tool_call_count': len(tool_calls),
            'llm_call_count': len(llm_calls),
            'tool_call_duration': sum(span.duration.total_seconds() for span in tool_calls),
            'llm_call_duration': sum(span.duration.total_seconds() for span in llm_calls),
            'parallel_tool_calls': self._count_parallel_calls(tool_calls),
            'error_count': len([span for span in self.all_spans() if span.has_error()]),
        }
    
    def _matches_query(self, span: SpanNode, query: SpanQuery) -> bool:
        """Check if a span matches the given query."""
        if callable(query):
            return query(span)
        
        # Dict-based query
        if isinstance(query, dict):
            for key, value in query.items():
                if key == 'name_contains':
                    if value not in span.name:
                        return False
                elif key == 'has_attribute':
                    if value not in span.attributes:
                        return False
                elif key == 'attribute_equals':
                    attr_name, expected_value = value
                    if span.attributes.get(attr_name) != expected_value:
                        return False
                elif key == 'duration_gt':
                    if span.duration.total_seconds() <= value:
                        return False
                elif key == 'has_error':
                    if span.has_error() != value:
                        return False
        
        return True
    
    def _count_parallel_calls(self, spans: List[SpanNode]) -> int:
        """Count how many spans were executing in parallel."""
        # Simplified implementation - count overlapping time windows
        if len(spans) <= 1:
            return 0
        
        events = []
        for span in spans:
            events.append(('start', span.start_time))
            events.append(('end', span.end_time))
        
        events.sort(key=lambda x: x[1])
        
        max_concurrent = 0
        current_concurrent = 0
        
        for event_type, _ in events:
            if event_type == 'start':
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            else:
                current_concurrent -= 1
        
        return max_concurrent - 1  # Subtract 1 because we want parallel calls
