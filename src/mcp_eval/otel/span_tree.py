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
    children: List["SpanNode"] = None

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
            self.attributes.get("status.code") == "ERROR"
            or self.attributes.get("result.isError", False)
            or any(event.get("name") == "exception" for event in self.events)
        )


SpanQuery = Union[
    Callable[[SpanNode], bool],
    Dict[str, Any],  # Query dict with keys like 'name_contains', 'has_attribute', etc.
]


@dataclass
class LLMRephrasingLoop:
    """Detected LLM rephrasing loop."""

    spans: List[SpanNode]
    loop_count: int
    total_duration: timedelta
    is_stuck: bool


@dataclass
class ToolPathAnalysis:
    """Analysis of tool call path efficiency."""

    actual_path: List[str]
    golden_path: List[str]
    efficiency_score: float
    unnecessary_calls: List[str]
    missing_calls: List[str]


@dataclass
class ErrorRecoverySequence:
    """Detected error recovery attempt."""

    error_span: SpanNode
    recovery_spans: List[SpanNode]
    recovery_successful: bool
    recovery_duration: timedelta


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
        return self.find(
            lambda span: "tool" in span.name.lower() or "call_tool" in span.name
        )

    def get_llm_calls(self) -> List[SpanNode]:
        """Get all LLM call spans."""
        return self.find(lambda span: span.attributes.get("gen_ai.system") is not None)

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics of the execution."""
        tool_calls = self.get_tool_calls()
        llm_calls = self.get_llm_calls()

        return {
            "total_duration": self.root.duration.total_seconds(),
            "tool_call_count": len(tool_calls),
            "llm_call_count": len(llm_calls),
            "tool_call_duration": sum(
                span.duration.total_seconds() for span in tool_calls
            ),
            "llm_call_duration": sum(
                span.duration.total_seconds() for span in llm_calls
            ),
            "parallel_tool_calls": self._count_parallel_calls(tool_calls),
            "error_count": len([span for span in self.all_spans() if span.has_error()]),
        }

    def get_llm_rephrasing_loops(
        self, max_similar_calls: int = 3, similarity_threshold: float = 0.8
    ) -> List[LLMRephrasingLoop]:
        """Detect if an agent is stuck in a loop, repeatedly rephrasing its request to an LLM."""
        llm_calls = self.get_llm_calls()
        loops = []

        # Group consecutive similar LLM calls
        i = 0
        while i < len(llm_calls):
            similar_group = [llm_calls[i]]
            base_prompt = self._extract_prompt(llm_calls[i])

            # Find consecutive similar calls
            j = i + 1
            while j < len(llm_calls):
                current_prompt = self._extract_prompt(llm_calls[j])
                if self._prompts_similar(
                    base_prompt, current_prompt, similarity_threshold
                ):
                    similar_group.append(llm_calls[j])
                    j += 1
                else:
                    break

            # If we found a loop (too many similar calls)
            if len(similar_group) >= max_similar_calls:
                total_duration = (
                    similar_group[-1].end_time - similar_group[0].start_time
                )

                loops.append(
                    LLMRephrasingLoop(
                        spans=similar_group,
                        loop_count=len(similar_group),
                        total_duration=total_duration,
                        is_stuck=len(similar_group)
                        > max_similar_calls * 1.5,  # Very stuck if much longer
                    )
                )

            i = j if j > i + 1 else i + 1

        return loops

    def get_inefficient_tool_paths(
        self, golden_paths: Dict[str, List[str]]
    ) -> List[ToolPathAnalysis]:
        """Compare tool call sequence to predefined golden paths to measure efficiency."""
        tool_calls = self.get_tool_calls()
        actual_path = [span.name for span in tool_calls]

        analyses = []
        for task_name, golden_path in golden_paths.items():
            if self._path_matches_task(actual_path, task_name):
                efficiency_score = self._calculate_path_efficiency(
                    actual_path, golden_path
                )
                unnecessary_calls = self._find_unnecessary_calls(
                    actual_path, golden_path
                )
                missing_calls = self._find_missing_calls(actual_path, golden_path)

                analyses.append(
                    ToolPathAnalysis(
                        actual_path=actual_path,
                        golden_path=golden_path,
                        efficiency_score=efficiency_score,
                        unnecessary_calls=unnecessary_calls,
                        missing_calls=missing_calls,
                    )
                )

        return analyses

    def get_error_recovery_sequences(self) -> List[ErrorRecoverySequence]:
        """Identify tool call failures and check if subsequent steps represent successful recovery."""
        error_spans = [span for span in self.all_spans() if span.has_error()]
        recovery_sequences = []

        for error_span in error_spans:
            # Find spans that occur after this error
            recovery_spans = []
            for span in self.all_spans():
                if (
                    span.start_time > error_span.end_time
                    and span.start_time < error_span.end_time + timedelta(minutes=5)
                ):  # Within 5 minutes
                    recovery_spans.append(span)

            if recovery_spans:
                # Check if recovery was successful (no more errors in recovery spans)
                recovery_successful = not any(
                    span.has_error() for span in recovery_spans
                )
                recovery_duration = (
                    recovery_spans[-1].end_time - recovery_spans[0].start_time
                )

                recovery_sequences.append(
                    ErrorRecoverySequence(
                        error_span=error_span,
                        recovery_spans=recovery_spans,
                        recovery_successful=recovery_successful,
                        recovery_duration=recovery_duration,
                    )
                )

        return recovery_sequences

    def _extract_prompt(self, llm_span: SpanNode) -> str:
        """Extract prompt text from LLM span."""
        # Look for prompt in various attribute locations
        prompt = (
            llm_span.attributes.get("gen_ai.prompt", "")
            or llm_span.attributes.get("llm.prompt", "")
            or llm_span.attributes.get("input", "")
            or llm_span.name
        )
        return str(prompt).lower().strip()

    def _prompts_similar(self, prompt1: str, prompt2: str, threshold: float) -> bool:
        """Check if two prompts are similar using simple string similarity."""
        if not prompt1 or not prompt2:
            return False

        # Simple Jaccard similarity
        words1 = set(prompt1.split())
        words2 = set(prompt2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0

        return similarity >= threshold

    def _path_matches_task(self, actual_path: List[str], task_name: str) -> bool:
        """Check if the actual path seems to match a particular task."""
        # Simple heuristic - check if task name appears in any tool names
        task_words = task_name.lower().split()
        path_text = " ".join(actual_path).lower()
        return any(word in path_text for word in task_words)

    def _calculate_path_efficiency(
        self, actual_path: List[str], golden_path: List[str]
    ) -> float:
        """Calculate efficiency score comparing actual vs golden path."""
        if not golden_path:
            return 1.0

        # Simple efficiency: golden_length / actual_length, capped at 1.0
        efficiency = len(golden_path) / len(actual_path) if actual_path else 0.0
        return min(1.0, efficiency)

    def _find_unnecessary_calls(
        self, actual_path: List[str], golden_path: List[str]
    ) -> List[str]:
        """Find tool calls in actual path that aren't in golden path."""
        golden_set = set(golden_path)
        return [tool for tool in actual_path if tool not in golden_set]

    def _find_missing_calls(
        self, actual_path: List[str], golden_path: List[str]
    ) -> List[str]:
        """Find tool calls in golden path that are missing from actual path."""
        actual_set = set(actual_path)
        return [tool for tool in golden_path if tool not in actual_set]

    def _matches_query(self, span: SpanNode, query: SpanQuery) -> bool:
        """Check if a span matches the given query."""
        if callable(query):
            return query(span)

        # Dict-based query
        if isinstance(query, dict):
            for key, value in query.items():
                if key == "name_contains":
                    if value not in span.name:
                        return False
                elif key == "has_attribute":
                    if value not in span.attributes:
                        return False
                elif key == "attribute_equals":
                    attr_name, expected_value = value
                    if span.attributes.get(attr_name) != expected_value:
                        return False
                elif key == "duration_gt":
                    if span.duration.total_seconds() <= value:
                        return False
                elif key == "has_error":
                    if span.has_error() != value:
                        return False

        return True

    def _count_parallel_calls(self, spans: List[SpanNode]) -> int:
        """Count how many spans were executing in parallel."""
        if len(spans) <= 1:
            return 0

        events = []
        for span in spans:
            events.append(("start", span.start_time))
            events.append(("end", span.end_time))

        events.sort(key=lambda x: x[1])

        max_concurrent = 0
        current_concurrent = 0

        for event_type, _ in events:
            if event_type == "start":
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            else:
                current_concurrent -= 1

        return max_concurrent - 1  # Subtract 1 because we want parallel calls
