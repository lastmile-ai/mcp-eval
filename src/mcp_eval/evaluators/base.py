"""Base evaluator classes and context."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, Union
from dataclasses import dataclass

from mcp_eval.metrics import TestMetrics
from mcp_eval.otel.span_tree import SpanTree


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class EvaluatorContext(Generic[InputType, OutputType]):
    """Context provided to evaluators during evaluation."""

    inputs: InputType
    output: OutputType
    expected_output: Optional[OutputType]
    metadata: Optional[Dict[str, Any]]
    metrics: TestMetrics
    span_tree: Optional[SpanTree] = None

    @property
    def tool_calls(self):
        """Convenience property to access tool calls from metrics."""
        return self.metrics.tool_calls

    @property
    def llm_metrics(self):
        """Convenience property to access LLM metrics."""
        return self.metrics.llm_metrics


class Evaluator(ABC, Generic[InputType, OutputType]):
    """Base class for all evaluators."""

    @abstractmethod
    async def evaluate(
        self, ctx: EvaluatorContext[InputType, OutputType]
    ) -> Union[float, bool, Dict[str, Any]]:
        """
        Evaluate the test case.

        Returns:
            - float: Score between 0.0 and 1.0
            - bool: Pass/fail result
            - dict: Detailed evaluation results
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluator to dictionary for serialization."""
        return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evaluator":
        """Create evaluator from dictionary."""
        return cls(**data)


class SyncEvaluator(Evaluator[InputType, OutputType]):
    """Base class for synchronous evaluators."""

    async def evaluate(
        self, ctx: EvaluatorContext[InputType, OutputType]
    ) -> Union[float, bool, Dict[str, Any]]:
        """Async wrapper for sync evaluation."""
        return self.evaluate_sync(ctx)

    @abstractmethod
    def evaluate_sync(
        self, ctx: EvaluatorContext[InputType, OutputType]
    ) -> Union[float, bool, Dict[str, Any]]:
        """Synchronous evaluation method."""
        pass
