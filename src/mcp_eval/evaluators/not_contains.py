"""NotContains evaluator for checking if response does NOT contain specific text."""

from dataclasses import dataclass

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators.response_contains import ResponseContains


@dataclass
class NotContains(ResponseContains):
    """Evaluator that checks if response does NOT contain specific text."""

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        result = super().evaluate_sync(ctx)
        inverted_passed = not result.passed

        return EvaluatorResult(
            passed=inverted_passed,
            expected=f"text NOT containing '{self.text}'",
            actual=result.actual,
            details=result.details,
        )
