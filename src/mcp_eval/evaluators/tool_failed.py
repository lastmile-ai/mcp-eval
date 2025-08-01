"""ToolFailed evaluator for checking if tools failed."""

from dataclasses import dataclass

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators.tool_success_rate import ToolSuccessRate


@dataclass
class ToolFailed(ToolSuccessRate):
    """Evaluator that checks if a tool failed (0% success rate)."""
    
    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        result = super().evaluate_sync(ctx)
        failed = result.details["rate"] == 0.0  # Invert success rate

        return EvaluatorResult(
            passed=failed,
            expected="0% success rate",
            actual=f"{result.details['rate']:.1%}",
            details=result.details,
        )