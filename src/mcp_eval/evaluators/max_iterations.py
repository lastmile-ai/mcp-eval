"""MaxIterations evaluator for checking iteration limits."""

from typing import Any, Dict
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class MaxIterations(SyncEvaluator):
    """Evaluator that checks if task completed within max iterations."""

    max_iterations: int

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        actual = ctx.metrics.iteration_count
        passed = actual <= self.max_iterations

        return EvaluatorResult(
            passed=passed,
            expected=f"<= {self.max_iterations}",
            actual=actual,
            details={"max_iterations": self.max_iterations},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"max_iterations": self.max_iterations}