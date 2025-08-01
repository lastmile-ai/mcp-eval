"""ResponseTimeCheck evaluator for checking response time thresholds."""

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators.max_iterations import MaxIterations


class ResponseTimeCheck(MaxIterations):
    """Evaluator that checks if the response time is under the threshold"""

    def __init__(self, max_ms: float):
        self.max_ms = max_ms

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        passed = ctx.metrics.latency_ms <= self.max_ms

        return EvaluatorResult(
            passed=passed,
            expected=f"latency <= {self.max_ms}",
            actual=ctx.metrics.latency_ms,
            details={"max_ms": self.max_ms},
        )