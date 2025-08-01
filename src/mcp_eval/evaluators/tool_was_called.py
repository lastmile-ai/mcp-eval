"""ToolWasCalled evaluator for checking if a specific tool was called."""

from typing import Any, Dict
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class ToolWasCalled(SyncEvaluator):
    """Evaluator that checks if a specific tool was called."""

    tool_name: str
    min_times: int = 1

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]
        actual_calls = len(tool_calls)
        passed = actual_calls >= self.min_times

        return EvaluatorResult(
            passed=passed,
            expected=f">= {self.min_times}",
            actual=actual_calls,
            details={
                "tool_name": self.tool_name,
                "min_times": self.min_times,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "min_times": self.min_times}