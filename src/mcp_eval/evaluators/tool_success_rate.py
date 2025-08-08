"""ToolSuccessRate evaluator for checking tool success rates."""

from typing import Any, Dict, Optional
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class ToolSuccessRate(SyncEvaluator):
    """Evaluator that checks tool success rate."""

    min_rate: float = 0.9
    tool_name: Optional[str] = None  # If None, checks all tools

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        if self.tool_name:
            tool_calls = [
                call for call in ctx.tool_calls if call.name == self.tool_name
            ]
        else:
            tool_calls = ctx.tool_calls

        if not tool_calls:
            return EvaluatorResult(
                passed=False,
                expected=f">= {self.min_rate:.1%}",
                actual="N/A (no tool calls)",
                score=0.0,
                details={
                    "rate": 0,
                    "min_required": self.min_rate,
                    "total_calls": 0,
                    "successful_calls": 0,
                    "tool_name": self.tool_name,
                },
            )

        successful_calls = [call for call in tool_calls if not call.is_error]
        success_rate = len(successful_calls) / len(tool_calls)
        passed = success_rate >= self.min_rate

        return EvaluatorResult(
            passed=passed,
            expected=f">= {self.min_rate:.1%}",
            actual=f"{success_rate:.1%}",
            score=success_rate,
            details={
                "rate": success_rate,
                "min_required": self.min_rate,
                "total_calls": len(tool_calls),
                "successful_calls": len(successful_calls),
                "tool_name": self.tool_name,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"min_rate": self.min_rate, "tool_name": self.tool_name}