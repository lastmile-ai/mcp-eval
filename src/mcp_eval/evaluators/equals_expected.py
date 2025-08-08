"""EqualsExpected evaluator for checking if output equals expected output."""

from typing import Any, Dict
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class EqualsExpected(SyncEvaluator):
    """Evaluator that checks if output equals expected output."""

    exact_match: bool = True
    case_sensitive: bool = True

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        if ctx.expected_output is None:
            return EvaluatorResult(
                passed=True,
                expected="no expected output",
                actual=ctx.output,
                details={"reason": "no_expected_output"},
            )

        if self.exact_match:
            if isinstance(ctx.output, str) and isinstance(ctx.expected_output, str):
                if not self.case_sensitive:
                    matches = ctx.output.lower() == ctx.expected_output.lower()
                else:
                    matches = ctx.output == ctx.expected_output
            else:
                matches = ctx.output == ctx.expected_output
        else:
            # Fuzzy matching for strings
            if isinstance(ctx.output, str) and isinstance(ctx.expected_output, str):
                output = ctx.output.lower() if not self.case_sensitive else ctx.output
                expected = (
                    ctx.expected_output.lower()
                    if not self.case_sensitive
                    else ctx.expected_output
                )
                matches = expected in output
            else:
                matches = ctx.output == ctx.expected_output

        return EvaluatorResult(
            passed=matches,
            expected=ctx.expected_output,
            actual=ctx.output,
            details={
                "exact_match": self.exact_match,
                "case_sensitive": self.case_sensitive,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"exact_match": self.exact_match, "case_sensitive": self.case_sensitive}