"""ToolSequence evaluator for checking tool call sequences."""

from typing import Any, Dict, List
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class ToolSequence(SyncEvaluator):
    """Evaluator that checks if tools were called in a specific sequence."""

    expected_sequence: List[str]
    allow_other_calls: bool = True

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        actual_sequence = [call.name for call in ctx.tool_calls]

        if not self.allow_other_calls:
            matches = actual_sequence == self.expected_sequence
        else:
            # Check if expected sequence appears as subsequence
            matches = self._is_subsequence(self.expected_sequence, actual_sequence)

        return EvaluatorResult(
            passed=matches, expected=self.expected_sequence, actual=actual_sequence
        )

    def _is_subsequence(self, subseq: List[str], seq: List[str]) -> bool:
        """Check if subseq is a subsequence of seq."""
        it = iter(seq)
        return all(item in it for item in subseq)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_sequence": self.expected_sequence,
            "allow_other_calls": self.allow_other_calls,
        }