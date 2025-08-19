"""ResponseContains evaluator for checking if response contains specific text."""

import re
from typing import Any, Dict
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class ResponseContains(SyncEvaluator):
    """Evaluator that checks if response contains specific text."""

    text: str
    case_sensitive: bool = False
    regex: bool = False

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        if not isinstance(ctx.output, str):
            return EvaluatorResult(
                passed=False,
                expected=f"string containing '{self.text}'",
                actual=f"{type(ctx.output).__name__}: {ctx.output}",
                error="Output is not a string",
            )

        response = ctx.output
        if not self.case_sensitive:
            response = response.lower()
            text = self.text.lower()
        else:
            text = self.text

        if self.regex:
            matches = bool(re.search(text, response))
        else:
            matches = text in response

        match_type = "regex match" if self.regex else "substring"
        case_note = (
            " (case-sensitive)" if self.case_sensitive else " (case-insensitive)"
        )

        return EvaluatorResult(
            passed=matches,
            expected=f"{match_type} '{self.text}'{case_note}",
            actual=ctx.output,
            details={
                "text": self.text,
                "regex": self.regex,
                "case_sensitive": self.case_sensitive,
                "match_type": match_type,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "case_sensitive": self.case_sensitive,
            "regex": self.regex,
        }
