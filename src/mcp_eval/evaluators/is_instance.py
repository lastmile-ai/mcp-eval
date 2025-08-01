"""IsInstance evaluator for checking output types."""

from typing import Any, Dict
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class IsInstance(SyncEvaluator):
    """Evaluator that checks if output is of expected type."""

    type_name: str

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        # Simplified type checking - would use proper type registry in production
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        expected_type = type_map.get(self.type_name, str)
        passed = isinstance(ctx.output, expected_type)

        return EvaluatorResult(
            passed=passed, expected=self.type_name, actual=type(ctx.output).__name__
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type_name": self.type_name}