"""ExactToolCount evaluator for checking exact tool call counts."""

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators.tool_was_called import ToolWasCalled


class ExactToolCount(ToolWasCalled):
    """Evaluator that checks for an exact number of tool calls."""

    def __init__(self, tool_name: str, expected_count: int):
        super().__init__(tool_name)
        self.expected_count = expected_count

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]
        passed = len(tool_calls) == self.expected_count

        return EvaluatorResult(
            passed=passed,
            expected=self.expected_count,
            actual=len(tool_calls),
            details={"expected_count": self.expected_count},
        )
