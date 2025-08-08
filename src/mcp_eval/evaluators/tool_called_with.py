"""ToolCalledWith evaluator for checking tool call arguments."""

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators.tool_was_called import ToolWasCalled


class ToolCalledWith(ToolWasCalled):
    """Evaluator that checks if a tool was called with specific arguments."""
    
    def __init__(self, tool_name: str, expected_args: dict):
        super().__init__(tool_name)
        self.expected_args = expected_args

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]
        matching_calls = [
            call
            for call in tool_calls
            if all(call.arguments.get(k) == v for k, v in self.expected_args.items())
        ]
        matches = bool(matching_calls)

        return EvaluatorResult(
            passed=matches,
            expected=f"tool '{self.tool_name}' called with {self.expected_args}",
            actual=f"{len(tool_calls)} calls found",
            details={
                "tool_name": self.tool_name,
                "expected_args": self.expected_args,
                "matching_calls": len(matching_calls),
            },
        )