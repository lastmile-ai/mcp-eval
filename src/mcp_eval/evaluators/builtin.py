"""Built-in evaluators for common evaluation patterns."""

import re
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from mcp_eval.evaluators.base import Evaluator, SyncEvaluator, EvaluatorContext


class EvaluatorResult(BaseModel):
    """Standardized result format for all evaluators."""

    passed: bool = Field(description="Whether the evaluation passed")
    expected: Union[str, int, float, List, Dict] = Field(
        description="What was expected"
    )
    actual: Union[str, int, float, List, Dict, None] = Field(
        description="What was actually received"
    )
    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context-specific information"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if applicable"
    )

    class Config:
        extra = "forbid"


class EvaluationRecord(BaseModel):
    """Record of an evaluation result."""

    name: str = Field(description="Name of the evaluator")
    result: EvaluatorResult = Field(description="The evaluation result")
    passed: bool = Field(description="Whether the evaluation passed")
    error: Optional[str] = Field(
        default=None, description="Error message if applicable"
    )

    class Config:
        extra = "forbid"


class JudgeResult(BaseModel):
    """Structured result from LLM judge evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation of the score")
    passed: bool = Field(description="Whether the response passes the rubric")
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Confidence in the judgment"
    )


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
            score=1.0 if passed else 0.0,
            details={
                "tool_name": self.tool_name,
                "min_times": self.min_times,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "min_times": self.min_times}


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
            passed=matches,
            expected=self.expected_sequence,
            actual=actual_sequence,
            score=1.0 if matches else 0.0,
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
                score=0.0,
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
            score=1.0 if matches else 0.0,
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
            score=1.0
            if passed
            else max(0.0, 1.0 - (actual - self.max_iterations) / self.max_iterations),
            details={
                "max_allowed": self.max_iterations,
                "over_by": actual - self.max_iterations if not passed else 0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"max_iterations": self.max_iterations}


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


@dataclass
class LLMJudge(Evaluator):
    """Evaluator that uses an LLM to judge response quality."""

    rubric: str
    min_score: float = 0.8
    model: Optional[str] = None
    include_input: bool = False
    include_expected: bool = True
    require_reasoning: bool = True

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorResult:
        # Build prompt for LLM judge with structured output request
        prompt_parts = [
            f"Evaluate the following response based on this rubric: {self.rubric}",
            "",
            "Response to evaluate:",
            "---",
            f"{ctx.output}",
            "---",
        ]

        if self.include_input:
            prompt_parts.extend(
                [
                    "",
                    "Original input:",
                    f"{ctx.inputs}",
                ]
            )

        if self.include_expected and ctx.expected_output is not None:
            prompt_parts.extend(
                [
                    "",
                    "Expected output:",
                    f"{ctx.expected_output}",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Provide your evaluation as a JSON object with the following structure:",
                "{",
                '  "score": <float between 0.0 and 1.0>,',
                '  "reasoning": "<detailed explanation of your score>",',
                '  "passed": <boolean indicating if the response meets the rubric>,',
                '  "confidence": <float between 0.0 and 1.0 indicating your confidence>'
                "}",
                "",
                "Ensure your JSON is valid and complete.",
            ]
        )

        prompt = "\n".join(prompt_parts)

        try:
            from mcp_eval.llm_client import get_judge_client

            client = get_judge_client(self.model)
            response = await client.generate_str(prompt)

            # Extract and parse JSON response
            json_str = self._extract_json(response)
            judge_data = json.loads(json_str)

            # Validate with Pydantic
            judge_result = JudgeResult(**judge_data)

            # Use the structured result
            passed = judge_result.passed and judge_result.score >= self.min_score

            return EvaluatorResult(
                passed=passed,
                expected=f"score >= {self.min_score}",
                actual=f"score = {judge_result.score}",
                score=judge_result.score,
                details={
                    "reasoning": judge_result.reasoning,
                    "confidence": judge_result.confidence,
                    "min_score": self.min_score,
                    "rubric": self.rubric,
                    "judge_response": response,
                },
            )

        except Exception as e:
            # Fallback to simple parsing if structured output fails
            try:
                score = self._extract_numeric_score(response)
                passed = score >= self.min_score

                return EvaluatorResult(
                    passed=passed,
                    expected=f"score >= {self.min_score}",
                    actual=f"score = {score}",
                    score=score,
                    details={
                        "reasoning": "Fallback parsing used",
                        "confidence": 0.5,
                        "min_score": self.min_score,
                        "rubric": self.rubric,
                        "judge_response": response,
                        "parsing_error": str(e),
                    },
                )
            except Exception as fallback_error:
                return EvaluatorResult(
                    passed=False,
                    expected=f"score >= {self.min_score}",
                    actual="failed to parse",
                    score=0.0,
                    error=str(fallback_error),
                    details={
                        "reasoning": "Failed to parse judge response",
                        "confidence": 0.0,
                        "rubric": self.rubric,
                        "judge_response": response,
                    },
                )

    def _extract_json(self, response: str) -> str:
        """Extract JSON from response, handling various formats."""
        # Try to find JSON block
        import re

        # Look for JSON between ``` markers
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Look for JSON object directly
        json_match = re.search(
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", response, re.DOTALL
        )
        if json_match:
            return json_match.group(1)

        # If no JSON found, try the whole response
        return response.strip()

    def _extract_numeric_score(self, response: str) -> float:
        """Fallback method to extract numeric score."""
        import re

        # Look for decimal numbers between 0 and 1
        scores = re.findall(r"\b(0?\.\d+|1\.0|0\.0|1)\b", response)
        if scores:
            score = float(scores[0])
            if 0.0 <= score <= 1.0:
                return score

        # Look for percentages
        percentages = re.findall(r"(\d+(?:\.\d+)?)%", response)
        if percentages:
            return float(percentages[0]) / 100.0

        raise ValueError("Could not extract numeric score from response")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric": self.rubric,
            "min_score": self.min_score,
            "model": self.model,
            "include_input": self.include_input,
            "include_expected": self.include_expected,
            "require_reasoning": self.require_reasoning,
        }


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
            passed=passed,
            expected=self.type_name,
            actual=type(ctx.output).__name__,
            score=1.0 if passed else 0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type_name": self.type_name}


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
                score=1.0,
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
            score=1.0 if matches else 0.0,
            details={
                "exact_match": self.exact_match,
                "case_sensitive": self.case_sensitive,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"exact_match": self.exact_match, "case_sensitive": self.case_sensitive}


class ResponseTimeCheck(MaxIterations):
    """Evaluator that checks if the response time is under the threshold"""

    def __init__(self, max_ms: float):
        self.max_ms = max_ms

    def evaluate_sync(self, ctx):
        passed = ctx.metrics.latency_ms <= self.max_ms

        return EvaluatorResult(
            passed=passed,
            expected=f"latency <= {self.max_ms}",
            actual=ctx.metrics.latency_ms,
            score=1.0 if passed else 0.0,
            details={"max_ms": self.max_ms},
        )


class ExactToolCount(ToolWasCalled):
    def __init__(self, tool_name: str, expected_count: int):
        super().__init__(tool_name)
        self.expected_count = expected_count

    def evaluate_sync(self, ctx):
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]
        passed = len(tool_calls) == self.expected_count

        return EvaluatorResult(
            passed=passed,
            expected=len(tool_calls),
            actual=self.expected_count,
            score=1.0 if passed else 0.0,
            details={"expected_count": self.expected_count},
        )


@dataclass
class ToolFailed(ToolSuccessRate):
    def evaluate_sync(self, ctx):
        result = super().evaluate_sync(ctx)
        failed = result.details["rate"] == 0.0  # Invert success rate

        return EvaluatorResult(
            passed=failed,
            expected="0% success rate",
            actual=f"{result.details['rate']:.1%}",
            score=1.0 if failed else 0.0,
            details=result.details,
        )


@dataclass
class ToolCalledWith(ToolWasCalled):
    def __init__(self, tool_name: str, expected_args: dict):
        super().__init__(tool_name)
        self.expected_args = expected_args

    def evaluate_sync(self, ctx):
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
            score=1.0 if matches else 0.0,
            details={
                "tool_name": self.tool_name,
                "expected_args": self.expected_args,
                "matching_calls": len(matching_calls),
            },
        )


@dataclass
class NotContains(ResponseContains):
    def evaluate_sync(self, ctx):
        result = super().evaluate_sync(ctx)
        inverted_passed = not result.passed

        return EvaluatorResult(
            passed=inverted_passed,
            expected=f"text NOT containing '{self.text}'",
            actual=result.actual,
            score=1.0 if inverted_passed else 0.0,
            details=result.details,
        )


# Registry for dynamic loading
_EVALUATOR_REGISTRY = {
    "ToolWasCalled": ToolWasCalled,
    "ToolSequence": ToolSequence,
    "ResponseContains": ResponseContains,
    "MaxIterations": MaxIterations,
    "ToolSuccessRate": ToolSuccessRate,
    "LLMJudge": LLMJudge,
    "IsInstance": IsInstance,
    "EqualsExpected": EqualsExpected,
    "ResponseTimeCheck": ResponseTimeCheck,
    "ExactToolCount": ExactToolCount,
    "ToolFailed": ToolFailed,
    "ToolCalledWith": ToolCalledWith,
    "NotContains": NotContains,
}


def get_evaluator_by_name(name: str, config: Dict[str, Any]) -> Optional[Evaluator]:
    """Get evaluator instance by name and configuration."""
    evaluator_class = _EVALUATOR_REGISTRY.get(name)
    if evaluator_class:
        return evaluator_class.from_dict(config)
    return None


def register_evaluator(name: str, evaluator_class: type):
    """Register a custom evaluator."""
    _EVALUATOR_REGISTRY[name] = evaluator_class
