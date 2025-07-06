"""Built-in evaluators for common evaluation patterns."""

import re
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from mcp_eval.evaluators.base import Evaluator, SyncEvaluator, EvaluatorContext


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

    def evaluate_sync(self, ctx: EvaluatorContext) -> bool:
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]
        return len(tool_calls) >= self.min_times

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "min_times": self.min_times}


@dataclass
class ToolSequence(SyncEvaluator):
    """Evaluator that checks if tools were called in a specific sequence."""

    expected_sequence: List[str]
    allow_other_calls: bool = True

    def evaluate_sync(self, ctx: EvaluatorContext) -> Dict[str, Any]:
        actual_sequence = [call.name for call in ctx.tool_calls]

        if not self.allow_other_calls:
            matches = actual_sequence == self.expected_sequence
        else:
            # Check if expected sequence appears as subsequence
            matches = self._is_subsequence(self.expected_sequence, actual_sequence)

        return {
            "matches": matches,
            "expected": self.expected_sequence,
            "actual": actual_sequence,
            "score": 1.0 if matches else 0.0,
        }

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

    def evaluate_sync(self, ctx: EvaluatorContext) -> bool:
        if not isinstance(ctx.output, str):
            return False

        response = ctx.output
        if not self.case_sensitive:
            response = response.lower()
            text = self.text.lower()
        else:
            text = self.text

        if self.regex:
            return bool(re.search(text, response))
        else:
            return text in response

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

    def evaluate_sync(self, ctx: EvaluatorContext) -> Dict[str, Any]:
        actual = ctx.metrics.iteration_count
        passed = actual <= self.max_iterations

        return {
            "passed": passed,
            "max_allowed": self.max_iterations,
            "actual": actual,
            "score": 1.0
            if passed
            else max(0.0, 1.0 - (actual - self.max_iterations) / self.max_iterations),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"max_iterations": self.max_iterations}


@dataclass
class ToolSuccessRate(SyncEvaluator):
    """Evaluator that checks tool success rate."""

    min_rate: float = 0.9
    tool_name: Optional[str] = None  # If None, checks all tools

    def evaluate_sync(self, ctx: EvaluatorContext) -> Dict[str, Any]:
        if self.tool_name:
            tool_calls = [
                call for call in ctx.tool_calls if call.name == self.tool_name
            ]
        else:
            tool_calls = ctx.tool_calls

        if not tool_calls:
            return {
                "passed": True,
                "rate": 1.0,
                "score": 1.0,
            }  # No calls = perfect success

        successful_calls = [call for call in tool_calls if not call.is_error]
        success_rate = len(successful_calls) / len(tool_calls)
        passed = success_rate >= self.min_rate

        return {
            "passed": passed,
            "rate": success_rate,
            "min_required": self.min_rate,
            "total_calls": len(tool_calls),
            "successful_calls": len(successful_calls),
            "score": success_rate,
        }

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

    async def evaluate(self, ctx: EvaluatorContext) -> Dict[str, Any]:
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

            return {
                "passed": passed,
                "score": judge_result.score,
                "reasoning": judge_result.reasoning,
                "confidence": judge_result.confidence,
                "min_score": self.min_score,
                "rubric": self.rubric,
                "judge_response": response,
            }

        except Exception as e:
            # Fallback to simple parsing if structured output fails
            try:
                score = self._extract_numeric_score(response)
                passed = score >= self.min_score

                return {
                    "passed": passed,
                    "score": score,
                    "reasoning": "Fallback parsing used",
                    "confidence": 0.5,
                    "min_score": self.min_score,
                    "rubric": self.rubric,
                    "judge_response": response,
                    "parsing_error": str(e),
                }
            except Exception as fallback_error:
                return {
                    "passed": False,
                    "score": 0.0,
                    "reasoning": "Failed to parse judge response",
                    "confidence": 0.0,
                    "error": str(fallback_error),
                    "rubric": self.rubric,
                    "judge_response": response,
                }

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

    def evaluate_sync(self, ctx: EvaluatorContext) -> bool:
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
        return isinstance(ctx.output, expected_type)

    def to_dict(self) -> Dict[str, Any]:
        return {"type_name": self.type_name}


@dataclass
class EqualsExpected(SyncEvaluator):
    """Evaluator that checks if output equals expected output."""

    exact_match: bool = True
    case_sensitive: bool = True

    def evaluate_sync(self, ctx: EvaluatorContext) -> Dict[str, Any]:
        if ctx.expected_output is None:
            return {"passed": True, "score": 1.0, "reason": "no_expected_output"}

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

        return {
            "passed": matches,
            "score": 1.0 if matches else 0.0,
            "exact_match": self.exact_match,
            "case_sensitive": self.case_sensitive,
            "output": ctx.output,
            "expected": ctx.expected_output,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"exact_match": self.exact_match, "case_sensitive": self.case_sensitive}


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
