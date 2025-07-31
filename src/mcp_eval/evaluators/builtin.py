"""Built-in evaluators for common evaluation patterns."""

import re
import json
import asyncio
from typing import Any, Dict, List, Literal, Optional, Union, Pattern, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from collections import defaultdict

from mcp_eval.evaluators.base import Evaluator, SyncEvaluator, EvaluatorContext


class EvaluatorResult(BaseModel):
    """Standardized result format for all evaluators."""

    passed: bool = Field(description="Whether the evaluation passed")
    expected: Union[str, int, float, List, Dict, None] = Field(
        default=None, description="What was expected"
    )
    actual: Union[str, int, float, List, Dict, None] = Field(
        default=None, description="What was actually received"
    )
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Score between 0.0 and 1.0"
    )
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


class CriterionResult(BaseModel):
    """Structured result from a single criterion evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Score for this criterion")
    explanation: str = Field(description="Detailed explanation of the score")
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Confidence in the judgment"
    )
    reasoning: str = Field(
        default="", description="Step-by-step reasoning if COT enabled"
    )


@dataclass
class EvaluationCriterion:
    """Single evaluation criterion."""

    name: str
    description: str
    weight: float = 1.0
    min_score: float = 0.7
    examples: Optional[Dict[str, str]] = field(default=None)  # score -> example

    def to_prompt(self) -> str:
        """Convert to prompt text."""
        prompt = f"{self.name}: {self.description}"
        if self.examples:
            prompt += "\nExamples:"
            for score, example in sorted(self.examples.items()):
                prompt += f"\n  Score {score}: {example}"
        return prompt


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
            details={"max_iterations": self.max_iterations},
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
            passed=passed, expected=self.type_name, actual=type(ctx.output).__name__
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
            details=result.details,
        )


@dataclass
class ToolOutputMatches(SyncEvaluator):
    """
    Evaluator that validates tool output against expected patterns.

    This evaluator allows you to validate the output of tool calls against expected
    values using various matching strategies. It supports nested field extraction,
    different comparison types, and flexible call targeting.

    Examples:
        ```
        # Exact match on full output
        ToolOutputMatches(tool_name="read_file", expected_output="Hello world")

        # Match substring in output
        ToolOutputMatches(tool_name="search", expected_output="found", match_type="contains")

        # Regex pattern matching
        ToolOutputMatches(tool_name="validate", expected_output=r"\\d+", match_type="regex")

        # Extract nested field and match
        ToolOutputMatches(
            tool_name="api_call",
            expected_output="success",
            field_path="result.status"
        )

        # Partial dictionary matching
        ToolOutputMatches(
            tool_name="get_config",
            expected_output={"debug": True},
            match_type="partial"
        )
        ```
    """

    tool_name: str
    """Name of the tool whose output should be validated."""

    expected_output: Union[Dict[str, Any], str, Pattern, int, float, List[Any]]
    """Expected output value or pattern to match against."""

    field_path: Optional[str] = None
    """Optional path to extract nested field from tool output.
    
    Supports dot notation for nested objects and bracket notation for arrays:
    - "content.text" - Extract text field from content object
    - "items[0].name" - Extract name from first item in items array
    - "result.data[2].value" - Complex nested extraction
    """

    match_type: Literal["exact", "contains", "regex", "partial"] = "exact"
    """Type of matching to perform:
    
    - "exact": Exact equality comparison (default)
    - "contains": Substring/item containment check
    - "regex": Regular expression pattern matching (string outputs only)
    - "partial": Partial matching for dicts/lists (all expected items must be present)
    """

    case_sensitive: bool = True
    """Whether string comparisons should be case sensitive.
    
    Applies to "contains" and "regex" match types when comparing strings.
    """

    call_index: int = -1
    """Which tool call to validate when multiple calls exist:
    
    - -1: Last call (most recent, default)
    - 0: First call
    - 1: Second call
    - etc.
    """

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        """Evaluate tool output against expected patterns."""
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]

        if not tool_calls:
            return EvaluatorResult(
                passed=False,
                expected=f"Tool '{self.tool_name}' to be called",
                actual="Tool not called",
                error="No matching tool calls found",
            )

        # Get the specified call
        try:
            target_call = tool_calls[self.call_index]
        except IndexError:
            return EvaluatorResult(
                passed=False,
                expected=f"At least {abs(self.call_index) + 1} calls to '{self.tool_name}'",
                actual=f"{len(tool_calls)} calls",
                error="Not enough tool calls",
            )

        # Extract the value to validate
        try:
            actual_value = self._extract_field_value(target_call.result)
        except Exception as e:
            return EvaluatorResult(
                passed=False,
                expected=f"Valid field path: {self.field_path}",
                actual=f"Error extracting field: {str(e)}",
                error=f"Field extraction failed: {str(e)}",
            )

        # Perform validation based on match type
        try:
            passed = self._validate_match(actual_value)
        except Exception as e:
            return EvaluatorResult(
                passed=False,
                expected=self.expected_output,
                actual=actual_value,
                error=f"Validation failed: {str(e)}",
            )

        return EvaluatorResult(
            passed=passed,
            expected=self.expected_output,
            actual=actual_value,
            details={
                "tool_name": self.tool_name,
                "call_index": self.call_index,
                "field_path": self.field_path,
                "match_type": self.match_type,
                "case_sensitive": self.case_sensitive,
            },
        )

    def _extract_field_value(self, result: Any) -> Any:
        """Extract value from result using field_path."""
        if self.field_path is None:
            return result

        current = result
        path_parts = self._parse_field_path(self.field_path)

        for part in path_parts:
            if isinstance(part, int):  # Array index
                if not isinstance(current, (list, tuple)):
                    raise ValueError(f"Cannot index non-list with [{part}]")
                if part >= len(current) or part < -len(current):
                    raise ValueError(
                        f"Index [{part}] out of range for list of length {len(current)}"
                    )
                current = current[part]
            else:  # Dictionary key
                if not isinstance(current, dict):
                    raise ValueError(f"Cannot access key '{part}' on non-dict")
                if part not in current:
                    raise ValueError(f"Key '{part}' not found in result")
                current = current[part]

        return current

    def _parse_field_path(self, path: str) -> List[Union[str, int]]:
        """Parse field path into components."""
        parts = []
        current = ""
        i = 0

        while i < len(path):
            char = path[i]
            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != "]":
                    j += 1
                if j >= len(path):
                    raise ValueError(f"Unclosed bracket in field path: {path}")
                index_str = path[i + 1 : j]
                try:
                    parts.append(int(index_str))
                except ValueError:
                    raise ValueError(f"Invalid array index: {index_str}")
                i = j  # Skip the closing bracket
            else:
                current += char
            i += 1

        if current:
            parts.append(current)

        return parts

    def _validate_match(self, actual_value: Any) -> bool:
        """Validate actual value against expected output based on match type."""
        if self.match_type == "exact":
            return actual_value == self.expected_output

        elif self.match_type == "contains":
            if isinstance(actual_value, str) and isinstance(self.expected_output, str):
                if self.case_sensitive:
                    return self.expected_output in actual_value
                else:
                    return self.expected_output.lower() in actual_value.lower()
            elif isinstance(actual_value, (list, tuple)):
                return self.expected_output in actual_value
            elif isinstance(actual_value, dict) and isinstance(
                self.expected_output, str
            ):
                # Search in dict values
                for value in actual_value.values():
                    if isinstance(value, str):
                        if self.case_sensitive:
                            if self.expected_output in value:
                                return True
                        else:
                            if self.expected_output.lower() in value.lower():
                                return True
                return False
            else:
                return False

        elif self.match_type == "regex":
            if not isinstance(actual_value, str):
                return False

            if isinstance(self.expected_output, Pattern):
                pattern = self.expected_output
            elif isinstance(self.expected_output, str):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                pattern = re.compile(self.expected_output, flags)
            else:
                return False

            return bool(pattern.search(actual_value))

        elif self.match_type == "partial":
            if isinstance(self.expected_output, dict) and isinstance(
                actual_value, dict
            ):
                # Check if all expected keys and values are present
                for key, expected_val in self.expected_output.items():
                    if key not in actual_value:
                        return False
                    if isinstance(expected_val, dict) and isinstance(
                        actual_value[key], dict
                    ):
                        # Recursive partial matching for nested dicts
                        nested_validator = ToolOutputMatches(
                            tool_name=self.tool_name,
                            expected_output=expected_val,
                            match_type="partial",
                            case_sensitive=self.case_sensitive,
                        )
                        if not nested_validator._validate_match(actual_value[key]):
                            return False
                    else:
                        if actual_value[key] != expected_val:
                            return False
                return True
            elif isinstance(self.expected_output, list) and isinstance(
                actual_value, list
            ):
                # Check if all expected items are present
                for expected_item in self.expected_output:
                    if expected_item not in actual_value:
                        return False
                return True
            else:
                return False

        else:
            raise ValueError(f"Unknown match_type: {self.match_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evaluator to dict."""
        result = {
            "tool_name": self.tool_name,
            "expected_output": self.expected_output,
            "field_path": self.field_path,
            "match_type": self.match_type,
            "case_sensitive": self.case_sensitive,
            "call_index": self.call_index,
        }

        # Handle Pattern objects
        if isinstance(self.expected_output, Pattern):
            result["expected_output"] = self.expected_output.pattern

        return result


@dataclass
class PathEfficiency(SyncEvaluator):
    """Evaluates if agent took the optimal path to complete task."""

    optimal_steps: Optional[int] = None
    """Expected optimal number of steps (auto-calculated if None)."""

    expected_tool_sequence: Optional[List[str]] = None
    """Expected sequence of tool calls."""

    allow_extra_steps: int = 0
    """Tolerance for additional steps beyond optimal."""

    penalize_backtracking: bool = True
    """Whether to penalize returning to previous tools."""

    penalize_repeated_tools: bool = True
    """Whether to penalize excessive tool repetition."""

    tool_usage_limits: Optional[Dict[str, int]] = None
    """Custom limits per tool (e.g., {"read": 2, "write": 1})."""

    default_tool_limit: int = 1
    """Default limit for tools not in tool_usage_limits."""

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        actual_steps = ctx.metrics.iteration_count
        tool_sequence = [call.name for call in ctx.tool_calls]

        # Auto-calculate optimal if not provided
        if self.optimal_steps is None:
            # Try to get from context or use heuristic
            if hasattr(ctx, "baseline_steps"):
                self.optimal_steps = ctx.baseline_steps
            else:
                # Heuristic: unique tools used
                self.optimal_steps = len(set(tool_sequence))

        # Calculate efficiency score
        efficiency_score = self.optimal_steps / actual_steps if actual_steps > 0 else 0

        # Check for inefficiencies
        inefficiencies = []

        # Check sequence
        sequence_correct = True
        if self.expected_tool_sequence:
            sequence_correct, seq_issues = self._check_sequence(
                tool_sequence, self.expected_tool_sequence
            )
            inefficiencies.extend(seq_issues)

        # Check for backtracking
        if self.penalize_backtracking:
            backtrack_count = self._count_backtracking(tool_sequence)
            if backtrack_count > 0:
                inefficiencies.append(f"Backtracking detected: {backtrack_count} times")
                efficiency_score *= (
                    1 - 0.1 * backtrack_count
                )  # 10% penalty per backtrack

        # Check for repeated tools
        if self.penalize_repeated_tools:
            repetitions = self._count_repetitions(tool_sequence)
            if repetitions:
                inefficiencies.append(f"Repeated tools: {repetitions}")
                efficiency_score *= 0.9  # 10% penalty for repetitions

        # Check individual failure conditions
        steps_exceeded = actual_steps > self.optimal_steps + self.allow_extra_steps
        has_inefficiencies = len(inefficiencies) > 0

        # Overall pass/fail
        passed = not steps_exceeded and sequence_correct and not has_inefficiencies

        # Build comprehensive expected description
        if not sequence_correct:
            expected = f"sequence: {self.expected_tool_sequence}"
            actual = f"sequence: {tool_sequence}"
        elif has_inefficiencies:
            actual = f"inefficiencies: {inefficiencies}"
            expected = "No inefficiencies"
        elif steps_exceeded:
            expected = f"≤{self.optimal_steps + self.allow_extra_steps} steps"
            actual = f"{actual_steps} steps"

        return EvaluatorResult(
            passed=passed,
            expected=expected,
            actual=actual,
            score=efficiency_score,
            details={
                "tool_sequence": tool_sequence,
                "efficiency_score": efficiency_score,
                "sequence_correct": sequence_correct,
                "steps_exceeded": steps_exceeded,
                "has_inefficiencies": has_inefficiencies,
                "inefficiencies": inefficiencies,
                "optimal_path": self.expected_tool_sequence,
                "actual_steps": actual_steps,
                "max_allowed_steps": self.optimal_steps + self.allow_extra_steps,
            },
        )

    def _check_sequence(
        self, actual: List[str], expected: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if actual sequence matches expected."""
        issues = []

        # Check if expected tools appear in order
        expected_idx = 0
        for tool in actual:
            if expected_idx < len(expected) and tool == expected[expected_idx]:
                expected_idx += 1

        if expected_idx != len(expected):
            missing = expected[expected_idx:]
            issues.append(f"Missing expected tools: {missing}")
            return False, issues

        # Check for unexpected tools between expected ones
        actual_filtered = [t for t in actual if t in expected]
        if actual_filtered != expected:
            issues.append("Expected tools not in correct order")
            return False, issues

        return True, issues

    def _count_backtracking(self, sequence: List[str]) -> int:
        """Count times agent went back to previous tools."""
        if len(sequence) < 2:
            return 0

        backtrack_count = 0
        seen_tools = set()
        tool_last_index = {}

        for i, tool in enumerate(sequence):
            if tool in seen_tools:
                # Check if we're going backwards
                if i - tool_last_index[tool] > 2:  # Allow immediate retry
                    backtrack_count += 1
            seen_tools.add(tool)
            tool_last_index[tool] = i

        return backtrack_count

    def _count_repetitions(self, sequence: List[str]) -> Dict[str, int]:
        """Count unnecessary repetitions of tools."""
        tool_counts = defaultdict(int)
        repetitions = {}

        for tool in sequence:
            tool_counts[tool] += 1

        # Use configurable limits
        tool_limits = self.tool_usage_limits or {}

        for tool, count in tool_counts.items():
            expected = tool_limits.get(tool, self.default_tool_limit)
            if count > expected:
                repetitions[tool] = count - expected

        return repetitions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_steps": self.optimal_steps,
            "expected_tool_sequence": self.expected_tool_sequence,
            "allow_extra_steps": self.allow_extra_steps,
            "penalize_backtracking": self.penalize_backtracking,
            "penalize_repeated_tools": self.penalize_repeated_tools,
            "tool_usage_limits": self.tool_usage_limits,
            "default_tool_limit": self.default_tool_limit,
        }


@dataclass
class MultiCriteriaJudge(Evaluator):
    """Enhanced LLM judge with multiple criteria and detailed rubrics."""

    criteria: List[EvaluationCriterion]
    require_all_pass: bool = False
    use_cot: bool = True  # Chain of thought reasoning
    model: Optional[str] = None
    include_confidence: bool = True
    aggregate_method: str = "weighted"  # "weighted", "min", "harmonic_mean"

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorResult:
        # Evaluate each criterion in parallel
        criterion_results = await asyncio.gather(
            *[self._evaluate_criterion(ctx, criterion) for criterion in self.criteria]
        )

        # Extract scores and explanations
        scores = {}
        explanations = {}
        confidences = {}

        for criterion, result in zip(self.criteria, criterion_results):
            scores[criterion.name] = result.score
            explanations[criterion.name] = result.explanation
            if self.include_confidence:
                confidences[criterion.name] = result.confidence

        # Calculate overall score
        overall_score = self._aggregate_scores(scores, confidences)

        # Check pass conditions
        if self.require_all_pass:
            passed = all(scores[c.name] >= c.min_score for c in self.criteria)
        else:
            # Weighted pass threshold
            pass_threshold = sum(c.weight * c.min_score for c in self.criteria) / sum(
                c.weight for c in self.criteria
            )
            passed = overall_score >= pass_threshold

        # Generate summary explanation
        summary = self._generate_summary(scores, explanations)

        return EvaluatorResult(
            passed=passed,
            score=overall_score,
            expected="Meets all criteria"
            if self.require_all_pass
            else f"Score ≥ {pass_threshold:.2f}",
            actual=f"Overall score: {overall_score:.2f}",
            details={
                "criteria_scores": scores,
                "explanations": explanations,
                "confidences": confidences,
                "summary": summary,
                "failed_criteria": [
                    c.name for c in self.criteria if scores[c.name] < c.min_score
                ],
            },
        )

    async def _evaluate_criterion(
        self, ctx: EvaluatorContext, criterion: EvaluationCriterion
    ) -> CriterionResult:
        """Evaluate a single criterion."""
        from mcp_eval.llm_client import get_judge_client

        # Build evaluation prompt
        prompt = self._build_criterion_prompt(ctx, criterion)

        # Get LLM evaluation using structured generation
        client = get_judge_client(self.model)
        result = await client.generate_structured(
            prompt, response_model=CriterionResult
        )

        return result

    def _build_criterion_prompt(
        self, ctx: EvaluatorContext, criterion: EvaluationCriterion
    ) -> str:
        """Build prompt for evaluating a single criterion."""
        parts = [
            "Evaluate the following response based on this criterion:",
            "",
            criterion.to_prompt(),
            "",
            "Input:",
            "---",
            str(ctx.inputs),
            "---",
            "",
            "Response to evaluate:",
            "---",
            str(ctx.output),
            "---",
        ]

        if self.use_cot:
            parts.extend(
                [
                    "",
                    "First, think through your evaluation step by step.",
                    "Then provide your final assessment.",
                ]
            )

        parts.extend(
            [
                "",
                "Provide a score between 0.0 and 1.0 for how well the response meets this criterion.",
                "Include a detailed explanation of your score and your confidence level (0.0-1.0).",
            ]
        )

        if self.use_cot:
            parts.append("Also include your step-by-step reasoning.")

        return "\n".join(parts)

    def _aggregate_scores(
        self, scores: Dict[str, float], confidences: Dict[str, float]
    ) -> float:
        """Aggregate multiple scores into overall score."""
        if self.aggregate_method == "weighted":
            # Weighted average with confidence
            total_weight = sum(c.weight for c in self.criteria)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(
                scores[c.name] * c.weight * confidences.get(c.name, 1.0)
                for c in self.criteria
            )
            confidence_sum = sum(
                c.weight * confidences.get(c.name, 1.0) for c in self.criteria
            )
            return weighted_sum / confidence_sum if confidence_sum > 0 else 0.0

        elif self.aggregate_method == "min":
            # Minimum score (most conservative)
            return min(scores.values())

        elif self.aggregate_method == "harmonic_mean":
            # Harmonic mean (penalizes low scores)
            values = [scores[c.name] for c in self.criteria]
            if any(v == 0 for v in values):
                return 0.0
            return len(values) / sum(1 / v for v in values)

        else:
            # Simple average
            return sum(scores.values()) / len(scores)

    def _generate_summary(
        self, scores: Dict[str, float], explanations: Dict[str, str]
    ) -> str:
        """Generate summary of evaluation."""
        summary_parts = []

        # Overall assessment
        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 0.9:
            summary_parts.append("Excellent performance across all criteria.")
        elif avg_score >= 0.7:
            summary_parts.append("Good performance with some areas for improvement.")
        elif avg_score >= 0.5:
            summary_parts.append(
                "Adequate performance but significant improvements needed."
            )
        else:
            summary_parts.append("Poor performance requiring major improvements.")

        # Highlight strengths
        strengths = [c.name for c in self.criteria if scores[c.name] >= 0.8]
        if strengths:
            summary_parts.append(f"Strengths: {', '.join(strengths)}")

        # Highlight weaknesses
        weaknesses = [c.name for c in self.criteria if scores[c.name] < 0.6]
        if weaknesses:
            summary_parts.append(f"Areas for improvement: {', '.join(weaknesses)}")

        return " ".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criteria": [
                {
                    "name": c.name,
                    "description": c.description,
                    "weight": c.weight,
                    "min_score": c.min_score,
                    "examples": c.examples,
                }
                for c in self.criteria
            ],
            "require_all_pass": self.require_all_pass,
            "use_cot": self.use_cot,
            "model": self.model,
            "include_confidence": self.include_confidence,
            "aggregate_method": self.aggregate_method,
        }


# Predefined criteria sets
STANDARD_CRITERIA = [
    EvaluationCriterion(
        name="Accuracy",
        description="Response is factually correct and addresses the question",
        weight=2.0,
        min_score=0.8,
    ),
    EvaluationCriterion(
        name="Completeness",
        description="Response covers all aspects of the question",
        weight=1.5,
        min_score=0.7,
    ),
    EvaluationCriterion(
        name="Clarity",
        description="Response is clear, well-organized, and easy to understand",
        weight=1.0,
        min_score=0.7,
    ),
    EvaluationCriterion(
        name="Efficiency",
        description="Response is concise without unnecessary information",
        weight=0.5,
        min_score=0.6,
    ),
]

CODE_GENERATION_CRITERIA = [
    EvaluationCriterion(
        name="Correctness",
        description="Code is syntactically correct and would execute without errors",
        weight=3.0,
        min_score=0.9,
        examples={
            "1.0": "Code runs perfectly with no errors",
            "0.5": "Code has minor syntax errors that are easily fixable",
            "0.0": "Code has major errors or wouldn't run",
        },
    ),
    EvaluationCriterion(
        name="Functionality",
        description="Code correctly implements the requested functionality",
        weight=3.0,
        min_score=0.8,
    ),
    EvaluationCriterion(
        name="Style",
        description="Code follows good practices and conventions",
        weight=1.0,
        min_score=0.6,
    ),
    EvaluationCriterion(
        name="Documentation",
        description="Code includes appropriate comments and documentation",
        weight=0.5,
        min_score=0.5,
    ),
]

SQL_QUERY_CRITERIA = [
    EvaluationCriterion(
        name="Syntax",
        description="Query has correct SQL syntax",
        weight=2.0,
        min_score=1.0,  # Must be perfect
    ),
    EvaluationCriterion(
        name="Logic",
        description="Query logic correctly addresses the requirement",
        weight=3.0,
        min_score=0.8,
    ),
    EvaluationCriterion(
        name="Efficiency",
        description="Query is optimized and avoids unnecessary operations",
        weight=1.0,
        min_score=0.6,
    ),
    EvaluationCriterion(
        name="Readability",
        description="Query is well-formatted and easy to understand",
        weight=0.5,
        min_score=0.5,
    ),
]


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
    "ToolOutputMatches": ToolOutputMatches,
    "PathEfficiency": PathEfficiency,
    "MultiCriteriaJudge": MultiCriteriaJudge,
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
