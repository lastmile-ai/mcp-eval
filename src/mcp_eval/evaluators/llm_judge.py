"""LLMJudge evaluator for using LLM to judge response quality."""

import json
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.llm_client import get_judge_client


class JudgeResult(BaseModel):
    """Structured result from LLM judge evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation of the score")
    passed: bool = Field(description="Whether the response passes the rubric")
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Confidence in the judgment"
    )


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
            client = get_judge_client(self.model)
            
            # Ensure LLM is initialized to get actual config
            await client._get_llm()
            judge_config = client.get_config()
            
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
                actual=f"score = {judge_result.score}. {judge_result.reasoning}",
                score=judge_result.score,
                details={
                    "reasoning": judge_result.reasoning,
                    "confidence": judge_result.confidence,
                    "min_score": self.min_score,
                    "rubric": self.rubric,
                    "judge_response": response,
                    "judge_config": judge_config,
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
                    actual=f"score = {score}. {response}",
                    score=score,
                    details={
                        "reasoning": "Fallback parsing used",
                        "confidence": 0.5,
                        "min_score": self.min_score,
                        "rubric": self.rubric,
                        "judge_response": response,
                        "parsing_error": str(e),
                        "judge_config": judge_config,
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
                        "judge_config": judge_config if 'judge_config' in locals() else None,
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
