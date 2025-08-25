"""MultiCriteriaJudge evaluator with multiple criteria and detailed rubrics."""

import asyncio
from typing import Any, Dict, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


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
    examples: Dict[str, str] | None = field(default=None)  # score -> example

    def to_prompt(self) -> str:
        """Convert to prompt text."""
        prompt = f"{self.name}: {self.description}"
        if self.examples:
            prompt += "\\nExamples:"
            for score, example in sorted(self.examples.items()):
                prompt += f"\\n  Score {score}: {example}"
        return prompt


@dataclass
class MultiCriteriaJudge(Evaluator):
    """Enhanced LLM judge with multiple criteria and detailed rubrics."""

    criteria: List[EvaluationCriterion]
    require_all_pass: bool = False
    use_cot: bool = True  # Chain of thought reasoning
    model: str | None = None
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

        # Build actual field with explanations for failed criteria
        actual_parts = [f"Overall score: {overall_score:.2f}"]
        failed_criteria_names = [
            c.name for c in self.criteria if scores[c.name] < c.min_score
        ]
        if failed_criteria_names:
            actual_parts.append("Failed criteria:")
            for criteria_name in failed_criteria_names:
                score = scores[criteria_name]
                explanation = explanations[criteria_name]
                actual_parts.append(f"  • {criteria_name}: {score:.2f} - {explanation}")

        return EvaluatorResult(
            passed=passed,
            score=overall_score,
            expected="Meets all criteria"
            if self.require_all_pass
            else f"Score ≥ {pass_threshold:.2f}",
            actual="\n".join(actual_parts),
            details={
                "criteria_scores": scores,
                "explanations": explanations,
                "confidences": confidences,
                "summary": summary,
                "failed_criteria": failed_criteria_names,
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

        return "\\n".join(parts)

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
