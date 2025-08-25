"""Discovery-friendly catalog of checks/evaluators.

This module provides a single, IntelliSense-friendly surface area to discover
what MCP-Eval supports. Use with the unified assertion API:

    from mcp_eval import Expect
    await session.assert_that(Expect.tools.was_called("fetch"))
    await session.assert_that(Expect.content.contains("Example Domain"), response=response)

The functions below return evaluator instances; they do not execute assertions
themselves. Pair them with `session.assert_that(...)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Pattern

from mcp_eval.evaluators import (
    ToolWasCalled,
    ToolSequence,
    ResponseContains,
    NotContains,
    MaxIterations,
    ToolSuccessRate,
    LLMJudge,
    ResponseTimeCheck,
    ExactToolCount,
    ToolFailed,
    ToolCalledWith,
    ToolOutputMatches,
    PathEfficiency,
    MultiCriteriaJudge,
    EvaluationCriterion,
)


class Content:
    """Content-focused checks."""

    @staticmethod
    def contains(text: str, *, case_sensitive: bool = False) -> ResponseContains:
        return ResponseContains(text=text, case_sensitive=case_sensitive)

    @staticmethod
    def not_contains(text: str, *, case_sensitive: bool = False) -> NotContains:
        return NotContains(text=text, case_sensitive=case_sensitive)

    @staticmethod
    def regex(pattern: str, *, case_sensitive: bool = False) -> ResponseContains:
        return ResponseContains(text=pattern, case_sensitive=case_sensitive, regex=True)


class Tools:
    """Tool usage and behavior checks."""

    @staticmethod
    def was_called(tool_name: str, *, min_times: int = 1) -> ToolWasCalled:
        return ToolWasCalled(tool_name=tool_name, min_times=min_times)

    @staticmethod
    def called_with(tool_name: str, arguments: Dict[str, Any]) -> ToolCalledWith:
        return ToolCalledWith(tool_name, arguments)

    @staticmethod
    def count(tool_name: str, expected_count: int) -> ExactToolCount:
        return ExactToolCount(tool_name, expected_count)

    @staticmethod
    def success_rate(min_rate: float, tool_name: str | None = None) -> ToolSuccessRate:
        return ToolSuccessRate(min_rate=min_rate, tool_name=tool_name)

    @staticmethod
    def failed(tool_name: str) -> ToolFailed:
        return ToolFailed(min_rate=0.0, tool_name=tool_name)

    @staticmethod
    def output_matches(
        tool_name: str,
        expected_output: Dict[str, Any] | str | Pattern | int | float | List[Any],
        *,
        field_path: str | None = None,
        match_type: str = "exact",
        case_sensitive: bool = True,
        call_index: int = -1,
    ) -> ToolOutputMatches:
        return ToolOutputMatches(
            tool_name=tool_name,
            expected_output=expected_output,
            field_path=field_path,
            match_type=match_type,  # type: ignore[arg-type]
            case_sensitive=case_sensitive,
            call_index=call_index,
        )

    @staticmethod
    def sequence(
        sequence: List[str], *, allow_other_calls: bool = False
    ) -> ToolSequence:
        return ToolSequence(sequence, allow_other_calls=allow_other_calls)


class Performance:
    """Latency, iteration count, and related checks."""

    @staticmethod
    def max_iterations(max_iterations: int) -> MaxIterations:
        return MaxIterations(max_iterations=max_iterations)

    @staticmethod
    def response_time_under(ms: float) -> ResponseTimeCheck:
        return ResponseTimeCheck(ms)


class Judge:
    """LLM judge options."""

    @staticmethod
    def llm(
        rubric: str,
        *,
        min_score: float = 0.8,
        include_input: bool = False,
        require_reasoning: bool = True,
    ) -> LLMJudge:
        return LLMJudge(
            rubric=rubric,
            min_score=min_score,
            include_input=include_input,
            require_reasoning=require_reasoning,
        )

    @staticmethod
    def multi_criteria(
        criteria: Dict[str, str] | List[EvaluationCriterion],
        *,
        aggregate_method: str = "weighted",
        require_all_pass: bool = False,
        include_confidence: bool = True,
        use_cot: bool = True,
        model: str | None = None,
    ) -> MultiCriteriaJudge:
        # If dict is provided, convert to EvaluationCriterion list with default weights/min_scores
        if isinstance(criteria, dict):
            criteria = [
                EvaluationCriterion(name=name, description=desc)
                for name, desc in criteria.items()
            ]
        return MultiCriteriaJudge(
            criteria=criteria,  # type: ignore[arg-type]
            aggregate_method=aggregate_method,
            require_all_pass=require_all_pass,
            include_confidence=include_confidence,
            use_cot=use_cot,
            model=model,
        )


class Path:
    """Path efficiency and related checks."""

    @staticmethod
    def efficiency(
        *,
        optimal_steps: int | None = None,
        expected_tool_sequence: List[str] | None = None,
        allow_extra_steps: int = 0,
        penalize_backtracking: bool = True,
        penalize_repeated_tools: bool = True,
        tool_usage_limits: Dict[str, int] | None = None,
        default_tool_limit: int = 1,
    ) -> PathEfficiency:
        return PathEfficiency(
            optimal_steps=optimal_steps,
            expected_tool_sequence=expected_tool_sequence,
            allow_extra_steps=allow_extra_steps,
            penalize_backtracking=penalize_backtracking,
            penalize_repeated_tools=penalize_repeated_tools,
            tool_usage_limits=tool_usage_limits,
            default_tool_limit=default_tool_limit,
        )


class Expect:
    """Top-level discovery namespace.

    - Expect.content.*    → content assertions
    - Expect.tools.*      → tool usage/output assertions
    - Expect.performance.*→ latency/iterations
    - Expect.judge.*      → LLM judges
    - Expect.path.*       → path efficiency/sequence
    """

    content = Content
    tools = Tools
    performance = Performance
    judge = Judge
    path = Path


__all__ = [
    "Expect",
    "Content",
    "Tools",
    "Performance",
    "Judge",
    "Path",
]
