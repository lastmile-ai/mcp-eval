"""Evaluators package - imports all evaluators and shared components."""

from typing import Any, Dict

# Import shared components
from mcp_eval.evaluators.shared import EvaluatorResult, EvaluationRecord

# Import all evaluators from separate files
from mcp_eval.evaluators.tool_was_called import ToolWasCalled
from mcp_eval.evaluators.tool_sequence import ToolSequence
from mcp_eval.evaluators.response_contains import ResponseContains
from mcp_eval.evaluators.max_iterations import MaxIterations
from mcp_eval.evaluators.tool_success_rate import ToolSuccessRate
from mcp_eval.evaluators.llm_judge import LLMJudge, JudgeResult
from mcp_eval.evaluators.is_instance import IsInstance
from mcp_eval.evaluators.equals_expected import EqualsExpected
from mcp_eval.evaluators.response_time_check import ResponseTimeCheck
from mcp_eval.evaluators.exact_tool_count import ExactToolCount
from mcp_eval.evaluators.tool_failed import ToolFailed
from mcp_eval.evaluators.tool_called_with import ToolCalledWith
from mcp_eval.evaluators.not_contains import NotContains
from mcp_eval.evaluators.tool_output_matches import ToolOutputMatches
from mcp_eval.evaluators.path_efficiency import PathEfficiency
from mcp_eval.evaluators.multi_criteria_judge import (
    MultiCriteriaJudge,
    CriterionResult,
    EvaluationCriterion,
    STANDARD_CRITERIA,
    CODE_GENERATION_CRITERIA,
    SQL_QUERY_CRITERIA,
)

from mcp_eval.evaluators.base import Evaluator

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


def get_evaluator_by_name(name: str, config: Dict[str, Any]) -> Evaluator | None:
    """Get evaluator instance by name and configuration."""
    evaluator_class = _EVALUATOR_REGISTRY.get(name)
    if evaluator_class:
        return evaluator_class.from_dict(config)
    return None


def register_evaluator(name: str, evaluator_class: type):
    """Register a custom evaluator."""
    _EVALUATOR_REGISTRY[name] = evaluator_class


# Export all evaluators and shared components
__all__ = [
    # Base classes
    "Evaluator",
    # Shared components
    "EvaluatorResult",
    "EvaluationRecord",
    "JudgeResult",
    "CriterionResult",
    "EvaluationCriterion",
    # Evaluators
    "ToolWasCalled",
    "ToolSequence",
    "ResponseContains",
    "MaxIterations",
    "ToolSuccessRate",
    "LLMJudge",
    "IsInstance",
    "EqualsExpected",
    "ResponseTimeCheck",
    "ExactToolCount",
    "ToolFailed",
    "ToolCalledWith",
    "NotContains",
    "ToolOutputMatches",
    "PathEfficiency",
    "MultiCriteriaJudge",
    # Predefined criteria sets
    "STANDARD_CRITERIA",
    "CODE_GENERATION_CRITERIA",
    "SQL_QUERY_CRITERIA",
    # Registry functions
    "get_evaluator_by_name",
    "register_evaluator",
]
