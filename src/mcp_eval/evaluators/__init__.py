"""Evaluators package - imports all evaluators and shared components."""

from typing import Any, Dict, Optional

# Import shared components
from .shared import EvaluatorResult, EvaluationRecord

# Import all evaluators from separate files
from .tool_was_called import ToolWasCalled
from .tool_sequence import ToolSequence
from .response_contains import ResponseContains
from .max_iterations import MaxIterations
from .tool_success_rate import ToolSuccessRate
from .llm_judge import LLMJudge, JudgeResult
from .is_instance import IsInstance
from .equals_expected import EqualsExpected
from .response_time_check import ResponseTimeCheck
from .exact_tool_count import ExactToolCount
from .tool_failed import ToolFailed
from .tool_called_with import ToolCalledWith
from .not_contains import NotContains
from .tool_output_matches import ToolOutputMatches
from .path_efficiency import PathEfficiency
from .multi_criteria_judge import (
    MultiCriteriaJudge,
    CriterionResult,
    EvaluationCriterion,
    STANDARD_CRITERIA,
    CODE_GENERATION_CRITERIA,
    SQL_QUERY_CRITERIA,
)

from .base import Evaluator

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