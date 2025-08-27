"""Shared classes and types for evaluators."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):
    """Standardized result format for all evaluators."""

    passed: bool = Field(description="Whether the evaluation passed")
    expected: str | int | float | List[Any] | Dict[str, Any] | None = Field(
        default=None, description="What was expected"
    )
    actual: str | int | float | List[Any] | Dict[str, Any] | None = Field(
        default=None, description="What was actually received"
    )
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score between 0.0 and 1.0"
    )
    details: Dict[str, Any] | None = Field(
        default=None, description="Additional context-specific information"
    )
    error: str | None = Field(default=None, description="Error message if applicable")

    class Config:
        extra = "forbid"


class EvaluationRecord(BaseModel):
    """Record of an evaluation result."""

    name: str = Field(description="Name of the evaluator")
    result: EvaluatorResult = Field(description="The evaluation result")
    passed: bool = Field(description="Whether the evaluation passed")
    error: str | None = Field(default=None, description="Error message if applicable")

    class Config:
        extra = "forbid"
