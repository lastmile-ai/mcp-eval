"""Data models for evaluation reports."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..metrics import TestMetrics
from ..evaluators import EvaluationRecord


@dataclass
class CaseResult:
    """Result of evaluating a single case."""

    case_name: str
    inputs: Any
    output: Any
    expected_output: Optional[Any]
    metadata: Optional[Dict[str, Any]]
    evaluation_results: List[EvaluationRecord]
    metrics: TestMetrics
    passed: bool
    duration_ms: float
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report for a dataset."""

    dataset_name: str
    task_name: str
    results: List[CaseResult]
    metadata: Optional[Dict[str, Any]] = None

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def success_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def average_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.duration_ms for r in self.results) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "success_rate": self.success_rate,
                "average_duration_ms": self.average_duration_ms,
            },
            "results": [
                {
                    "case_name": r.case_name,
                    "inputs": r.inputs,
                    "output": r.output,
                    "expected_output": r.expected_output,
                    "metadata": r.metadata,
                    "evaluation_results": [
                        {
                            "name": eval_record.name,
                            "passed": eval_record.passed,
                            "error": eval_record.error,
                            "result": eval_record.result,
                        }
                        for eval_record in r.evaluation_results
                    ],
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "metrics": {
                        "iteration_count": r.metrics.iteration_count,
                        "tool_calls": len(r.metrics.tool_calls),
                        "latency_ms": r.metrics.latency_ms,
                        "cost_estimate": r.metrics.cost_estimate,
                    },
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }