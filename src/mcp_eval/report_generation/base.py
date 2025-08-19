"""Base utilities for report generation."""

from typing import List, Dict, Any

from mcp_eval.core import TestResult
from mcp_eval.report_generation.models import EvaluationReport


def calculate_overall_stats(
    test_results: List[TestResult], dataset_reports: List[EvaluationReport]
) -> Dict[str, Any]:
    """Calculate overall statistics for combined reports."""
    total_decorator_tests = len(test_results)
    passed_decorator_tests = sum(1 for r in test_results if r.passed)

    total_dataset_cases = sum(r.total_cases for r in dataset_reports)
    passed_dataset_cases = sum(r.passed_cases for r in dataset_reports)

    total_tests = total_decorator_tests + total_dataset_cases
    total_passed = passed_decorator_tests + passed_dataset_cases

    return {
        "total_decorator_tests": total_decorator_tests,
        "passed_decorator_tests": passed_decorator_tests,
        "total_dataset_cases": total_dataset_cases,
        "passed_dataset_cases": passed_dataset_cases,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_success_rate": (total_passed / total_tests * 100)
        if total_tests > 0
        else 0,
    }
