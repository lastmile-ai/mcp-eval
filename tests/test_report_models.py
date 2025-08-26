from mcp_eval.report_generation.models import CaseResult, EvaluationReport
from mcp_eval.metrics import TestMetrics
from mcp_eval.evaluators.shared import EvaluationRecord, EvaluatorResult


def test_evaluation_report_properties_and_serialization():
    metrics = TestMetrics()
    metrics.iteration_count = 2
    metrics.latency_ms = 123.4
    metrics.cost_estimate = 0.01

    rec = EvaluationRecord(
        name="EqualsExpected",
        result=EvaluatorResult(passed=True),
        passed=True,
        error=None,
    )

    case = CaseResult(
        case_name="c1",
        inputs="i",
        output="o",
        expected_output="o",
        metadata={"k": "v"},
        evaluation_results=[rec],
        metrics=metrics,
        passed=True,
        duration_ms=10.0,
    )

    report = EvaluationReport(dataset_name="ds", task_name="task", results=[case])

    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.success_rate == 1.0
    assert report.average_duration_ms == 10.0

    d = report.to_dict()
    assert d["summary"]["total_cases"] == 1
    assert d["results"][0]["metrics"]["iteration_count"] == 2
