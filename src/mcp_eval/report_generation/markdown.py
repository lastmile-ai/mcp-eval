"""Markdown report generation for MCP-Eval."""

from typing import Dict, Any


def generate_combined_markdown_report(report_data: Dict[str, Any], output_path: str) -> None:
    """Generate a combined markdown report."""
    summary = report_data["summary"]

    report = f"""# MCP-Eval Combined Test Report

## Summary

- **Decorator Tests**: {summary["passed_decorator_tests"]}/{summary["total_decorator_tests"]} passed
- **Dataset Cases**: {summary["passed_dataset_cases"]}/{summary["total_dataset_cases"]} passed
- **Overall Success Rate**: {(summary["passed_decorator_tests"] + summary["passed_dataset_cases"]) / (summary["total_decorator_tests"] + summary["total_dataset_cases"]) * 100:.1f}%

## Decorator Test Results

| Test | Status | Duration | Server |
|------|--------|----------|--------|
"""

    for test_data in report_data["decorator_tests"]:
        status = "✅ PASS" if test_data["passed"] else "❌ FAIL"
        duration = f"{test_data.get('duration_ms', 0):.1f}ms"
        server = test_data.get("server_name", "unknown")

        report += f"| {test_data['test_name']} | {status} | {duration} | {server} |\n"

    report += "\n## Dataset Evaluation Results\n\n"

    for dataset_data in report_data["dataset_reports"]:
        dataset_summary = dataset_data["summary"]
        report += f"### {dataset_data['dataset_name']}\n\n"
        report += f"- **Cases**: {dataset_summary['passed_cases']}/{dataset_summary['total_cases']} passed\n"
        report += f"- **Success Rate**: {dataset_summary['success_rate'] * 100:.1f}%\n"
        report += f"- **Average Duration**: {dataset_summary['average_duration_ms']:.1f}ms\n\n"

    with open(output_path, "w") as f:
        f.write(report)