"""Markdown report generation for MCP-Eval."""

from typing import Dict, Any, Optional
from urllib.parse import quote

from mcp_eval.report_generation.utils import (
    get_environment_info,
    load_config_info,
    format_config_for_display,
)


def generate_combined_markdown_report(
    report_data: Dict[str, Any],
    output_path: str,
    env_info: Dict[str, Any] | None = None,
    config_data: Dict[str, Any] | None = None,
    config_path: str | None = None,
    output_dir: str = "./test-reports",
) -> None:
    """Generate a combined markdown report."""
    summary = report_data["summary"]

    # Generate environment information
    if env_info is None:
        env_info = get_environment_info()

    # Load configuration info if not provided
    if config_data is None:
        config_info = load_config_info()
        if config_info:
            config_data = config_info
            config_path = config_info.get("_config_path", "Unknown")

    # Generate configuration content
    config_content = (
        format_config_for_display(config_data)
        if config_data
        else "No configuration file found"
    )

    report = f"""# MCP-Eval Combined Test Report

<details>
<summary><strong>Environment Information</strong></summary>

**Python Version:** {env_info.get("python_version", "Unknown")}  
**Platform:** {env_info.get("platform", "Unknown")}  
**System:** {env_info.get("system", "Unknown")}  
**Machine:** {env_info.get("machine", "Unknown")}  
**Processor:** {env_info.get("processor", "Unknown")}  
**Timestamp:** {env_info.get("timestamp", "Unknown")}  
**Working Directory:** {env_info.get("working_directory", "Unknown")}

</details>

<details>
<summary><strong>MCP-Eval Configuration</strong></summary>

**Configuration loaded from:** {config_path or "Unknown"}

```yaml
{config_content}
```

</details>

## Summary

- **Decorator Tests**: {summary["passed_decorator_tests"]}/{summary["total_decorator_tests"]} passed
- **Dataset Cases**: {summary["passed_dataset_cases"]}/{summary["total_dataset_cases"]} passed
- **Overall Success Rate**: {(summary["passed_decorator_tests"] + summary["passed_dataset_cases"]) / (summary["total_decorator_tests"] + summary["total_dataset_cases"]) * 100:.1f}%

## Decorator Test Results

| Test | Status | Duration | Server | Test Report | Error |
|------|--------|----------|--------|-------------|-------|
"""

    for test_data in report_data["decorator_tests"]:
        status = "✅ PASS" if test_data["passed"] else "❌ FAIL"
        duration = f"{test_data.get('duration_ms', 0):.1f}ms"
        server = test_data.get("server_name", "unknown")

        # Create link to individual test report
        test_name = test_data["test_name"]
        # URL-encode the test name to handle special characters
        encoded_test_name = quote(test_name, safe="")
        test_report_link = f"[Link]({output_dir}/{encoded_test_name}.json)"

        error = (
            test_data.get("error", "").replace("\n", " ").replace("|", "\\|")
            if not test_data["passed"]
            else ""
        )

        report += f"| {test_name} | {status} | {duration} | {server} | {test_report_link} | {error} |\n"

    report += "\n## Dataset Evaluation Results\n\n"

    for dataset_data in report_data["dataset_reports"]:
        report += f"### {dataset_data['dataset_name']}\n\n"

        # Display summary statistics
        dataset_summary = dataset_data["summary"]
        report += f"**Summary:** {dataset_summary['passed_cases']}/{dataset_summary['total_cases']} passed ({dataset_summary['success_rate'] * 100:.1f}% success rate)\n\n"

        # Add case details table similar to decorator tests
        if dataset_data.get("results"):
            report += "| Case | Status | Duration | Test Report | Error |\n"
            report += "|------|--------|----------|-------------|-------|\n"

            for case in dataset_data["results"]:
                case_name = case.get("case_name", "Unknown")
                case_status = "✅ PASS" if case.get("passed", False) else "❌ FAIL"
                case_duration = f"{case.get('duration_ms', 0):.1f}ms"

                # Create link to individual test report (if reports are generated per case)
                # URL-encode the dataset name and case name to handle special characters
                encoded_dataset_name = quote(dataset_data["dataset_name"], safe="")
                encoded_case_name = quote(case_name, safe="")
                test_report_link = f"[Link]({output_dir}/{encoded_dataset_name}__{encoded_case_name}.json)"

                # Get error message if failed
                error = (
                    case.get("error", "").replace("\n", " ").replace("|", "\\|")
                    if not case.get("passed", False)
                    else ""
                )

                report += f"| {case_name} | {case_status} | {case_duration} | {test_report_link} | {error} |\n"

            report += "\n"

    with open(output_path, "w") as f:
        f.write(report)
