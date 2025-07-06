"""Rich reporting for evaluation results."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from mcp_eval.metrics import TestMetrics


@dataclass
class CaseResult:
    """Result of evaluating a single case."""

    case_name: str
    inputs: Any
    output: Any
    expected_output: Optional[Any]
    metadata: Optional[Dict[str, Any]]
    evaluation_results: Dict[str, Any]
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

    def print(
        self,
        include_input: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_scores: bool = True,
        console: Optional[Console] = None,
    ):
        """Print a rich table of evaluation results."""
        if console is None:
            console = Console()

        # Summary table
        table = Table(title=f"Evaluation Summary: {self.task_name}")

        # Add columns based on options
        table.add_column("Case ID", style="cyan")

        if include_input:
            table.add_column("Inputs")

        if include_output:
            table.add_column("Outputs")

        if include_scores:
            table.add_column("Scores", style="yellow")

        table.add_column("Status", justify="center")

        if include_durations:
            table.add_column("Duration", justify="right", style="green")

        # Add rows for each case
        for result in self.results:
            row_data = []

            # Case ID
            row_data.append(result.case_name)

            # Inputs
            if include_input:
                input_str = str(result.inputs)
                if len(input_str) > 50:
                    input_str = input_str[:47] + "..."
                row_data.append(input_str)

            # Outputs
            if include_output:
                output_str = str(result.output)
                if len(output_str) > 50:
                    output_str = output_str[:47] + "..."
                row_data.append(output_str)

            # Scores
            if include_scores:
                scores = []
                for evaluator_name, eval_result in result.evaluation_results.items():
                    if isinstance(eval_result, (int, float)):
                        scores.append(f"{evaluator_name}: {eval_result:.2f}")
                    elif isinstance(eval_result, dict) and "score" in eval_result:
                        scores.append(f"{evaluator_name}: {eval_result['score']:.2f}")
                    elif isinstance(eval_result, bool):
                        scores.append(
                            f"{evaluator_name}: {'✓' if eval_result else '✗'}"
                        )
                row_data.append("\n".join(scores))

            # Status (pass/fail indicators)
            status = "✅ PASS" if result.passed else "❌ FAIL"
            if result.error:
                status += f" ({result.error})"
            row_data.append(status)

            # Duration
            if include_durations:
                row_data.append(f"{result.duration_ms:.0f}ms")

            table.add_row(*row_data)

        console.print(table)
        console.print(
            f"\nSummary: {self.passed_cases}/{self.total_cases} cases passed ({self.success_rate:.1%})"
        )

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
                    "evaluation_results": r.evaluation_results,
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
