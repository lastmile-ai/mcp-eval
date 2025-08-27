"""Dataset and Case definitions for structured evaluation."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Generic, Callable
from dataclasses import dataclass, field

from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators import (
    EqualsExpected,
    EvaluatorResult,
    EvaluationRecord,
    get_evaluator_by_name,
)
from mcp_eval.metrics import TestMetrics
from mcp_eval.report_generation.console import generate_failure_message
from mcp_eval.report_generation.models import EvaluationReport, CaseResult
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_eval.session import TestSession


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
MetadataType = TypeVar("MetadataType", bound=Dict[str, Any])


@dataclass
class Case(Generic[InputType, OutputType, MetadataType]):
    """A single test case for evaluation."""

    name: str
    inputs: InputType
    expected_output: OutputType | None = None
    metadata: MetadataType | None = None
    evaluators: List[Evaluator] = field(default_factory=list)

    def __post_init__(self):
        """Add default evaluators if expected_output is provided."""
        if self.expected_output is not None and not any(
            isinstance(e, EqualsExpected) for e in self.evaluators
        ):
            self.evaluators.append(EqualsExpected())


class Dataset(Generic[InputType, OutputType, MetadataType]):
    """
    A collection of test cases for systematic evaluation.
    Uses the same unified TestSession as @task decorators.
    """

    def __init__(
        self,
        name: str = "Unnamed Dataset",
        cases: List[Case[InputType, OutputType, MetadataType]] = None,
        evaluators: List[Evaluator] = None,
        server_name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        agent_spec: AgentSpec | str | None = None,
    ):
        self.name = name
        self.cases = cases or []
        self.evaluators = evaluators or []
        self.server_name = server_name
        self.metadata = metadata or {}
        # Agent selection for dataset evaluation (AgentSpec object or spec name)
        self.agent_spec: AgentSpec | str | None = agent_spec

    def add_case(self, case: Case[InputType, OutputType, MetadataType]):
        """Add a test case to the dataset."""
        self.cases.append(case)

    def add_evaluator(self, evaluator: Evaluator):
        """Add a global evaluator that applies to all cases."""
        self.evaluators.append(evaluator)

    async def evaluate(
        self,
        task_func: Callable,
        max_concurrency: int | None = None,
        progress_callback: Callable[[bool, bool], None] | None = None,
    ) -> EvaluationReport:
        """Evaluate the task function against all cases using unified TestSession.

        Task function signature: async def task_func(inputs, agent, session) -> output
        """
        import asyncio

        # No agent_config support; rely on AgentSpecs and global settings

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def evaluate_case(case: Case) -> CaseResult:
            async def _eval():
                # Use the same unified TestSession as @task decorators
                # Format test name consistently with how runner.py displays it
                session = TestSession(
                    test_name=f"{self.name}::{case.name}",
                    agent_override=self.agent_spec,
                )

                try:
                    async with session as agent:
                        # Execute the task with session access
                        output = await task_func(case.inputs, agent, session)

                        # Run evaluators
                        ctx = EvaluatorContext(
                            inputs=case.inputs,
                            output=output,
                            expected_output=case.expected_output,
                            metadata=case.metadata,
                            metrics=session.get_metrics(),
                            span_tree=session.get_span_tree(),
                        )

                        # Combine case-specific and global evaluators
                        all_evaluators = case.evaluators + self.evaluators
                        evaluation_results: list[EvaluationRecord] = []

                        for evaluator in all_evaluators:
                            try:
                                result = await evaluator.evaluate(ctx)
                                evaluator_name = evaluator.__class__.__name__

                                evaluation_results.append(
                                    EvaluationRecord(
                                        name=evaluator_name,
                                        result=result,
                                        passed=result.passed,
                                        error=result.error,
                                    )
                                )
                            except Exception as e:
                                evaluation_results.append(
                                    EvaluationRecord(
                                        name=evaluator.__class__.__name__,
                                        result=EvaluatorResult(
                                            passed=False, error=str(e)
                                        ),
                                        passed=False,
                                        error=str(e),
                                    )
                                )

                        case_result = CaseResult(
                            case_name=case.name,
                            inputs=case.inputs,
                            output=output,
                            expected_output=case.expected_output,
                            metadata=case.metadata,
                            evaluation_results=evaluation_results,
                            metrics=session.get_metrics(),
                            passed=all(r.passed for r in evaluation_results),
                            duration_ms=session.get_duration_ms(),
                            agent_name=session.agent.name if session.agent else None,
                            servers=session.agent.server_names
                            if session.agent
                            else None,
                            error=generate_failure_message(evaluation_results),
                        )

                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(case_result.passed, False)

                        return case_result

                except Exception as e:
                    case_result = CaseResult(
                        case_name=case.name,
                        inputs=case.inputs,
                        output=None,
                        expected_output=case.expected_output,
                        metadata=case.metadata,
                        evaluation_results=[],
                        metrics=TestMetrics(),
                        passed=False,
                        error=str(e),
                        duration_ms=0.0,
                        agent_name=session.agent.name
                        if session and session.agent
                        else None,
                        servers=session.agent.server_names
                        if session and session.agent
                        else None,
                    )

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(False, True)

                    return case_result
                finally:
                    session.cleanup()

            if semaphore is not None:
                async with semaphore:
                    return await _eval()
            else:
                return await _eval()

        # Run all cases
        tasks = [evaluate_case(case) for case in self.cases]
        results = await asyncio.gather(*tasks)

        # Extract agent name from first result if available
        agent_name = None
        if results and results[0].agent_name:
            agent_name = results[0].agent_name

        return EvaluationReport(
            dataset_name=self.name,
            task_name=task_func.__name__,
            results=results,
            metadata=self.metadata,
            agent_name=agent_name,
        )

    def evaluate_sync(
        self,
        task_func: Callable[[InputType], OutputType],
        max_concurrency: int | None = None,
    ) -> EvaluationReport:
        """Synchronous wrapper for evaluate."""
        import asyncio

        return asyncio.run(self.evaluate(task_func, max_concurrency))

    def to_file(self, path: str | Path, format: str | None = None):
        """Save dataset to file in YAML or JSON format."""
        path = Path(path)
        if format is None:
            format = path.suffix.lower().lstrip(".")

        if format not in ["yaml", "yml", "json"]:
            raise ValueError(f"Unsupported format: {format}")

        # Convert to serializable format
        data = {
            "name": self.name,
            "server_name": self.server_name,
            # Agent configuration removed; configure via AgentSpecs and settings
            "metadata": self.metadata,
            # Persist agent_spec as a name if possible; otherwise omit complex objects
            "agent_spec": (
                self.agent_spec.name
                if isinstance(self.agent_spec, AgentSpec)
                else self.agent_spec
            ),
            "cases": [
                {
                    "name": case.name,
                    "inputs": case.inputs,
                    "expected_output": case.expected_output,
                    "metadata": case.metadata,
                    "evaluators": [
                        {type(e).__name__: e.to_dict()} for e in case.evaluators
                    ],
                }
                for case in self.cases
            ],
            "evaluators": [{type(e).__name__: e.to_dict()} for e in self.evaluators],
        }

        if format in ["yaml", "yml"]:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        input_type: type = str,
        output_type: type = str,
        metadata_type: type = dict,
    ) -> "Dataset":
        """Load dataset from file."""
        path = Path(path)

        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        cases = []
        for case_data in data.get("cases", []):
            evaluators = []
            for eval_data in case_data.get("evaluators", []):
                for eval_name, eval_config in eval_data.items():
                    evaluator = get_evaluator_by_name(eval_name, eval_config)
                    if evaluator:
                        evaluators.append(evaluator)

            cases.append(
                Case(
                    name=case_data["name"],
                    inputs=case_data["inputs"],
                    expected_output=case_data.get("expected_output"),
                    metadata=case_data.get("metadata"),
                    evaluators=evaluators,
                )
            )

        global_evaluators = []
        for eval_data in data.get("evaluators", []):
            for eval_name, eval_config in eval_data.items():
                evaluator = get_evaluator_by_name(eval_name, eval_config)
                if evaluator:
                    global_evaluators.append(evaluator)

        return cls(
            name=data.get("name", "Loaded Dataset"),
            cases=cases,
            evaluators=global_evaluators,
            server_name=data.get("server_name"),
            metadata=data.get("metadata", {}),
            # Inline dict AgentSpec overrides were removed. Accept only a spec name (str) or None.
            agent_spec=(
                data.get("agent_spec") if isinstance(data.get("agent_spec"), str) else None
            ),
        )


async def generate_test_cases(
    server_name: str,
    available_tools: List[str],
    n_examples: int = 10,
    difficulty_levels: List[str] = None,
    categories: List[str] = None,
) -> List[Case]:
    """Generate test cases for an MCP server using LLM."""
    from mcp_eval.generation import MCPCaseGenerator

    generator = MCPCaseGenerator()
    return await generator.generate_cases(
        server_name=server_name,
        available_tools=available_tools,
        n_examples=n_examples,
        difficulty_levels=difficulty_levels or ["easy", "medium", "hard"],
        categories=categories
        or ["basic", "error_handling", "performance", "edge_cases"],
    )
