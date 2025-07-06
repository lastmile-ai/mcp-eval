"""Dataset and Case definitions for structured evaluation."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field

from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.builtin import EqualsExpected
from mcp_eval.metrics import TestMetrics
from mcp_eval.reports import EvaluationReport, CaseResult
from mcp_eval.session import TestSession


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
MetadataType = TypeVar("MetadataType", bound=Dict[str, Any])


@dataclass
class Case(Generic[InputType, OutputType, MetadataType]):
    """A single test case for evaluation."""

    name: str
    inputs: InputType
    expected_output: Optional[OutputType] = None
    metadata: Optional[MetadataType] = None
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
        server_name: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.cases = cases or []
        self.evaluators = evaluators or []
        self.server_name = server_name
        self.agent_config = agent_config or {}
        self.metadata = metadata or {}

    def add_case(self, case: Case[InputType, OutputType, MetadataType]):
        """Add a test case to the dataset."""
        self.cases.append(case)

    def add_evaluator(self, evaluator: Evaluator):
        """Add a global evaluator that applies to all cases."""
        self.evaluators.append(evaluator)

    async def evaluate(
        self,
        task_func: Callable[[InputType], OutputType],
        max_concurrency: Optional[int] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Evaluate the task function against all cases using unified TestSession."""
        import asyncio

        # Merge agent configurations
        final_agent_config = {**self.agent_config, **(agent_config or {})}

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def evaluate_case(case: Case) -> CaseResult:
            async def _eval():
                # Use the same unified TestSession as @task decorators
                session = TestSession(
                    server_name=self.server_name or "default",
                    test_name=case.name,
                    agent_config=final_agent_config,
                )

                try:
                    async with session as _:
                        # Execute the task
                        output = await task_func(case.inputs)

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
                        evaluation_results = {}

                        for evaluator in all_evaluators:
                            try:
                                result = await evaluator.evaluate(ctx)
                                evaluator_name = evaluator.__class__.__name__
                                evaluation_results[evaluator_name] = result
                            except Exception as e:
                                evaluation_results[evaluator.__class__.__name__] = {
                                    "error": str(e),
                                    "score": 0.0,
                                }

                        return CaseResult(
                            case_name=case.name,
                            inputs=case.inputs,
                            output=output,
                            expected_output=case.expected_output,
                            metadata=case.metadata,
                            evaluation_results=evaluation_results,
                            metrics=session.get_metrics(),
                            passed=all(
                                isinstance(r, (int, float))
                                and r > 0.5
                                or isinstance(r, dict)
                                and r.get("score", 0) > 0.5
                                or r is True
                                for r in evaluation_results.values()
                            ),
                            duration_ms=session.get_duration_ms(),
                        )

                except Exception as e:
                    return CaseResult(
                        case_name=case.name,
                        inputs=case.inputs,
                        output=None,
                        expected_output=case.expected_output,
                        metadata=case.metadata,
                        evaluation_results={},
                        metrics=TestMetrics(),
                        passed=False,
                        error=str(e),
                        duration_ms=0.0,
                    )
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

        return EvaluationReport(
            dataset_name=self.name,
            task_name=task_func.__name__,
            results=results,
            metadata=self.metadata,
        )

    def evaluate_sync(
        self,
        task_func: Callable[[InputType], OutputType],
        max_concurrency: Optional[int] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Synchronous wrapper for evaluate."""
        import asyncio

        return asyncio.run(self.evaluate(task_func, max_concurrency, agent_config))

    def to_file(self, path: Union[str, Path], format: Optional[str] = None):
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
            "agent_config": self.agent_config,
            "metadata": self.metadata,
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
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        input_type: type = str,
        output_type: type = str,
        metadata_type: type = dict,
    ) -> "Dataset":
        """Load dataset from file."""
        path = Path(path)

        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                data = json.load(f)

        # Reconstruct evaluators
        from mcp_eval.evaluators.builtin import get_evaluator_by_name

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
            agent_config=data.get("agent_config", {}),
            metadata=data.get("metadata", {}),
        )


async def generate_test_cases(
    server_name: str,
    available_tools: List[str],
    n_examples: int = 10,
    difficulty_levels: List[str] = None,
    categories: List[str] = None,
) -> List[Case]:
    """Generate test cases for an MCP server using LLM."""
    from .generation import MCPCaseGenerator

    generator = MCPCaseGenerator()
    return await generator.generate_cases(
        server_name=server_name,
        available_tools=available_tools,
        n_examples=n_examples,
        difficulty_levels=difficulty_levels or ["easy", "medium", "hard"],
        categories=categories
        or ["basic", "error_handling", "performance", "edge_cases"],
    )
