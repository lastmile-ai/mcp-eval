"""Test scenario generation and code emission for MCP servers.

This module provides two complementary approaches:
- Structured, agent-driven generation of scenarios and assertion specs
- Backward-compatible simple dataset generation
"""

from typing import List, Dict, Any, Optional, Annotated, Literal, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field
import json

from mcp_eval.datasets import Case, Dataset
from mcp_eval.evaluators import (
    ToolWasCalled,
    ResponseContains,
    LLMJudge,
    ToolCalledWith,
    ToolOutputMatches,
    MaxIterations,
    ResponseTimeCheck,
    ToolSequence,
)

# mcp-agent integration for agent-driven scenario generation
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.factory import _llm_factory
from jinja2 import Environment, FileSystemLoader
from pathlib import Path


@dataclass
class MCPCaseGenerator:
    """Generates test cases for MCP servers using LLM."""

    def __init__(self, model: Optional[str] = None):
        """Initialize generator with optional model override.

        If no model is provided, will use settings configuration.
        """
        self.model = model

    async def generate_cases(
        self,
        server_name: str,
        available_tools: List[str],
        n_examples: int = 10,
        difficulty_levels: List[str] = None,
        categories: List[str] = None,
    ) -> List[Case]:
        """Generate test cases for an MCP server."""
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]

        if categories is None:
            categories = [
                "basic_functionality",
                "error_handling",
                "edge_cases",
                "performance",
            ]

        # Create prompt for case generation
        prompt = self._build_generation_prompt(
            server_name=server_name,
            available_tools=available_tools,
            n_examples=n_examples,
            difficulty_levels=difficulty_levels,
            categories=categories,
        )

        # Generate cases using LLM
        from mcp_eval.llm_client import get_judge_client

        client = get_judge_client(self.model)

        try:
            response = await client.generate_str(prompt)

            # Parse the JSON response
            cases_data = json.loads(response)

            # Convert to Case objects
            cases = []
            for case_data in cases_data.get("cases", []):
                evaluators = self._create_evaluators_for_case(
                    case_data, available_tools
                )

                case = Case(
                    name=case_data["name"],
                    inputs=case_data["inputs"],
                    expected_output=case_data.get("expected_output"),
                    metadata=case_data.get("metadata", {}),
                    evaluators=evaluators,
                )
                cases.append(case)

            return cases

        except Exception:
            # Fallback to manual case generation
            return self._generate_fallback_cases(
                server_name, available_tools, n_examples
            )

    def _build_generation_prompt(
        self,
        server_name: str,
        available_tools: List[str],
        n_examples: int,
        difficulty_levels: List[str],
        categories: List[str],
    ) -> str:
        """Build the prompt for LLM case generation."""
        return f"""
        Generate {n_examples} diverse test cases for an MCP server named '{server_name}' with the following tools:
        {", ".join(available_tools)}
        
        Create test cases across these difficulty levels: {", ".join(difficulty_levels)}
        And these categories: {", ".join(categories)}
        
        For each test case, include:
        1. A unique name (snake_case)
        2. Input text (what to ask the agent to do)
        3. Expected output (optional, if deterministic)
        4. Metadata with difficulty and category
        5. Expected tools that should be used
        
        Guidelines:
        - Test individual tools and combinations
        - Include error scenarios (invalid inputs, edge cases)
        - Test performance scenarios (efficiency, parallel usage)
        - Ensure diversity in complexity and approach
        
        Return the result as JSON in this format:
        {{
            "cases": [
                {{
                    "name": "test_basic_functionality",
                    "inputs": "Do something with the server",
                    "expected_output": "Expected result (optional)",
                    "metadata": {{
                        "difficulty": "easy",
                        "category": "basic_functionality",
                        "expected_tools": ["tool1", "tool2"],
                        "description": "Brief description of what this tests"
                    }}
                }}
            ]
        }}
        """

    def _create_evaluators_for_case(
        self, case_data: Dict[str, Any], available_tools: List[str]
    ) -> List:
        """Create appropriate evaluators for a generated case."""
        evaluators = []
        metadata = case_data.get("metadata", {})

        # Add tool usage evaluators
        expected_tools = metadata.get("expected_tools", [])
        for tool in expected_tools:
            if tool in available_tools:
                evaluators.append(ToolWasCalled(tool_name=tool))

        # Add content evaluators if expected output exists
        if case_data.get("expected_output"):
            evaluators.append(ResponseContains(text=case_data["expected_output"]))

        # Add LLM judge for more complex scenarios
        if metadata.get("category") in ["error_handling", "edge_cases"]:
            evaluators.append(
                LLMJudge(
                    rubric=f"Response appropriately handles the {metadata.get('category', 'scenario')} scenario"
                )
            )

        return evaluators

    def _generate_fallback_cases(
        self, server_name: str, available_tools: List[str], n_examples: int
    ) -> List[Case]:
        """Generate basic fallback cases if LLM generation fails."""
        cases = []

        # Basic functionality cases for each tool
        for i, tool in enumerate(available_tools[:n_examples]):
            case = Case(
                name=f"test_{tool}_basic",
                inputs=f"Use the {tool} tool to perform its basic function",
                metadata={
                    "difficulty": "easy",
                    "category": "basic_functionality",
                    "expected_tools": [tool],
                },
                evaluators=[ToolWasCalled(tool_name=tool)],
            )
            cases.append(case)

        return cases


async def generate_dataset(
    dataset_type: type,
    server_name: str,
    available_tools: List[str] = None,
    n_examples: int = 10,
    extra_instructions: str = "",
) -> Dataset:
    """Generate a complete dataset for an MCP server."""
    if available_tools is None:
        # Would typically introspect the server to get available tools
        available_tools = []

    generator = MCPCaseGenerator()
    cases = await generator.generate_cases(
        server_name=server_name,
        available_tools=available_tools,
        n_examples=n_examples,
    )

    return Dataset(
        name=f"Generated tests for {server_name}",
        cases=cases,
        server_name=server_name,
        metadata={
            "generated": True,
            "generator_version": "0.2.0",
            "extra_instructions": extra_instructions,
        },
    )


# =====================
# Agent-driven generation
# =====================


class ToolSchema(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for tool input"
    )


class ToolWasCalledSpec(BaseModel):
    kind: Literal["tool_was_called"] = "tool_was_called"
    tool_name: str
    min_times: int = 1


class ToolCalledWithSpec(BaseModel):
    kind: Literal["tool_called_with"] = "tool_called_with"
    tool_name: str
    arguments: Dict[str, Any]


class ResponseContainsSpec(BaseModel):
    kind: Literal["response_contains"] = "response_contains"
    text: str
    case_sensitive: bool = False


class NotContainsSpec(BaseModel):
    kind: Literal["not_contains"] = "not_contains"
    text: str
    case_sensitive: bool = False


class ToolOutputMatchesSpec(BaseModel):
    kind: Literal["tool_output_matches"] = "tool_output_matches"
    tool_name: str
    expected_output: Union[Dict[str, Any], str, int, float, List[Any]]
    field_path: Optional[str] = None
    match_type: str = Field("exact", description="exact|contains|regex|partial")
    case_sensitive: bool = True
    call_index: int = -1


class MaxIterationsSpec(BaseModel):
    kind: Literal["max_iterations"] = "max_iterations"
    max_iterations: int


class ResponseTimeUnderSpec(BaseModel):
    kind: Literal["response_time_under"] = "response_time_under"
    ms: float


class LLMJudgeSpec(BaseModel):
    kind: Literal["llm_judge"] = "llm_judge"
    rubric: str
    min_score: float = 0.8


class ToolSequenceSpec(BaseModel):
    kind: Literal["tool_sequence"] = "tool_sequence"
    sequence: List[str]
    allow_other_calls: bool = False


AssertionSpec = Annotated[
    Union[
        ToolWasCalledSpec,
        ToolCalledWithSpec,
        ResponseContainsSpec,
        NotContainsSpec,
        ToolOutputMatchesSpec,
        MaxIterationsSpec,
        ResponseTimeUnderSpec,
        LLMJudgeSpec,
        ToolSequenceSpec,
    ],
    Field(discriminator="kind"),
]


class ScenarioSpec(BaseModel):
    name: str
    description: Optional[str] = None
    prompt: str
    expected_output: Optional[str] = None
    assertions: List[AssertionSpec]


class ScenarioBundle(BaseModel):
    scenarios: List[ScenarioSpec]


class AssertionBundle(BaseModel):
    assertions: List[AssertionSpec]


def _build_llm(agent: Agent, provider: str, model: Optional[str]):
    factory = _llm_factory(provider=provider, model=model, context=agent.context)
    return factory(agent)


def _assertion_catalog_prompt() -> str:
    return (
        "You can choose from these assertion types (use discriminated 'kind' field):\n"
        "- tool_was_called: {tool_name, min_times} -> verify tool usage\n"
        "- tool_called_with: {tool_name, arguments} -> verify arguments\n"
        "- response_contains: {text, case_sensitive?} -> content contains\n"
        "- not_contains: {text, case_sensitive?} -> content excludes\n"
        "- tool_output_matches: {tool_name, expected_output, field_path?, match_type?, case_sensitive?, call_index?}\n"
        "- max_iterations: {max_iterations} -> iteration budget\n"
        "- response_time_under: {ms} -> latency budget\n"
        "- llm_judge: {rubric, min_score?} -> LLM evaluation\n"
        "- tool_sequence: {sequence: [..], allow_other_calls?} -> path\n"
    )


async def generate_scenarios_with_agent(
    tools: List[Dict[str, Any]],
    *,
    n_examples: int = 8,
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> List[ScenarioSpec]:
    """Use an mcp-agent Agent to generate structured scenarios and assertion specs."""
    app = MCPApp()
    async with app.run() as running:
        # Minimal agent just for content generation
        agent = Agent(
            name="test_generator",
            instruction="You design high-quality tests.",
            server_names=[],
            context=running.context,
        )
        llm = _build_llm(agent, provider, model)

        # Build prompt with tool schemas and assertion catalog
        tool_lines = []
        for t in tools:
            nm = t.get("name") or "unknown"
            desc = t.get("description") or ""
            input_schema = (
                t.get("input_schema") or t.get("inputSchema") or t.get("input") or {}
            )
            tool_lines.append(
                {"name": nm, "description": desc, "input_schema": input_schema}
            )

        guidance = (
            "You are generating test scenarios for an MCP server. Each scenario is a user-facing prompt to the agent.\n"
            "For each, propose appropriate assertions using the available assertion catalog.\n"
            "Include path/efficiency or judge assertions when beneficial.\n"
        )

        payload = {
            "tools": tool_lines,
            "n_examples": n_examples,
            "assertion_catalog": _assertion_catalog_prompt(),
            "instructions": guidance,
        }

        prompt = (
            "Design high-quality test scenarios for the tools below. Return a JSON object that adheres to the provided Pydantic schema.\n"
            + json.dumps(payload, indent=2)
        )

        bundle = await llm.generate_structured(prompt, response_model=ScenarioBundle)
        return bundle.scenarios


async def refine_assertions_with_agent(
    scenarios: List[ScenarioSpec],
    tools: List[Dict[str, Any]],
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> List[ScenarioSpec]:
    """For each scenario, ask an agent to propose additional assertions using available tool schemas and the assertion catalog."""
    if not scenarios:
        return scenarios
    app = MCPApp()
    async with app.run() as running:
        agent = Agent(
            name="assertion_refiner",
            instruction="You propose precise assertions.",
            server_names=[],
            context=running.context,
        )
        llm = _build_llm(agent, provider, model)

        tool_lines = []
        for t in tools:
            tool_lines.append(
                {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "input_schema": t.get("input_schema") or {},
                }
            )

        updated: List[ScenarioSpec] = []
        for s in scenarios:
            payload = {
                "scenario": {
                    "name": s.name,
                    "prompt": s.prompt,
                    "expected_output": s.expected_output,
                },
                "tools": tool_lines,
                "assertion_catalog": _assertion_catalog_prompt(),
                "guidance": "Propose additional assertions that increase coverage: argument checks, tool outputs, sequences, performance and judge where applicable.",
            }
            prompt = (
                "Given the scenario and tool specs, return an AssertionBundle JSON following the schema.\n"
                + json.dumps(payload, indent=2)
            )
            try:
                bundle = await llm.generate_structured(
                    prompt, response_model=AssertionBundle
                )
                # Merge assertions (append; naive de-dupe by kind+repr)
                have = {f"{a.kind}:{repr(a)}" for a in s.assertions}
                merged = list(s.assertions)
                for a in bundle.assertions:
                    key = f"{a.kind}:{repr(a)}"
                    if key not in have:
                        merged.append(a)
                        have.add(key)
                s.assertions = merged
            except Exception:
                pass
            updated.append(s)
        return updated


def _spec_to_evaluator(spec: AssertionSpec):
    kind = getattr(spec, "kind", None)
    if kind == "tool_was_called":
        return ToolWasCalled(tool_name=spec.tool_name, min_times=spec.min_times)
    if kind == "tool_called_with":
        return ToolCalledWith(spec.tool_name, spec.arguments)
    if kind == "response_contains":
        return ResponseContains(text=spec.text, case_sensitive=spec.case_sensitive)
    if kind == "not_contains":
        from mcp_eval.evaluators import NotContains

        return NotContains(text=spec.text, case_sensitive=spec.case_sensitive)
    if kind == "tool_output_matches":
        return ToolOutputMatches(
            tool_name=spec.tool_name,
            expected_output=spec.expected_output,
            field_path=spec.field_path,
            match_type=spec.match_type,
            case_sensitive=spec.case_sensitive,
            call_index=spec.call_index,
        )
    if kind == "max_iterations":
        return MaxIterations(max_iterations=spec.max_iterations)
    if kind == "response_time_under":
        return ResponseTimeCheck(max_ms=spec.ms)
    if kind == "llm_judge":
        return LLMJudge(rubric=spec.rubric, min_score=spec.min_score)
    if kind == "tool_sequence":
        return ToolSequence(spec.sequence, allow_other_calls=spec.allow_other_calls)
    raise ValueError(f"Unknown assertion spec kind: {kind}")


def scenarios_to_cases(scenarios: List[ScenarioSpec]) -> List[Case]:
    cases: List[Case] = []
    for s in scenarios:
        evaluators = []
        for a in s.assertions:
            try:
                evaluators.append(_spec_to_evaluator(a))
            except Exception:
                continue
        cases.append(
            Case(
                name=s.name,
                inputs=s.prompt,
                expected_output=s.expected_output,
                metadata={"description": s.description} if s.description else {},
                evaluators=evaluators,
            )
        )
    return cases


def _create_jinja_env() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def py_ident(value: str) -> str:
        import re

        s = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
        if not s:
            s = "generated"
        if s[0].isdigit():
            s = f"gen_{s}"
        return s

    env.filters["py_ident"] = py_ident
    return env


def render_pytest_tests(scenarios: List[ScenarioSpec], server_name: str) -> str:
    env = _create_jinja_env()
    tmpl = env.get_template("test_pytest_generated.py.j2")
    return tmpl.render(scenarios=scenarios, server_name=server_name)


def render_decorator_tests(scenarios: List[ScenarioSpec], server_name: str) -> str:
    env = _create_jinja_env()
    tmpl = env.get_template("test_decorators_generated.py.j2")
    return tmpl.render(scenarios=scenarios, server_name=server_name)


def dataset_from_scenarios(scenarios: List[ScenarioSpec], server_name: str) -> Dataset:
    cases: List[Case] = []
    for s in scenarios:
        cases.append(
            Case(name=s.name, inputs=s.prompt, expected_output=s.expected_output)
        )
    return Dataset(
        name=f"Generated dataset for {server_name}",
        cases=cases,
        server_name=server_name,
    )
