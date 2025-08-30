"""Test scenario generation and code emission for MCP servers.

This module provides two complementary approaches:
- Structured, agent-driven generation of scenarios and assertion specs
- Backward-compatible simple dataset generation
"""

from typing import List, Dict, Any, Annotated, Literal, Callable, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
import json
from datetime import datetime

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
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.factory import _llm_factory
from mcp_agent.config import LoggerSettings
from jinja2 import Environment, FileSystemLoader
from pathlib import Path


@dataclass
class MCPCaseGenerator:
    """Generates test cases for MCP servers using LLM."""

    def __init__(self, model: str | None = None):
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
    description: str | None = None
    input_schema: Dict[str, Any] | None = Field(
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
    expected_output: Dict[str, Any] | str | int | float | List[Any]
    field_path: str | None = None
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
    min_score: float = Field(0.8, ge=0.0, le=1.0)

    @field_validator("min_score", mode="before")
    @classmethod
    def _bound_min_score(cls, v):
        try:
            val = float(v)
        except Exception:
            return 0.8
        if val < 0:
            return 0.8
        # Normalize common mistakes like 8 or 80 to 0.8
        while val > 1.0:
            val = val / 10.0
        return min(1.0, max(0.0, val))


class ToolSequenceSpec(BaseModel):
    kind: Literal["tool_sequence"] = "tool_sequence"
    sequence: List[str]
    allow_other_calls: bool = False


AssertionSpec = Annotated[
    ToolWasCalledSpec
    | ToolCalledWithSpec
    | ResponseContainsSpec
    | NotContainsSpec
    | ToolOutputMatchesSpec
    | MaxIterationsSpec
    | ResponseTimeUnderSpec
    | LLMJudgeSpec
    | ToolSequenceSpec,
    Field(discriminator="kind"),
]


class ScenarioSpec(BaseModel):
    name: str
    description: str | None = None
    prompt: str
    expected_output: str | None = None
    assertions: List[AssertionSpec]


class ScenarioBundle(BaseModel):
    scenarios: List[ScenarioSpec]


class AssertionBundle(BaseModel):
    assertions: List[AssertionSpec]


def _build_llm(agent: Agent, provider: str, model: str | None):
    factory = _llm_factory(provider=provider, model=model, context=agent.context)
    return factory(agent)


def _allowed_tool_names(tools: List["ToolSchema"]) -> List[str]:
    names: List[str] = []
    for t in tools:
        if t.name:
            names.append(t.name)
    return names


def _filter_assertions_for_known_tools(
    assertions: List[AssertionSpec], allowed: List[str]
) -> List[AssertionSpec]:
    """Drop or adjust tool-related assertions that reference unknown tools."""
    filtered: List[AssertionSpec] = []
    for a in assertions:
        kind = getattr(a, "kind", None)
        if kind in ("tool_was_called", "tool_called_with", "tool_output_matches"):
            tool_name = getattr(a, "tool_name", None)
            if tool_name not in allowed:
                continue
            filtered.append(a)
        elif kind == "tool_sequence":
            seq = list(getattr(a, "sequence", []) or [])
            seq = [s for s in seq if s in allowed]
            if not seq:
                continue
            a.sequence = seq  # type: ignore[attr-defined]
            filtered.append(a)
        else:
            filtered.append(a)
    return filtered


def _harden_assertions(assertions: List[AssertionSpec]) -> List[AssertionSpec]:
    """Apply robustness tweaks to generated assertions to reduce false negatives.

    - Prefer 'contains' over 'exact' for ToolOutputMatches when expected is a short string
      and field_path is not specified (payloads are often large/structured).
    - Ensure ResponseContains defaults to case-insensitive unless explicitly requested.
    """
    hardened: List[AssertionSpec] = []
    for a in assertions:
        kind = getattr(a, "kind", None)
        try:
            if kind == "tool_output_matches":
                # a is ToolOutputMatchesSpec
                if (
                    getattr(a, "match_type", "exact") == "exact"
                    and isinstance(getattr(a, "expected_output", None), str)
                    and (getattr(a, "field_path", None) is None)
                ):
                    text = getattr(a, "expected_output")
                    # Use contains for short markers or when JSON-like not intended
                    if text and (len(text) <= 120):
                        a.match_type = "contains"  # type: ignore[attr-defined]
            elif kind == "response_contains":
                # Make response contains case-insensitive by default
                if getattr(a, "case_sensitive", None) is None:
                    a.case_sensitive = False  # type: ignore[attr-defined]
        except Exception:
            pass
        hardened.append(a)
    return hardened


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
        "- llm_judge: {rubric, min_score? (0..1)} -> LLM evaluation\n"
        "- tool_sequence: {sequence: [..], allow_other_calls?} -> path\n"
    )


async def generate_scenarios_with_agent(
    tools: List["ToolSchema"],
    *,
    n_examples: int = 8,
    provider: str = "anthropic",
    model: str | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    max_retries: int = 1,
) -> List[ScenarioSpec]:
    """Use an mcp-agent Agent to generate structured scenarios and assertion specs."""
    # Load settings to get API keys
    from mcp_eval.config import load_config

    settings = load_config()

    # Reduce logging noise (or enable info logs if debug)
    settings.logger = LoggerSettings(type="console", level="info" if debug else "error")

    app = MCPApp(settings=settings)
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
        tool_lines: List[Dict[str, Any]] = []
        for t in tools:
            nm = t.name or "unknown"
            desc = t.description or ""
            input_schema = t.input_schema or {}
            tool_lines.append(
                {"name": nm, "description": desc, "input_schema": input_schema}
            )
        allowed_names = _allowed_tool_names(tools)

        base_guidance = (
            "You are generating pytest tests for an MCP server. Each scenario is a user-facing prompt to the agent.\n"
            "STRICT SYNTAX REQUIREMENTS:\n"
            "- Use valid Python literals only (True/False/None, not true/false/null).\n"
            "- When providing argument dicts or lists, ensure they are valid Python (use single quotes or proper repr).\n"
            "- Keep function names and identifiers valid Python (snake_case).\n"
            "ASSERTION GUIDANCE:\n"
            "- Prefer realistic tool arguments and paths.\n"
            "- LLM judge min_score must be in [0.0, 1.0]. If you intend 80%, use 0.8.\n"
            "- When asserting raw HTML or JSON, prefer 'contains' on distinctive substrings rather than full document equality.\n"
            "- For tool_output_matches on large payloads, use match_type='contains' with small, unambiguous substrings.\n"
            "TOOL CONSTRAINTS:\n"
            f"- Allowed tool names: {allowed_names}. You MUST NOT reference any other tools.\n"
            "- If no tools are allowed, do not emit any Expect.tools.* assertions.\n"
        )

        def _is_valid(s: ScenarioSpec) -> bool:
            if not allowed_names:
                return True
            # Must include at least one tool-related assertion if tools exist
            return any(
                getattr(a, "kind", None)
                in (
                    "tool_was_called",
                    "tool_called_with",
                    "tool_output_matches",
                    "tool_sequence",
                )
                for a in s.assertions
            )

        attempt = 0
        while attempt <= max_retries:
            guidance = base_guidance
            if attempt > 0:
                # Harden constraints for retries
                guidance += (
                    "\nHARD REQUIREMENTS (RETRY):\n"
                    f"- Produce exactly {n_examples} scenarios.\n"
                    "- Each scenario MUST include at least one Expect.tools.* assertion referencing only allowed tools.\n"
                    "- Do NOT reference any tools outside the allowed list.\n"
                )

            payload = {
                "tools": tool_lines,
                "n_examples": n_examples,
                "assertion_catalog": _assertion_catalog_prompt(),
                "instructions": guidance,
                "few_shot_examples": [
                    {
                        "name": "basic_url_fetch",
                        "prompt": "Fetch https://httpbin.org/html and summarize",
                        "assertions": [
                            {
                                "kind": "tool_was_called",
                                "tool_name": "fetch",
                                "min_times": 1,
                            },
                            {
                                "kind": "tool_called_with",
                                "tool_name": "fetch",
                                "arguments": {"url": "https://httpbin.org/html"},
                            },
                            {
                                "kind": "response_contains",
                                "text": "Herman Melville",
                                "case_sensitive": False,
                            },
                            {
                                "kind": "llm_judge",
                                "rubric": "Response shows content was fetched and summarized",
                                "min_score": 0.8,
                            },
                        ],
                    }
                ],
                "allowed_tools": allowed_names,
            }

            prompt = (
                "Design high-quality test scenarios for the tools below. Return a JSON object that adheres to the provided Pydantic schema.\n"
                + json.dumps(payload, indent=2)
            )

            if progress_callback:
                progress_callback("Calling LLM to generate scenarios...")
                if debug:
                    progress_callback(f"DEBUG allowed_tools={allowed_names}")
                    # Print full prompt (no truncation) and write to a file for inspection
                    progress_callback("DEBUG prompt preview (full):")
                    progress_callback(prompt)
                    try:
                        log_dir = Path("test-reports")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        fp = (
                            log_dir
                            / f"prompt_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        )
                        fp.write_text(prompt, encoding="utf-8")
                        progress_callback(f"DEBUG prompt saved to: {fp}")
                    except Exception:
                        pass

            bundle = await llm.generate_structured(
                prompt,
                response_model=ScenarioBundle,
                request_params=RequestParams(maxTokens=21000),
            )
            # Filter out any assertions that reference unknown tools
            if allowed_names:
                for s in bundle.scenarios:
                    s.assertions = _filter_assertions_for_known_tools(
                        s.assertions, allowed_names
                    )
                    s.assertions = _harden_assertions(s.assertions)
            else:
                for s in bundle.scenarios:
                    s.assertions = [
                        a
                        for a in s.assertions
                        if getattr(a, "kind", None)
                        not in (
                            "tool_was_called",
                            "tool_called_with",
                            "tool_output_matches",
                            "tool_sequence",
                        )
                    ]
                    s.assertions = _harden_assertions(s.assertions)

            scenarios = [s for s in bundle.scenarios if _is_valid(s)]
            if progress_callback:
                progress_callback(f"Generated {len(scenarios)} scenarios")

            # Retry if we got zero valid scenarios and we have allowed tools
            if allowed_names and len(scenarios) == 0 and attempt < max_retries:
                attempt += 1
                continue
            return scenarios


async def refine_assertions_with_agent(
    scenarios: List[ScenarioSpec],
    tools: List["ToolSchema"],
    *,
    provider: str = "anthropic",
    model: str | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    debug: bool = False,
) -> List[ScenarioSpec]:
    """For each scenario, ask an agent to propose additional assertions using available tool schemas and the assertion catalog."""
    if not scenarios:
        return scenarios

    # Load settings to get API keys
    from mcp_eval.config import load_config

    settings = load_config()

    # Reduce logging noise (or enable info logs if debug)
    settings.logger = LoggerSettings(type="console", level="info" if debug else "error")

    app = MCPApp(settings=settings)
    async with app.run() as running:
        agent = Agent(
            name="assertion_refiner",
            instruction="You propose precise assertions.",
            server_names=[],
            context=running.context,
        )
        llm = _build_llm(agent, provider, model)

        tool_lines: List[Dict[str, Any]] = []
        for t in tools:
            tool_lines.append(
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.input_schema or {},
                }
            )
        allowed_names = _allowed_tool_names(tools)

        updated: List[ScenarioSpec] = []
        for i, s in enumerate(scenarios, 1):
            if progress_callback:
                progress_callback(
                    f"Refining assertions for scenario {i}/{len(scenarios)}: {s.name}"
                )
            payload = {
                "scenario": {
                    "name": s.name,
                    "prompt": s.prompt,
                    "expected_output": s.expected_output,
                },
                "tools": tool_lines,
                "assertion_catalog": _assertion_catalog_prompt(),
                "guidance": (
                    "Propose additional assertions that increase coverage: argument checks, tool outputs, sequences, performance and judge where applicable.\n"
                    f"HARD CONSTRAINT: Only use these tool names: {allowed_names}. Do not invent other tools."
                ),
                "allowed_tools": allowed_names,
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
                    # Drop any references to unknown tools
                    if allowed_names and getattr(a, "kind", None) in (
                        "tool_was_called",
                        "tool_called_with",
                        "tool_output_matches",
                    ):
                        if getattr(a, "tool_name", None) not in allowed_names:
                            continue
                    if getattr(a, "kind", None) == "tool_sequence":
                        seq = list(getattr(a, "sequence", []) or [])
                        seq = [t for t in seq if t in allowed_names]
                        if not seq:
                            continue
                        a.sequence = seq  # type: ignore[attr-defined]
                    key = f"{a.kind}:{repr(a)}"
                    if key not in have:
                        merged.append(a)
                        have.add(key)
                s.assertions = _harden_assertions(merged)
            except Exception:
                pass
            updated.append(s)

        if progress_callback:
            progress_callback(f"Completed refining {len(updated)} scenarios")

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

    # Render values as valid Python literals (True/False/None, dict/list via repr)
    def py_value(value: Any) -> str:
        try:
            return repr(value)
        except Exception:
            return repr(str(value))

    env.filters["py"] = py_value
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
