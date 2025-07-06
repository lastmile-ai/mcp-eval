"""Dataset generation using LLMs."""

from typing import List, Dict, Any
from dataclasses import dataclass

from mcp_eval.datasets import Case, Dataset
from mcp_eval.evaluators.builtin import ToolWasCalled, ResponseContains, LLMJudge


@dataclass
class MCPCaseGenerator:
    """Generates test cases for MCP servers using LLM."""

    def __init__(self, model: str = "claude-3-haiku-20240307"):
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
            import json

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
