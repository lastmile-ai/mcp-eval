"""Enhanced generator that can leverage subagents for better test generation."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from importlib import resources

async def load_subagent_instruction(subagent_name: str) -> Optional[str]:
    """Load a subagent's instruction from package data.
    
    Args:
        subagent_name: Name of the subagent (without .md extension)
        
    Returns:
        The instruction portion of the subagent, or None if not found
    """
    try:
        subagents_path = resources.files("mcp_eval.data").joinpath("subagents")
        agent_file = subagents_path.joinpath(f"{subagent_name}.md")
        
        if not agent_file.exists():
            return None
            
        content = agent_file.read_text()
        
        # Parse the frontmatter and extract the instruction
        if content.startswith("---"):
            lines = content.split("\n")
            end_index = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end_index = i
                    break
            
            if end_index > 0:
                # Return everything after the frontmatter
                instruction = "\n".join(lines[end_index + 1:]).strip()
                return instruction
        
        return content.strip()
        
    except Exception as e:
        print(f"Warning: Could not load subagent {subagent_name}: {e}")
        return None


def get_scenario_designer_instruction() -> str:
    """Get the instruction for the scenario designer subagent."""
    instruction = load_subagent_instruction("test-scenario-designer")
    if instruction:
        return instruction
    
    # Fallback to a basic instruction
    return """You are an expert test scenario designer specializing in creating high-quality test scenarios for MCP servers.
    Design comprehensive scenarios covering functionality, error handling, edge cases, and performance."""


def get_assertion_refiner_instruction() -> str:
    """Get the instruction for the assertion refiner subagent."""
    instruction = load_subagent_instruction("test-assertion-refiner")
    if instruction:
        return instruction
    
    # Fallback to a basic instruction
    return """You are an expert at refining test assertions to create robust, comprehensive test coverage.
    Enhance existing scenarios by adding missing assertions and improving quality."""


def get_code_emitter_instruction() -> str:
    """Get the instruction for the code emitter subagent."""
    instruction = load_subagent_instruction("test-code-emitter")
    if instruction:
        return instruction
    
    # Fallback to a basic instruction
    return """You are an expert at emitting valid Python test code for MCP-Eval.
    Convert test scenarios into syntactically correct, runnable test files."""


async def generate_scenarios_with_subagents(
    tools: List[Any],
    *,
    n_examples: int = 8,
    provider: str = "anthropic",
    model: str | None = None,
    progress_callback: Optional[Any] = None,
    debug: bool = False,
    max_retries: int = 1,
    use_subagents: bool = True,
) -> List[Any]:
    """Enhanced scenario generation that can use specialized subagents.
    
    This function wraps the original generate_scenarios_with_agent but uses
    specialized subagent instructions for better quality generation.
    """
    from mcp_eval.generation import generate_scenarios_with_agent
    from mcp_eval.config import load_config
    from mcp_agent.app import MCPApp
    from mcp_agent.agents.agent import Agent
    from mcp_agent.config import LoggerSettings
    from mcp_agent.workflows.factory import _llm_factory
    
    if not use_subagents:
        # Fall back to the original implementation
        return await generate_scenarios_with_agent(
            tools=tools,
            n_examples=n_examples,
            provider=provider,
            model=model,
            progress_callback=progress_callback,
            debug=debug,
            max_retries=max_retries,
        )
    
    # Load settings
    settings = load_config()
    settings.logger = LoggerSettings(type="console", level="info" if debug else "error")
    
    app = MCPApp(settings=settings)
    async with app.run() as running:
        # Use the scenario designer subagent instruction
        scenario_agent = Agent(
            name="test_scenario_designer",
            instruction=await get_scenario_designer_instruction(),
            server_names=[],
            context=running.context,
        )
        
        # Generate scenarios with the enhanced agent
        scenarios = await generate_scenarios_with_agent(
            tools=tools,
            n_examples=n_examples,
            provider=provider,
            model=model,
            progress_callback=progress_callback,
            debug=debug,
            max_retries=max_retries,
        )
        
        return scenarios


async def refine_assertions_with_subagents(
    scenarios: List[Any],
    tools: List[Any],
    *,
    provider: str = "anthropic",
    model: str | None = None,
    progress_callback: Optional[Any] = None,
    debug: bool = False,
    use_subagents: bool = True,
) -> List[Any]:
    """Enhanced assertion refinement that can use specialized subagents.
    
    This function wraps the original refine_assertions_with_agent but uses
    specialized subagent instructions for better quality refinement.
    """
    from mcp_eval.generation import refine_assertions_with_agent
    from mcp_eval.config import load_config
    from mcp_agent.app import MCPApp
    from mcp_agent.agents.agent import Agent
    from mcp_agent.config import LoggerSettings
    
    if not use_subagents:
        # Fall back to the original implementation
        return await refine_assertions_with_agent(
            scenarios,
            tools,
            provider=provider,
            model=model,
            progress_callback=progress_callback,
            debug=debug,
        )
    
    # Load settings
    settings = load_config()
    settings.logger = LoggerSettings(type="console", level="info" if debug else "error")
    
    app = MCPApp(settings=settings)
    async with app.run() as running:
        # Use the assertion refiner subagent instruction
        refiner_agent = Agent(
            name="test_assertion_refiner",
            instruction=await get_assertion_refiner_instruction(),
            server_names=[],
            context=running.context,
        )
        
        # Refine assertions with the enhanced agent
        refined = await refine_assertions_with_agent(
            scenarios,
            tools,
            provider=provider,
            model=model,
            progress_callback=progress_callback,
            debug=debug,
        )
        
        return refined