"""MCP‑Eval Test Case Generator CLI.

Interactive flow to:
- capture provider + API key (writes mcpeval.yaml + mcpeval.secrets.yaml)
- capture/construct MCP server settings using typed models (MCPServerSettings)
- connect to the server and list tools
- generate structured scenarios + assertion specs using an mcp‑agent Agent
- emit tests (pytest/decorators) or a dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

from mcp_agent.app import MCPApp
from mcp_agent.mcp.gen_client import gen_client
from mcp_agent.config import MCPServerSettings
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.factory import _llm_factory
from mcp_agent.workflows.llm.llm_selector import ModelSelector, ModelPreferences
from mcp.types import Tool as MCPTool

from mcp_eval.generation import (
    generate_scenarios_with_agent,
    refine_assertions_with_agent,
    render_pytest_tests,
    render_decorator_tests,
    dataset_from_scenarios,
    ToolSchema,
)
from mcp_eval.cli.models import (
    ServerImport,
    ConfigPaths,
    MCPServerConfig,
    AgentConfig,
    MCPEvalConfig,
    GeneratorState,
)
from mcp_eval.cli.utils import (
    load_yaml,
    save_yaml,
    deep_merge,
    find_mcp_json,
    find_config_files,
    import_servers_from_json,
    import_servers_from_dxt,
    load_all_servers,
    load_all_agents,
    ensure_mcpeval_yaml,
    write_server_to_mcpeval,
    write_agent_to_mcpeval,
)

app = typer.Typer(help="Generate MCP‑Eval tests for an MCP server")
console = Console()


# --------------- helpers -----------------


def _set_default_agent(project: Path, agent_name: str) -> None:
    """Set the default agent in mcpeval.yaml."""
    cfg_path = ensure_mcpeval_yaml(project)
    cfg = load_yaml(cfg_path)
    cfg["default_agent"] = agent_name
    save_yaml(cfg_path, cfg)
    console.print(f"[green]✓[/] Set default_agent='{agent_name}' in {cfg_path}")


def _sanitize_filename_component(value: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", value.strip())
    if not s:
        s = "generated"
    # Avoid dotfiles or leading hyphens
    if s[0] in (".", "-"):
        s = f"x{s}"
    return s


def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    # Try timestamp first
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"{stem}_{ts}{suffix}"
    if not candidate.exists():
        return candidate
    # Fallback to incrementing counter
    for i in range(1, 200):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    # Last resort: random-ish hash from timestamp
    candidate = (
        parent / f"{stem}_{int(datetime.now().timestamp() * 1000) % 100000}{suffix}"
    )
    return candidate


def _sanitize_slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", value.lower())
    s = s.strip("-")
    return s or "gen"


def _build_llm(agent: Agent, provider: str, model: Optional[str]) -> Any:
    factory = _llm_factory(provider=provider, model=model, context=agent.context)
    return factory(agent)


async def _generate_llm_slug(
    server_name: str, provider: str, model: Optional[str]
) -> Optional[str]:
    try:
        mcp_app = MCPApp()
        async with mcp_app.run() as running:
            agent = Agent(
                name="filename_disambiguator",
                instruction="You output short, friendly slugs for filenames.",
                server_names=[],
                context=running.context,
            )
            llm = _build_llm(agent, provider, model)
            prompt = (
                "Generate a very short, lowercase, hyphenated slug (max 8 chars) to disambiguate a filename for server '"
                + server_name
                + "'. Only return the slug, no quotes, no punctuation except hyphens."
            )
            raw = await llm.generate_str(prompt)
            if not isinstance(raw, str):
                raw = str(raw)
            slug = _sanitize_slug(raw.split()[0])
            return slug[:8]
    except Exception:
        return None


def _convert_servers_to_mcp_settings(servers: Dict[str, MCPServerConfig]) -> Dict[str, MCPServerSettings]:
    """Convert MCPServerConfig to MCPServerSettings."""
    result: Dict[str, MCPServerSettings] = {}
    for name, server in servers.items():
        result[name] = MCPServerSettings(**server.to_mcp_agent_settings())
    return result


def _load_existing_provider(project: Path) -> tuple[Optional[str], Optional[str]]:
    """Load existing provider configuration from secrets."""
    sec_path = project / "mcpeval.secrets.yaml"
    sec = load_yaml(sec_path)
    # Also check .mcp-eval/secrets.yaml
    if not sec:
        sec = load_yaml(project / ".mcp-eval" / "secrets.yaml")
    if not sec:
        return None, None
    if "anthropic" in sec and isinstance(sec["anthropic"], dict):
        key = sec["anthropic"].get("api_key")
        return ("anthropic", key) if key else ("anthropic", None)
    if "openai" in sec and isinstance(sec["openai"], dict):
        key = sec["openai"].get("api_key")
        return ("openai", key) if key else ("openai", None)
    return None, None


def _prompt_provider(
    existing_provider: Optional[str], existing_key: Optional[str]
) -> tuple[str, Optional[str], Optional[str]]:
    """Prompt for provider, API key, and optional model."""
    if existing_provider:
        console.print(f"[cyan]Using existing provider: {existing_provider}[/cyan]")
        if existing_key:
            console.print("[green]API key already configured[/green]")
            # Optionally ask for model override
            model = Prompt.ask("Model (press Enter to auto-select)", default="").strip() or None
            return existing_provider, existing_key, model
        # Ask only for missing key
        api_key = Prompt.ask(f"Enter {existing_provider} API key", password=True)
        model = Prompt.ask("Model (press Enter to auto-select)", default="").strip() or None
        return existing_provider, api_key, model
    # No existing provider; prompt fresh
    provider = Prompt.ask(
        "LLM provider", 
        choices=["anthropic", "openai"],
        default="anthropic"
    ).strip().lower()
    api_key = Prompt.ask(f"Enter {provider} API key", password=True)
    model = Prompt.ask("Model (press Enter to auto-select)", default="").strip() or None
    return provider, api_key, model


def _write_mcpeval_configs(project: Path, provider: str, api_key: str, model: Optional[str] = None) -> None:
    """Write provider configuration to mcpeval.yaml and secrets."""
    cfg_path = ensure_mcpeval_yaml(project)
    sec_path = project / "mcpeval.secrets.yaml"

    cfg = load_yaml(cfg_path)
    sec = load_yaml(sec_path)

    # Use ModelSelector to pick the best model for the provider if not specified
    if not model:
        try:
            selector = ModelSelector()
            # For judge, prioritize intelligence and cost-effectiveness
            preferences = ModelPreferences(
                costPriority=0.4,
                speedPriority=0.2,
                intelligencePriority=0.4
            )
            model_info = selector.select_best_model(
                model_preferences=preferences,
                provider=provider
            )
            judge_model = model_info.name
        except Exception as e:
            # Fallback if ModelSelector fails
            console.print(f"[yellow]Warning: Could not select model automatically: {e}[/yellow]")
            judge_model = "claude-sonnet-4-0" if provider == "anthropic" else "gpt-4o"
    else:
        judge_model = model
    
    cfg_overlay = {
        "judge": {"model": judge_model, "min_score": 0.8},
        "reporting": {"formats": ["json", "markdown"], "output_dir": "./test-reports"},
    }
    if provider == "anthropic":
        sec_overlay = {"anthropic": {"api_key": api_key}}
    else:
        sec_overlay = {"openai": {"api_key": api_key}}

    save_yaml(cfg_path, deep_merge(cfg, cfg_overlay))
    save_yaml(sec_path, deep_merge(sec, sec_overlay))
    console.print(f"[green]✓[/] Wrote {cfg_path} and {sec_path}")


# This function is replaced by load_all_servers from utils


def _prompt_server_settings(
    imported: Dict[str, MCPServerConfig],
) -> tuple[str, MCPServerConfig]:
    """Prompt user for server settings."""
    if imported:
        console.print("[cyan]Available servers:[/cyan]")
        for n in imported.keys():
            console.print(f" - {n}")

    server_name = Prompt.ask("Server name (e.g., fetch)")
    if server_name in imported:
        return server_name, imported[server_name]

    transport = Prompt.ask(
        "Transport",
        choices=["stdio", "sse", "streamable_http", "websocket"],
        default="stdio"
    ).strip()
    
    server = MCPServerConfig(name=server_name, transport=transport)
    
    if transport == "stdio":
        server.command = Prompt.ask("Command (e.g., npx, uvx, python)")
        args = Prompt.ask(
            "Args (space-separated, blank for none)", default=""
        ).strip()
        server.args = [a for a in args.split(" ") if a]
        
        if Confirm.ask("Add environment variables?", default=False):
            kv = Prompt.ask("KEY=VALUE pairs (comma-separated)", default="").strip()
            if kv:
                env: Dict[str, str] = {}
                for pair in kv.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        env[k.strip()] = v.strip()
                server.env = env
    else:
        server.url = Prompt.ask("Server URL")
        if Confirm.ask("Add HTTP headers?", default=False):
            kv = Prompt.ask(
                "Header:Value pairs (comma-separated)", default=""
            ).strip()
            if kv:
                headers: Dict[str, str] = {}
                for pair in kv.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        headers[k.strip()] = v.strip()
                if headers:
                    server.headers = headers

    return server_name, server


async def _discover_tools(server_name: str) -> List[ToolSchema]:
    """Connect to the server and return typed tool specs."""
    tools: List[ToolSchema] = []
    try:
        mcp_app = MCPApp()
        async with mcp_app.run() as running:
            async with gen_client(
                server_name, server_registry=running.context.server_registry
            ) as client:
                result = await client.list_tools()
                # Prefer typed access; fall back to dict if needed
                items: List[MCPTool] = []
                if hasattr(result, "tools") and isinstance(result.tools, list):
                    items = result.tools  # type: ignore[assignment]
                    for t in items:
                        name: str = getattr(t, "name", None) or getattr(t, "tool", None) or getattr(t, "id", None) or ""
                        if not name:
                            continue
                        description: Optional[str] = getattr(t, "description", None)
                        input_schema: Optional[Dict[str, Any]] = (
                            getattr(t, "inputSchema", None)
                            or getattr(t, "input_schema", None)
                            or getattr(t, "input", None)
                        )
                        tools.append(
                            ToolSchema(name=name, description=description, input_schema=input_schema)
                        )
                else:
                    try:
                        data = result.model_dump()
                    except Exception:
                        data = getattr(result, "dict", lambda: {})()
                    raw_items = data.get("tools") or data.get("items") or []
                    for t in raw_items:
                        name = t.get("name") or t.get("tool") or t.get("id")
                        if not name:
                            continue
                        tools.append(
                            ToolSchema(
                                name=name,
                                description=t.get("description"),
                                input_schema=t.get("inputSchema") or t.get("input_schema") or t.get("input") or None,
                            )
                        )
    except Exception as e:
        console.print(f"[red]Error discovering tools for '{server_name}': {e}[/red]")
    return tools


# These functions are replaced by write_server_to_mcpeval from utils


def _emit_tests(
    project: Path,
    style: str,
    server_name: str,
    scenarios: List[Any],
    provider: str,
    model: Optional[str] = None,
) -> None:
    style = style.strip().lower()
    safe_server = _sanitize_filename_component(server_name)
    if style == "dataset":
        ds = dataset_from_scenarios(scenarios, server_name)
        ds_path = project / "datasets"
        ds_path.mkdir(parents=True, exist_ok=True)
        base_file = ds_path / f"{safe_server}_generated.yaml"
        out_file = base_file
        if out_file.exists():
            try:
                import asyncio

                slug = asyncio.run(_generate_llm_slug(server_name, provider, model))
            except Exception:
                slug = None
            if slug:
                out_file = ds_path / f"{safe_server}_{slug}.yaml"
            out_file = _unique_path(out_file)
        # Dump via Dataset.to_file for correct structure
        try:
            ds.to_file(out_file)
        except Exception:
            # Fallback to raw yaml if needed
            raw = {
                "name": ds.name,
                "server_name": server_name,
                "cases": [
                    {
                        "name": c.name,
                        "inputs": c.inputs,
                        "expected_output": c.expected_output,
                    }
                    for c in ds.cases
                ],
            }
            _save_yaml(out_file, raw)
        console.print(f"[green]✓[/] Wrote dataset {out_file}")
        return

    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    base_path = tests_dir / f"test_{safe_server}_generated.py"
    out_path = base_path
    if out_path.exists():
        try:
            import asyncio

            slug = asyncio.run(_generate_llm_slug(server_name, provider, model))
        except Exception:
            slug = None
        if slug:
            out_path = tests_dir / f"test_{safe_server}_{slug}.py"
        out_path = _unique_path(out_path)
    if style == "decorators":
        content = render_decorator_tests(scenarios, server_name)
    else:
        content = render_pytest_tests(scenarios, server_name)
    out_path.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/] Wrote tests {out_path}")


# --------------- main command -----------------


@app.command("init")
def init_project(
    out_dir: str = typer.Option(".", help="Project directory for configs"),
):
    """Initialize an mcp-eval project.

    - Ensure mcpeval.yaml and mcpeval.secrets.yaml exist (write minimal defaults)
    - Configure provider + API key
    - Import servers from mcp-agent.config.yaml and optionally mcp.json (cursor/vscode)
    - Define/select a default AgentSpec and set default_agent

    Examples:
      - Initialize project:
        mcp-eval init
    """
    project = Path(out_dir)
    project.mkdir(parents=True, exist_ok=True)
    ensure_mcpeval_yaml(project)

    # Provider + API key
    console.print("[cyan]Configuring LLM provider and secrets...[/cyan]")
    existing_provider, existing_key = _load_existing_provider(project)
    provider, api_key, model = _prompt_provider(existing_provider, existing_key)
    if api_key:
        _write_mcpeval_configs(project, provider, api_key, model)
    else:
        console.print("[green]Using existing secrets[/green]")

    # Servers
    console.print("[cyan]Discovering servers...[/cyan]")
    existing_servers = load_all_servers(project)
    imported_servers: Dict[str, MCPServerConfig] = {}
    
    # Check for mcp.json files and prompt user
    mcp_json = find_mcp_json()
    if mcp_json:
        console.print(f"[cyan]Found MCP configuration at {mcp_json}[/cyan]")
        if Confirm.ask("Would you like to import servers from this file?", default=True):
            console.print(f"[cyan]Importing servers from {mcp_json}...[/cyan]")
            import_result = import_servers_from_json(mcp_json)
            if import_result.success:
                for name, cfg in import_result.servers.items():
                    imported_servers[name] = MCPServerConfig(**cfg)
                console.print(f"[green]Imported {len(imported_servers)} servers[/green]")
            else:
                console.print(f"[yellow]Failed to import: {import_result.error}[/yellow]")
    
    merged_servers: Dict[str, MCPServerConfig] = {**existing_servers, **imported_servers}
    
    if merged_servers:
        console.print("[cyan]Available servers:[/cyan]")
        for n in sorted(merged_servers.keys()):
            console.print(f" - {n}")
    else:
        console.print("[yellow]No servers found; let's add one[/yellow]")
    
    server_name, server_config = _prompt_server_settings(merged_servers)
    write_server_to_mcpeval(project, server_config)

    # Default AgentSpec
    console.print("[cyan]Define default agent (will be stored in mcpeval.yaml)[/cyan]")
    agent_name = Prompt.ask("Agent name", default="default")
    instruction = Prompt.ask(
        "Agent instruction",
        default="You are a helpful assistant that can use MCP servers effectively.",
    )
    # Allow multiple servers
    server_list_str = Prompt.ask(
        "Server names for this agent (comma-separated)", default=server_name
    )
    server_list = [s.strip() for s in server_list_str.split(",") if s.strip()]
    
    agent = AgentConfig(
        name=agent_name,
        instruction=instruction,
        server_names=server_list,
        provider=provider,
        model=model,
    )
    write_agent_to_mcpeval(project, agent, set_default=True)
    console.print("[bold green]✓ Project initialized[/bold green]")


@app.command("generate")
def run_generator(
    out_dir: str = typer.Option(".", help="Project directory to write configs/tests"),
    style: Optional[str] = typer.Option(None, help="Test style: pytest|decorators|dataset"),
    n_examples: int = typer.Option(6, help="Number of scenarios to generate"),
    provider: str = typer.Option("anthropic", help="LLM provider (anthropic|openai)"),
    model: Optional[str] = typer.Option(None, help="Model id (optional)"),
):
    """Generate scenarios and write a single test file.

    Examples:
      - Quick start (prompt for style):
        mcp-eval generate

      - Explicit pytest style and 10 scenarios:
        mcp-eval generate --style pytest --n-examples 10
    """
    project = Path(out_dir)
    _ensure_dir(project)

    console.print("[cyan]Checking credentials and writing mcpeval configs if needed...[/cyan]")
    # Provider + API key (load existing when re-running)
    existing_provider, existing_key = _load_existing_provider(project)
    # Get provider, api_key, and optional model from prompt
    provider, api_key, prompted_model = _prompt_provider(existing_provider, existing_key)
    # Use CLI model if provided, otherwise use prompted model
    final_model = model or prompted_model
    if api_key:
        _write_mcpeval_configs(project, provider, api_key, final_model)
    else:
        console.print("[green]Using existing secrets[/green]")

    # Server capture (only from configs; mcp.json import happens in init)
    existing_servers = load_all_servers(project)
    if existing_servers:
        console.print("[cyan]Available servers:[/cyan]")
        for n in sorted(existing_servers.keys()):
            console.print(f" - {n}")
        console.print(
            "Type an existing server name to use it, or type a new name to add one."
        )
        chosen = Prompt.ask("Server name", default=next(iter(sorted(existing_servers.keys()))))
        if chosen in existing_servers:
            server_name = chosen
            server_config = existing_servers[chosen]
        else:
            # Add new server via prompt
            server_name, server_config = _prompt_server_settings({})
    else:
        # No servers known yet; prompt fresh
        console.print("[yellow]No servers configured yet[/yellow]")
        server_name, server_config = _prompt_server_settings({})
    
    # Persist to mcpeval.yaml (source of truth for this tool)
    write_server_to_mcpeval(project, server_config)
    console.print(f"[cyan]Server '{server_name}' configured.[/cyan]")

    # Agent selection by name from mcpeval.yaml agents.definitions
    agents = load_all_agents(project)
    cfg_path = ensure_mcpeval_yaml(project)
    cfg = load_yaml(cfg_path)
    default_agent = cfg.get("default_agent")
    
    if agents:
        console.print("[cyan]Available agents:[/cyan]")
        for agent in agents:
            marker = "(default)" if agent.name == default_agent else ""
            console.print(f" - {agent.name} {marker}")
        
        agent_names = [a.name for a in agents]
        chosen_agent = Prompt.ask(
            "Agent to use", 
            choices=agent_names,
            default=default_agent or agent_names[0]
        )
        _set_default_agent(project, chosen_agent)
    else:
        console.print("[yellow]No agents defined. Consider running 'mcp-eval init' first to define a default agent.[/yellow]")

    # Discovery
    console.print(f"[cyan]Discovering tools for server '{server_name}'...[/cyan]")
    try:
        import asyncio
        tools = asyncio.run(_discover_tools(server_name))
    except Exception as e:
        console.print(f"[yellow]Warning: Could not list tools: {e}[/yellow]")
        tools = []
        if not Confirm.ask("Continue without tool discovery?", default=True):
            raise typer.Exit(1)
    
    if tools:
        console.print(
            f"[green]Discovered {len(tools)} tools:[/green] "
            + ", ".join([t.name for t in tools[:10]])
            + (" ..." if len(tools) > 10 else "")
        )
    else:
        console.print("[yellow]No tools discovered[/yellow]")

    # Two-stage generation (first: scenarios; second: assertions refinement per scenario)
    console.print(f"[cyan]Generating test scenarios...[/cyan]")
    try:
        import asyncio
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Generating scenarios...", total=None)
            scenarios = asyncio.run(
                generate_scenarios_with_agent(
                    tools=tools, n_examples=n_examples, provider=provider, model=final_model
                )
            )
            progress.update(task, description=f"Generated {len(scenarios)} scenarios")
            
            progress.update(task, description="Refining assertions...")
            scenarios = asyncio.run(
                refine_assertions_with_agent(
                    scenarios, tools, provider=provider, model=final_model
                )
            )
            progress.update(task, description="Completed generation")
        
        console.print(f"[green]✓ Generated {len(scenarios)} test scenarios[/green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to generate scenarios: {e}[/red]")
        raise typer.Exit(1)

    # Prompt for style if not provided
    if not style:
        style = typer.prompt("Test style (pytest|decorators|dataset)", default="pytest").strip().lower()

    _emit_tests(project, style, server_name, scenarios, provider=provider, model=final_model)
    # Summary of generated scenarios
    if scenarios:
        console.print("\n[bold]Summary of generated scenarios:[/bold]")
        for s in scenarios[:20]:  # cap for display
            console.print(f" - [green]{s.name}[/green]: {s.description or ''}")
        if len(scenarios) > 20:
            console.print(f" ... and {len(scenarios) - 20} more")


@app.command("update")
def update_tests(
    out_dir: str = typer.Option(".", help="Project directory"),
    target_file: str = typer.Option(..., help="Path to an existing test file to append to"),
    server_name: str = typer.Option(None, help="Server to generate against (prompted if omitted)"),
    style: str = typer.Option("pytest", help="Test style for new tests: pytest|decorators|dataset"),
    n_examples: int = typer.Option(4, help="Number of new scenarios to generate"),
    provider: str = typer.Option("anthropic", help="LLM provider (anthropic|openai)"),
    model: Optional[str] = typer.Option(None, help="Model id (optional)"),
):
    """Append newly generated tests to an existing test file (non-interactive).

    Examples:
      - Append 4 pytest-style tests to a file:
        mcp-eval update --target-file tests/test_fetch_generated.py --style pytest --n-examples 4
    """
    project = Path(out_dir)
    file_path = Path(target_file)
    if not file_path.exists():
        console.print(f"[red]Target file not found:[/] {file_path}")
        raise typer.Exit(1)

    console.print("[cyan]Preparing to append tests...[/cyan]")
    existing_servers = load_all_servers(project)
    if not server_name:
        if not existing_servers:
            console.print("[yellow]No servers configured. Run 'mcp-eval init' first.[/yellow]")
            raise typer.Exit(1)
        console.print("Available servers:")
        for n in sorted(existing_servers.keys()):
            console.print(f" - {n}")
        server_name = typer.prompt("Server name", default=next(iter(sorted(existing_servers.keys()))))

    # Provider + key (reuse flow)
    existing_provider, existing_key = _load_existing_provider(project)
    provider, api_key, prompted_model = _prompt_provider(existing_provider, existing_key)
    final_model = model or prompted_model
    if api_key:
        _write_mcpeval_configs(project, provider, api_key, final_model)

    console.print(f"[cyan]Listing tools for '{server_name}'...[/cyan]")
    try:
        import asyncio

        tools = asyncio.run(_discover_tools(server_name))
    except Exception as e:
        console.print(f"[red]Failed to list tools:[/] {e}")
        raise typer.Exit(1)
    console.print(f"Discovered {len(tools)} tools")

    # Generate and refine new scenarios
    try:
        scenarios = asyncio.run(
            generate_scenarios_with_agent(
                tools=tools, n_examples=n_examples, provider=provider, model=final_model
            )
        )
        scenarios = asyncio.run(
            refine_assertions_with_agent(
                scenarios, tools, provider=provider, model=final_model
            )
        )
    except Exception as e:
        console.print(f"[red]Failed to generate scenarios:[/] {e}")
        raise typer.Exit(1)

    # Render content and append
    if style.strip().lower() == "decorators":
        content = render_decorator_tests(scenarios, server_name)
    elif style.strip().lower() == "dataset":
        content = render_pytest_tests(scenarios, server_name)  # dataset append not ideal
    else:
        content = render_pytest_tests(scenarios, server_name)

    sep = "\n\n# ---- mcp-eval: additional generated tests ----\n\n"
    appended = sep + content
    file_path.write_text(file_path.read_text(encoding="utf-8") + appended, encoding="utf-8")
    console.print(f"[green]✓[/] Appended {len(scenarios)} tests to {file_path}")


add_app = typer.Typer(help="Add resources to mcpeval.yaml (servers, agents).\n\nExamples:\n  - Add a server interactively:\n    mcp-eval add server\n\n  - Import servers from mcp.json:\n    mcp-eval add server --from-mcp-json .cursor/mcp.json\n\n  - Add an agent:\n    mcp-eval add agent")
app.add_typer(add_app, name="add")


@add_app.command("server")
def add_server(
    out_dir: str = typer.Option(".", help="Project directory"),
    from_mcp_json: Optional[str] = typer.Option(None, help="Path to mcp.json to import servers from"),
    from_dxt: Optional[str] = typer.Option(None, help="Path to DXT file to import servers from"),
):
    """Add a server to mcpeval.yaml, either interactively or from mcp.json/DXT file.

    Examples:
      - Interactive add:
        mcp-eval add server

      - From mcp.json:
        mcp-eval add server --from-mcp-json .cursor/mcp.json
    """
    project = Path(out_dir)
    project.mkdir(parents=True, exist_ok=True)

    imported_servers: Dict[str, MCPServerConfig] = {}
    
    if from_mcp_json:
        json_path = Path(from_mcp_json)
        if not json_path.exists():
            console.print(f"[red]Error: File not found: {json_path}[/red]")
            raise typer.Exit(1)
        
        import_result = import_servers_from_json(json_path)
        if import_result.success:
            for name, cfg in import_result.servers.items():
                imported_servers[name] = MCPServerConfig(**cfg)
            console.print(f"[green]Found {len(imported_servers)} servers in {json_path}[/green]")
        else:
            console.print(f"[red]Failed to import: {import_result.error}[/red]")
            raise typer.Exit(1)
            
    elif from_dxt:
        dxt_path = Path(from_dxt)
        if not dxt_path.exists():
            console.print(f"[red]Error: File not found: {dxt_path}[/red]")
            raise typer.Exit(1)
            
        import_result = import_servers_from_dxt(dxt_path)
        if import_result.success:
            for name, cfg in import_result.servers.items():
                imported_servers[name] = MCPServerConfig(**cfg)
            console.print(f"[green]Found {len(imported_servers)} servers in {dxt_path}[/green]")
        else:
            console.print(f"[yellow]No servers found in DXT file: {import_result.error}[/yellow]")

    if imported_servers:
        console.print("[cyan]Imported servers:[/cyan]")
        for n in imported_servers.keys():
            console.print(f" - {n}")
        
        server_names = list(imported_servers.keys())
        chosen = Prompt.ask("Server to add", choices=server_names, default=server_names[0])
        
        if chosen in imported_servers:
            write_server_to_mcpeval(project, imported_servers[chosen])
            console.print(f"[green]✓ Added server '{chosen}'[/green]")
            return

    # Interactive add
    server_name, server_config = _prompt_server_settings({})
    write_server_to_mcpeval(project, server_config)
    console.print(f"[green]✓ Added server '{server_name}'[/green]")


@add_app.command("agent")
def add_agent(
    out_dir: str = typer.Option(".", help="Project directory"),
):
    """Add an AgentSpec to mcpeval.yaml (validates referenced servers exist).

    Examples:
      - Add an agent and set as default:
        mcp-eval add agent
    """
    project = Path(out_dir)
    project.mkdir(parents=True, exist_ok=True)
    ensure_mcpeval_yaml(project)

    # Check existing agents to avoid duplicates
    existing_agents = load_all_agents(project)
    existing_names = [a.name for a in existing_agents]
    
    # Gather AgentSpec fields
    name = Prompt.ask("Agent name")
    if name in existing_names:
        if not Confirm.ask(f"Agent '{name}' already exists. Replace it?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    instruction = Prompt.ask(
        "Instruction", 
        default="You are a helpful assistant that can use MCP servers effectively."
    )
    
    # Show available servers
    existing_servers = load_all_servers(project)
    if existing_servers:
        console.print("[cyan]Available servers:[/cyan]")
        for server_name in existing_servers.keys():
            console.print(f" - {server_name}")
    
    server_list_str = Prompt.ask("Server names (comma-separated)")
    server_names = [s.strip() for s in server_list_str.split(",") if s.strip()]

    # Validate servers exist
    missing = [s for s in server_names if s not in existing_servers]
    if missing:
        console.print(f"[yellow]Warning: Referenced servers not found: {', '.join(missing)}[/yellow]")
        if Confirm.ask("Would you like to add them now?", default=True):
            for s in missing:
                console.print(f"\n[cyan]Adding server '{s}'...[/cyan]")
                # Pre-fill the name
                console.print(f"Server name: {s}")
                _, server_config = _prompt_server_settings({})
                server_config.name = s  # Ensure name matches
                write_server_to_mcpeval(project, server_config)

    # Create and save agent
    agent = AgentConfig(
        name=name,
        instruction=instruction,
        server_names=server_names
    )
    
    set_default = Confirm.ask("Set this as default agent?", default=len(existing_agents) == 0)
    write_agent_to_mcpeval(project, agent, set_default=set_default)
    console.print(f"[green]✓ Added agent '{name}'[/green]")
