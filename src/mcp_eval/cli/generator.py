"""MCP‑Eval Test Case Generator CLI.

Interactive flow to:
- capture provider + API key (writes mcpeval.yaml + mcpeval.secrets.yaml)
- capture/construct MCP server settings using typed models (MCPServerSettings)
- connect to the server and list tools
- generate structured scenarios + assertion specs using an mcp‑agent Agent
- emit tests (pytest/decorators) or a dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import re
import subprocess
import asyncio
from datetime import datetime

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

from mcp_agent.app import MCPApp
from mcp_agent.mcp.gen_client import gen_client
from mcp_agent.config import MCPServerSettings, LoggerSettings, MCPSettings
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.factory import _llm_factory
from mcp_agent.workflows.llm.llm_selector import ModelSelector, ModelPreferences
from mcp_agent.core.context import Context
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
    MCPServerConfig,
    AgentConfig,
)
from mcp_eval.config import load_config, MCPEvalSettings, find_eval_config
from mcp_eval.cli.utils import (
    load_yaml,
    save_yaml,
    deep_merge,
    find_mcp_json,
    import_servers_from_json,
    import_servers_from_dxt,
    load_all_servers,
    load_all_agents,
    ensure_mcpeval_yaml,
    write_server_to_mcpeval,
    write_agent_to_mcpeval,
)
from mcp_eval.data.utils import copy_sample_template
from mcp_eval.cli.list_command import (
    list_servers as _list_servers_cmd,
    list_agents as _list_agents_cmd,
)

app = typer.Typer(help="Generate MCP‑Eval tests for an MCP server")
console = Console()


# --------------- helpers -----------------


async def _validate_and_fix_test_file(
    test_file: Path, provider: str, model: str | None, max_attempts: int = 3
) -> bool:
    """Validate Python test file and fix compile errors.

    Returns True if file is valid, False if unfixable after max attempts.
    """

    for attempt in range(1, max_attempts + 1):
        # Try to compile the Python file using uv
        result = subprocess.run(
            ["uv", "run", "python", "-m", "py_compile", str(test_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✓ Test file syntax is valid[/green]")
            return True

        # Parse error
        error_msg = result.stderr
        console.print(
            f"\n[yellow]Syntax error found (attempt {attempt}/{max_attempts}):[/yellow]"
        )
        console.print(f"[red]{error_msg}[/red]")

        if attempt >= max_attempts:
            console.print(
                "[red]Max fix attempts reached. Manual intervention required.[/red]"
            )
            return False

        # First pass: sanitize common JSON tokens to Python to self-heal trivial issues
        try:
            text = test_file.read_text(encoding="utf-8")
            sanitized = (
                text.replace("\nnull", "\nNone")
                .replace(": null", ": None")
                .replace("= null", "= None")
                .replace(" true,", " True,")
                .replace(" false,", " False,")
                .replace(" true)", " True)")
                .replace(" false)", " False)")
                .replace(" true\n", " True\n")
                .replace(" false\n", " False\n")
            )
            if sanitized != text:
                test_file.write_text(sanitized, encoding="utf-8")
                console.print(
                    "[cyan]Applied basic JSON->Python literal sanitization[/cyan]"
                )
                continue  # re-validate on next loop
        except Exception:
            pass

        # Use an agent with filesystem access to fix the remaining error
        console.print(
            "[cyan]Attempting to fix the error with filesystem agent...[/cyan]"
        )

        # Load settings and configure filesystem server
        from mcp_eval.config import load_config

        settings = load_config()
        settings.logger = LoggerSettings(
            type="none", level="error", progress_display=False
        )

        # Configure filesystem server with access to the test directory
        test_dir = test_file.parent
        settings.mcp = MCPSettings(
            servers={
                "filesystem": MCPServerSettings(
                    name="filesystem",
                    description="File system access for fixing test files",
                    command="npx",
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(test_dir),
                    ],
                )
            }
        )

        mcp_app = MCPApp(settings=settings)
        async with mcp_app.run() as running:
            agent = Agent(
                name="test_fixer",
                instruction="""You are a Python test file fixer. You have filesystem access to read and write files.
                When you encounter a Python syntax error, you should:
                1. Read the file with the error
                2. Fix the syntax error
                3. Write the corrected content back to the same file
                Be careful to preserve all the test logic and only fix syntax issues.""",
                server_names=["filesystem"],
                context=running.context,
            )

            llm = _build_llm(agent, provider, model)

            # Prepare the prompt with file path relative to the test directory
            relative_path = test_file.name
            prompt = f"""Fix the Python syntax error in the file '{relative_path}'.

The file has this syntax error:
{error_msg}

Please:
1. Read the file '{relative_path}' 
2. Fix the syntax error
3. Write the corrected content back to '{relative_path}'

Use the filesystem tools to read and write the file. Only fix the syntax error, preserve all other content and logic.
 
 STRICT REQUIREMENTS:
 - Replace JSON literals with Python equivalents (true->True, false->False, null->None)
 - Ensure dicts/lists are valid Python (use repr/quotes correctly)
"""

            try:
                # Let the agent fix the file using filesystem tools
                response = await llm.generate_str(prompt)
                console.print(
                    f"[dim]Agent response: {response[:200]}...[/dim]"
                    if len(response) > 200
                    else f"[dim]Agent response: {response}[/dim]"
                )
                console.print("[green]Fix attempted, re-validating...[/green]")
            except Exception as e:
                console.print(f"[red]Failed to fix: {e}[/red]")
                return False

    return False


def _display_run_command(style: str, output_file: Path | None) -> None:
    """Display the command to run the generated tests."""
    if not output_file:
        return

    console.print("\n[bold green]To run the generated tests:[/bold green]")

    if style == "pytest":
        console.print(f"  [cyan]pytest {output_file}[/cyan]")
        console.print("\n  Or with verbose output:")
        console.print(f"  [cyan]pytest -v {output_file}[/cyan]")
    elif style == "decorators":
        console.print(f"  [cyan]mcp-eval run decorators --file {output_file}[/cyan]")
    elif style == "dataset":
        console.print(f"  [cyan]mcp-eval run dataset --file {output_file}[/cyan]")
    else:
        console.print(f"  [cyan]mcp-eval run --file {output_file}[/cyan]")


def _parse_command_string(cmd: str) -> tuple[str, List[str]]:
    """Parse a command string into command and args.

    Examples:
        "uvx mcp-server-fetch" -> ("uvx", ["mcp-server-fetch"])
        "npx -y @modelcontextprotocol/server-filesystem /path" -> ("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/path"])
        "python -m server" -> ("python", ["-m", "server"])
    """
    import shlex

    parts = shlex.split(cmd.strip())
    if not parts:
        return "", []
    return parts[0], parts[1:] if len(parts) > 1 else []


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


def _build_llm(agent: Agent, provider: str, model: str | None) -> Any:
    factory = _llm_factory(provider=provider, model=model, context=agent.context)
    return factory(agent)


async def _generate_llm_slug(
    server_name: str, provider: str, model: str | None
) -> str | None:
    try:
        # Load settings from config (includes secrets)
        settings = load_config()

        # Set logger to errors only to reduce noise
        settings.logger = LoggerSettings(type="console", level="error")

        mcp_app = MCPApp(settings=settings)
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


def _convert_servers_to_mcp_settings(
    servers: Dict[str, MCPServerConfig],
) -> Dict[str, MCPServerSettings]:
    """Convert MCPServerConfig to MCPServerSettings."""
    result: Dict[str, MCPServerSettings] = {}
    for name, server in servers.items():
        result[name] = MCPServerSettings(**server.to_mcp_agent_settings())
    return result


def _load_existing_provider() -> tuple[
    str | None, str | None, Dict[str, str], MCPEvalSettings
]:
    """Load existing provider configuration from environment and config files.

    Returns:
        (selected_provider, selected_key, available_providers, settings)
        where available_providers is a dict of provider_name -> api_key
    """
    available_providers = {}
    selected_provider = None
    selected_key = None

    # Load settings - this handles all merging of env, configs, and secrets
    settings = load_config()

    # Show which config file we're using
    config_path = find_eval_config()
    if config_path:
        console.print(f"[dim]Using config: {config_path}[/dim]")

    # Check all providers for API keys
    if settings.anthropic and settings.anthropic.api_key:
        available_providers["anthropic"] = settings.anthropic.api_key
        if not selected_provider:
            selected_provider = "anthropic"
            selected_key = settings.anthropic.api_key

    if settings.openai and settings.openai.api_key:
        available_providers["openai"] = settings.openai.api_key
        if not selected_provider:
            selected_provider = "openai"
            selected_key = settings.openai.api_key

    if settings.google and settings.google.api_key:
        available_providers["google"] = settings.google.api_key
        if not selected_provider:
            selected_provider = "google"
            selected_key = settings.google.api_key

    if settings.cohere and settings.cohere.api_key:
        available_providers["cohere"] = settings.cohere.api_key
        if not selected_provider:
            selected_provider = "cohere"
            selected_key = settings.cohere.api_key

    if settings.azure and settings.azure.api_key:
        available_providers["azure"] = settings.azure.api_key
        if not selected_provider:
            selected_provider = "azure"
            selected_key = settings.azure.api_key

    return selected_provider, selected_key, available_providers, settings


def _prompt_provider(
    existing_provider: str | None,
    existing_key: str | None,
    available_providers: Dict[str, str] | None = None,
) -> tuple[str, str | None, str | None]:
    """Prompt for provider, API key, and optional model."""
    # If we have available providers from environment, show them to user
    if available_providers:
        console.print("\n[bold]🔍 Detected API keys from environment:[/bold]")
        for provider_name in available_providers:
            key_preview = (
                available_providers[provider_name][:8] + "..."
                if len(available_providers[provider_name]) > 8
                else "***"
            )
            console.print(f"  • {provider_name}: {key_preview}")

        if existing_provider and existing_key:
            use_existing = Confirm.ask(f"\nUse {existing_provider}?", default=True)
            if use_existing:
                model = (
                    Prompt.ask("Model (press Enter to auto-select)", default="").strip()
                    or None
                )
                return existing_provider, existing_key, model
            else:
                # Respect user's choice to not use the detected provider
                existing_provider = None
                existing_key = None

    if existing_provider:
        # If the user previously declined, existing_provider will be None and this block is skipped
        use_existing_provider = Confirm.ask(
            f"Use existing provider '{existing_provider}'?", default=True
        )
        if use_existing_provider:
            console.print(f"[cyan]Using existing provider: {existing_provider}[/cyan]")
            if existing_key:
                console.print("[green]API key already configured[/green]")
                # Optionally ask for model override
                model = (
                    Prompt.ask("Model (press Enter to auto-select)", default="").strip()
                    or None
                )
                return existing_provider, existing_key, model
            # Ask only for missing key
            api_key = Prompt.ask(f"Enter {existing_provider} API key", password=True)
            model = (
                Prompt.ask("Model (press Enter to auto-select)", default="").strip()
                or None
            )
            return existing_provider, api_key, model
        else:
            # User declined using existing; fall through to fresh selection
            existing_provider = None
            existing_key = None

    # No existing provider; prompt fresh
    # Build choices based on available providers
    choices = ["anthropic", "openai"]
    if available_providers:
        # Put available providers first
        available_choices = list(available_providers.keys())
        other_choices = [c for c in choices if c not in available_choices]
        choices = available_choices + other_choices

    provider = (
        Prompt.ask("LLM provider", choices=choices, default=choices[0]).strip().lower()
    )

    # If this provider has a key in environment, use it
    if available_providers and provider in available_providers:
        api_key = available_providers[provider]
        console.print(f"[green]Using API key from environment for {provider}[/green]")
    else:
        api_key = Prompt.ask(f"Enter {provider} API key", password=True)

    model = Prompt.ask("Model (press Enter to auto-select)", default="").strip() or None
    return provider, api_key, model


async def _write_mcpeval_configs(
    project: Path,
    settings: MCPEvalSettings,
    provider: str,
    api_key: str | None,
    model: str | None = None,
    context: Context | None = None,
) -> MCPEvalSettings:
    """Update settings and write provider configuration to mcpeval.yaml and secrets.

    Returns:
        Updated MCPEvalSettings object
    """
    cfg_path = ensure_mcpeval_yaml(project)
    sec_path = project / "mcpeval.secrets.yaml"

    cfg = load_yaml(cfg_path)
    sec = load_yaml(sec_path)

    # Use ModelSelector to pick the best model for the provider if not specified
    if not model:
        try:
            selector = ModelSelector(context=context)
            # For judge, prioritize intelligence and cost-effectiveness
            preferences = ModelPreferences(
                costPriority=0.4, speedPriority=0.2, intelligencePriority=0.4
            )
            model_info = selector.select_best_model(
                model_preferences=preferences, provider=provider
            )
            judge_model = model_info.name
            console.print(f"[dim]Selected model: {judge_model}[/dim]")
        except Exception as e:
            # Let ModelSelector error propagate if it fails
            console.print(f"[red]Error selecting model: {e}[/red]")
            raise
    else:
        judge_model = model
        console.print(f"[dim]Using specified model: {judge_model}[/dim]")

    # Update settings object
    settings.judge.model = judge_model
    settings.judge.provider = provider
    settings.judge.min_score = 0.8
    # Also set global provider/model so TestSession can attach an LLM
    settings.provider = provider
    settings.model = judge_model

    cfg_overlay = {
        "provider": provider,
        "model": judge_model,
        "judge": {"provider": provider, "model": judge_model, "min_score": 0.8},
        "reporting": {"formats": ["json", "markdown"], "output_dir": "./test-reports"},
    }

    save_yaml(cfg_path, deep_merge(cfg, cfg_overlay))

    # Only write/update secrets if an API key was provided
    if api_key:
        if provider == "anthropic":
            sec_overlay = {"anthropic": {"api_key": api_key}}
        elif provider == "openai":
            sec_overlay = {"openai": {"api_key": api_key}}
        else:
            # Fallback: write under the provider name
            sec_overlay = {provider: {"api_key": api_key}}
        save_yaml(sec_path, deep_merge(sec, sec_overlay))
        console.print(f"[green]✓[/] Wrote {cfg_path} and {sec_path}")
    else:
        console.print(f"[green]✓[/] Updated {cfg_path} (using existing secrets)")

    # Reload settings to get the merged config
    return load_config()


# This function is replaced by load_all_servers from utils


def _prompt_server_settings(
    imported: Dict[str, MCPServerConfig],
    server_name: str | None = None,
) -> tuple[str, MCPServerConfig]:
    """Prompt user for server settings."""
    # Check if user wants to import from file
    import_choice = Prompt.ask(
        "How would you like to add the server?",
        choices=["interactive", "from-mcp-json", "from-dxt"],
        default="interactive",
    )

    if import_choice == "from-mcp-json":
        mcp_json_path = Prompt.ask("Path to mcp.json file")
        try:
            import_result = import_servers_from_json(Path(mcp_json_path))
            if import_result.success:
                imported_new: Dict[str, MCPServerConfig] = {}
                for name, cfg in import_result.servers.items():
                    imported_new[name] = MCPServerConfig(**cfg)
                console.print(f"[green]Found {len(imported_new)} servers[/green]")
                if imported_new:
                    server_names = list(imported_new.keys())
                    chosen = Prompt.ask(
                        "Server to add", choices=server_names, default=server_names[0]
                    )
                    return chosen, imported_new[chosen]
        except Exception as e:
            console.print(f"[red]Error importing from mcp.json: {e}[/red]")
            console.print("Falling back to interactive entry...")

    elif import_choice == "from-dxt":
        dxt_path = Prompt.ask("Path to .dxt file")
        try:
            import_result = import_servers_from_dxt(Path(dxt_path))
            if import_result.success:
                imported_new: Dict[str, MCPServerConfig] = {}
                for name, cfg in import_result.servers.items():
                    imported_new[name] = MCPServerConfig(**cfg)
                console.print(f"[green]Found {len(imported_new)} servers[/green]")
                if imported_new:
                    server_names = list(imported_new.keys())
                    chosen = Prompt.ask(
                        "Server to add", choices=server_names, default=server_names[0]
                    )
                    return chosen, imported_new[chosen]
        except Exception as e:
            console.print(f"[red]Error importing from .dxt: {e}[/red]")
            console.print("Falling back to interactive entry...")

    # Manual entry or if imports failed
    if imported:
        console.print("[cyan]Available servers:[/cyan]")
        for n in imported.keys():
            console.print(f" - {n}")

    # If server_name was already provided, use it; otherwise prompt
    if not server_name:
        server_name = Prompt.ask("Server name (e.g., fetch)")
    if server_name in imported:
        return server_name, imported[server_name]

    transport = Prompt.ask(
        "Transport",
        choices=["stdio", "sse", "streamable_http", "websocket"],
        default="stdio",
    ).strip()

    server = MCPServerConfig(name=server_name, transport=transport)

    if transport == "stdio":
        # Allow user to enter full command or separate command/args
        command_input = Prompt.ask(
            "Command (full command like 'uvx mcp-server-fetch' or just 'uvx')"
        )

        # Check if this looks like a full command with args
        if " " in command_input:
            command, args = _parse_command_string(command_input)
            console.print(f"[dim]Parsed as: command='{command}' args={args}[/dim]")
            server.command = command
            server.args = args
        else:
            server.command = command_input
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
            kv = Prompt.ask("Header:Value pairs (comma-separated)", default="").strip()
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
        # Load settings from config (includes secrets)
        settings = load_config()

        # Show which config file we're using
        config_path = find_eval_config()
        if config_path:
            console.print(f"[dim]Using config: {config_path}[/dim]")

        # Set logger to errors only to reduce noise (disable transports entirely)
        settings.logger = LoggerSettings(
            type="none", level="error", progress_display=False
        )

        mcp_app = MCPApp(settings=settings)
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
                        name: str = (
                            getattr(t, "name", None)
                            or getattr(t, "tool", None)
                            or getattr(t, "id", None)
                            or ""
                        )
                        if not name:
                            continue
                        description: str | None = getattr(t, "description", None)
                        input_schema: Dict[str, Any] | None = (
                            getattr(t, "inputSchema", None)
                            or getattr(t, "input_schema", None)
                            or getattr(t, "input", None)
                        )
                        tools.append(
                            ToolSchema(
                                name=name,
                                description=description,
                                input_schema=input_schema,
                            )
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
                                input_schema=t.get("inputSchema")
                                or t.get("input_schema")
                                or t.get("input")
                                or None,
                            )
                        )
    except Exception as e:
        console.print(f"[red]Error discovering tools for '{server_name}': {e}[/red]")
    return tools


# These functions are replaced by write_server_to_mcpeval from utils


async def _emit_tests(
    project: Path,
    style: str,
    server_name: str,
    scenarios: List[Any],
    provider: str,
    model: str | None = None,
    output_path: Path | None = None,
) -> Path | None:
    style = style.strip().lower()
    safe_server = _sanitize_filename_component(server_name)
    if style == "dataset":
        ds = dataset_from_scenarios(scenarios, server_name)
        # Resolve output path
        if output_path is not None:
            out_file = (
                output_path if output_path.is_absolute() else (project / output_path)
            )
            # Ensure .yaml suffix
            if out_file.suffix.lower() not in (".yaml", ".yml"):
                out_file = out_file.with_suffix(".yaml")
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            ds_path = project / "datasets"
            ds_path.mkdir(parents=True, exist_ok=True)
            base_file = ds_path / f"{safe_server}_generated.yaml"
            out_file = base_file
            if out_file.exists():
                try:
                    slug = await _generate_llm_slug(server_name, provider, model)
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
            save_yaml(out_file, raw)
        console.print(f"[green]✓[/] Wrote dataset {out_file}")
        return out_file

    # Resolve test file output path
    if output_path is not None:
        out_path = output_path if output_path.is_absolute() else (project / output_path)
        if out_path.suffix.lower() != ".py":
            out_path = out_path.with_suffix(".py")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tests_dir = project / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        base_path = tests_dir / f"test_{safe_server}_generated.py"
        out_path = base_path
        if out_path.exists():
            try:
                slug = await _generate_llm_slug(server_name, provider, model)
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

    # Also print first few lines of generated test
    console.print("\n[bold]Generated test preview:[/bold]")
    lines = content.split("\n")[:30]  # Show first 30 lines
    for line in lines:
        console.print(f"[dim]{line}[/dim]")
    if len(content.split("\n")) > 30:
        console.print("[dim]... (truncated)[/dim]")

    return out_path


# --------------- main command -----------------


async def init_project(
    out_dir: str = typer.Option(".", help="Project directory for configs"),
    template: str = typer.Option(
        "basic",
        help="Bootstrap template: empty (no files), basic (config only), sample (examples + config)",
    ),
):
    """Initialize an mcp-eval project.

    Steps:

    - Ensure mcpeval.yaml and mcpeval.secrets.yaml exist (write minimal defaults)

    - Configure provider + API key

    - Import servers from mcp-agent.config.yaml and optionally mcp.json (cursor/vscode)

    - Define/select a default AgentSpec and set default_agent



    Examples:

    Initialize project: $ mcp-eval init
    """
    project = Path(out_dir)
    project.mkdir(parents=True, exist_ok=True)

    # Copy template files first so ensure_mcpeval_yaml does not overwrite
    tpl = (template or "").strip().lower()
    if tpl not in ("empty", "basic", "sample"):
        tpl = "basic"

    if tpl == "sample":
        console.print("[cyan]Bootstrapping sample project files...[/cyan]")
        try:
            copied = copy_sample_template(project)
            if copied:
                for p in copied:
                    console.print(f"[dim]  wrote {p}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to copy sample files: {e}[/yellow]")
    elif tpl == "basic":
        console.print("[cyan]Bootstrapping basic config files...[/cyan]")
        try:
            copy_sample_template(
                project,
                files_to_copy=[
                    "mcpeval.yaml",
                    "mcpeval.secrets.yaml.example",
                    "usage_example.py",
                    "sample_server.py",
                ],
                overwrite=False,
            )
        except Exception:
            # Fall through to ensure minimal config
            pass

    # Ensure mcpeval.yaml and secrets exist
    ensure_mcpeval_yaml(project)

    # Copy packaged sample README if none exists
    readme_path = project / "README.md"
    if not readme_path.exists():
        try:
            from importlib import resources

            src = resources.files("mcp_eval.data.sample").joinpath("README.md")
            with resources.as_file(src) as src_path:
                readme_path.write_bytes(Path(src_path).read_bytes())
            console.print(f"[green]✓[/] Wrote {readme_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not copy README.md: {e}[/yellow]")

    # Create a lightweight context for ModelSelector to avoid global fallback
    context = Context()

    # Provider + API key
    console.print("[cyan]Configuring LLM provider and secrets...[/cyan]")
    existing_provider, existing_key, available_providers, settings = (
        _load_existing_provider()
    )
    provider, api_key, model = _prompt_provider(
        existing_provider, existing_key, available_providers
    )
    # Always persist provider/model to mcpeval.yaml; write secrets only if api_key provided
    settings = await _write_mcpeval_configs(
        project, settings, provider, api_key, model, context=context
    )

    # Servers
    console.print("[cyan]Discovering servers...[/cyan]")
    existing_servers = load_all_servers(project)
    imported_servers: Dict[str, MCPServerConfig] = {}

    # Check for mcp.json files and prompt user
    mcp_json = find_mcp_json()
    if mcp_json:
        console.print(f"[cyan]Found MCP configuration at {mcp_json}[/cyan]")
        if Confirm.ask(
            "Would you like to import servers from this file?", default=True
        ):
            console.print(f"[cyan]Importing servers from {mcp_json}...[/cyan]")
            import_result = import_servers_from_json(mcp_json)
            if import_result.success:
                for name, cfg in import_result.servers.items():
                    imported_servers[name] = MCPServerConfig(**cfg)
                console.print(
                    f"[green]Imported {len(imported_servers)} servers[/green]"
                )
            else:
                console.print(
                    f"[yellow]Failed to import: {import_result.error}[/yellow]"
                )

    merged_servers: Dict[str, MCPServerConfig] = {
        **existing_servers,
        **imported_servers,
    }

    if merged_servers:
        console.print("[cyan]Available servers:[/cyan]")
        for n in sorted(merged_servers.keys()):
            console.print(f" - {n}")
    else:
        console.print("[yellow]No servers found[/yellow]")

    server_name: str | None = None
    added_server_now = False
    if Confirm.ask(
        "Would you like to add a server now?",
        default=(len(merged_servers) == 0),
    ):
        server_name, server_config = _prompt_server_settings(
            merged_servers, server_name=None
        )
        write_server_to_mcpeval(project, server_config)
        added_server_now = True
    else:
        console.print(
            "[yellow]Skipping server add. You can add servers later with 'mcp-eval server add'.[/yellow]"
        )

    # Default AgentSpec
    console.print("[cyan]Define default agent (will be stored in mcpeval.yaml)[/cyan]")
    agent_name = Prompt.ask("Agent name", default="default")
    instruction = Prompt.ask(
        "Agent instruction",
        default="You are a helpful assistant that can use MCP servers effectively.",
    )
    # Allow multiple servers
    default_servers_str = (
        server_name
        if (added_server_now and server_name)
        else ",".join(sorted(merged_servers.keys()))
        if merged_servers
        else ""
    )
    server_list_str = Prompt.ask(
        "Server names for this agent (comma-separated)",
        default=default_servers_str,
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


async def run_generator(
    out_dir: str = typer.Option(".", help="Project directory to write configs/tests"),
    style: str | None = typer.Option(
        None, help="Test style: pytest|decorators|dataset"
    ),
    n_examples: int = typer.Option(6, help="Number of scenarios to generate"),
    provider: str | None = typer.Option(None, help="LLM provider (anthropic|openai)"),
    model: str | None = typer.Option(None, help="Model id (optional)"),
    verbose: bool = typer.Option(False, help="Show detailed error messages"),
    output: str | None = typer.Option(
        None, help="Explicit output path for the generated file (.py or .yaml)"
    ),
    update: str | None = typer.Option(
        None,
        "--update",
        help="Append generated tests to an existing file instead of creating a new one",
    ),
):
    """Generate scenarios and write a single test file.

    Examples:

    Quick start (prompt for style): $ mcp-eval generate

    Explicit pytest style and 10 scenarios: $ mcp-eval generate --style pytest --n-examples 10

    With verbose output: $ mcp-eval generate --verbose
    """
    project = Path(out_dir)
    project.mkdir(parents=True, exist_ok=True)

    # If update mode is requested, delegate to update flow and return
    if update:
        console.print("[cyan]Update mode: appending tests to existing file[/cyan]")
        return await update_tests(
            out_dir=out_dir,
            target_file=update,
            server_name=None,
            style=style or "pytest",
            n_examples=n_examples,
            provider=provider,
            model=model,
        )

    # Create a lightweight context for ModelSelector to avoid global fallback
    context = Context()

    console.print(
        "[cyan]Checking credentials and writing mcpeval configs if needed...[/cyan]"
    )
    # Provider + API key (load existing when re-running)
    existing_provider, existing_key, available_providers, settings = (
        _load_existing_provider()
    )
    # Determine provider/api_key/model respecting CLI flags
    selected_provider: str
    api_key: str | None
    prompted_model: str | None = None
    if provider:
        selected_provider = provider.strip().lower()
        # Use env key if present; otherwise prompt
        if available_providers and selected_provider in available_providers:
            api_key = available_providers[selected_provider]
            console.print(
                f"[green]Using API key from environment for {selected_provider}[/green]"
            )
        else:
            api_key = Prompt.ask(f"Enter {selected_provider} API key", password=True)
        if not model:
            prompted_model = (
                Prompt.ask("Model (press Enter to auto-select)", default="").strip()
                or None
            )
    else:
        # Interactively prompt using detected defaults
        selected_provider, api_key, prompted_model = _prompt_provider(
            existing_provider, existing_key, available_providers
        )
    # Use CLI model if provided, otherwise use prompted model
    final_model = model or prompted_model
    # Always persist provider/model to mcpeval.yaml; write secrets only if api_key provided
    settings = await _write_mcpeval_configs(
        project, settings, selected_provider, api_key, final_model, context=context
    )

    # If no specific LLM model was chosen, select an intelligent one for generation
    generation_model = final_model
    if generation_model is None:
        try:
            selector = ModelSelector(context=context)
            preferences = ModelPreferences(
                costPriority=0.2, speedPriority=0.2, intelligencePriority=0.6
            )
            model_info = selector.select_best_model(
                model_preferences=preferences,
                provider=selected_provider,
                tool_calling=True,
                structured_outputs=True,
            )
            generation_model = model_info.name
            console.print(f"[dim]Selected generation model: {generation_model}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not auto-select generation model: {e}[/yellow]"
            )
            generation_model = final_model

    # Server capture (only from configs; mcp.json import happens in init)
    existing_servers = load_all_servers(project)
    if existing_servers:
        console.print("[cyan]Available servers:[/cyan]")
        for n in sorted(existing_servers.keys()):
            console.print(f" - {n}")
        console.print(
            "Type an existing server name to use it, or type a new name to add one."
        )
        chosen = Prompt.ask(
            "Server name", default=next(iter(sorted(existing_servers.keys())))
        )
        if chosen in existing_servers:
            server_name = chosen
            server_config = existing_servers[chosen]
        else:
            # Add new server via prompt - pass the server name user already typed
            server_name, server_config = _prompt_server_settings({}, server_name=chosen)
    else:
        # No servers known yet; prompt fresh
        console.print("[yellow]No servers configured yet[/yellow]")
        server_name, server_config = _prompt_server_settings({}, server_name=None)

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
            "Agent to use", choices=agent_names, default=default_agent or agent_names[0]
        )
        _set_default_agent(project, chosen_agent)
    else:
        console.print("[yellow]No agents defined.[/yellow]")
        if Confirm.ask("Create a default test agent now?", default=True):
            agent_name = Prompt.ask("Agent name", default="default")
            instruction = Prompt.ask(
                "Agent instruction",
                default="You are a helpful assistant that can use MCP servers effectively.",
            )
            server_list_str = Prompt.ask(
                "Server names for this agent (comma-separated)", default=server_name
            )
            server_list = [s.strip() for s in server_list_str.split(",") if s.strip()]
            agent_cfg = AgentConfig(
                name=agent_name,
                instruction=instruction,
                server_names=server_list,
                provider=provider,
                model=generation_model,
            )
            write_agent_to_mcpeval(project, agent_cfg, set_default=True)
            console.print(f"[green]✓ Created default agent '{agent_name}'[/green]")
        else:
            console.print(
                "[yellow]Proceeding without a default agent. Tests may use minimal defaults.[/yellow]"
            )

    # Discovery
    console.print(f"[cyan]Discovering tools for server '{server_name}'...[/cyan]")
    try:
        tools = await _discover_tools(server_name)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not list tools: {e}[/yellow]")
        tools = []
        if not Confirm.ask("Continue without tool discovery?", default=True):
            raise typer.Exit(1)

    if tools:
        console.print(f"[green]Discovered {len(tools)} tools:[/green]")
        for i, tool in enumerate(tools):
            # Truncate description if too long
            desc = tool.description or "No description"
            if len(desc) > 60:
                desc = desc[:57] + "..."
            console.print(f"  • [cyan]{tool.name}[/cyan]: {desc}")
            # Show first 10 tools, then summarize
            if i >= 9 and len(tools) > 10:
                console.print(f"  ... and {len(tools) - 10} more tools")
                break
    else:
        console.print("[yellow]No tools discovered[/yellow]")

    # Two-stage generation (first: scenarios; second: assertions refinement per scenario)
    console.print("[cyan]Generating test scenarios...[/cyan]")

    def progress_reporter(message: str):
        """Report progress to console."""
        console.print(f"[dim]  {message}[/dim]")

    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,  # Keep progress visible
        ) as progress:
            task = progress.add_task("Generating scenarios...", total=None)

            scenarios = await generate_scenarios_with_agent(
                tools=tools,
                n_examples=n_examples,
                provider=selected_provider,
                model=generation_model,
                progress_callback=progress_reporter,
                debug=verbose,
                max_retries=2,
            )
            progress.update(task, description=f"Generated {len(scenarios)} scenarios")

            # Print generated scenarios immediately
            console.print("\n[bold green]Generated Scenarios:[/bold green]")
            for i, scenario in enumerate(scenarios, 1):
                console.print(
                    f"  {i}. [cyan]{scenario.name}[/cyan]: {scenario.description or scenario.prompt[:60]}..."
                )
            console.print()

            progress.update(task, description="Refining assertions...")
            scenarios = await refine_assertions_with_agent(
                scenarios,
                tools,
                provider=selected_provider,
                model=generation_model,
                progress_callback=progress_reporter,
                debug=verbose,
            )
            progress.update(task, description="Completed generation")

            # Post-filter: drop scenarios without any tool assertions if tools exist
            allowed_tool_names = [t.name for t in tools if t.name]
            if allowed_tool_names:
                before = len(scenarios)

                def _has_tool_assertion(s):
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

                scenarios = [s for s in scenarios if _has_tool_assertion(s)]
                dropped = before - len(scenarios)
                if dropped > 0:
                    console.print(
                        f"[yellow]Dropped {dropped} scenarios without tool assertions[/yellow]"
                    )

        console.print(
            f"[green]✓ Generated and refined {len(scenarios)} test scenarios[/green]"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to generate scenarios: {e}[/red]")
        if verbose:
            import traceback

            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        raise typer.Exit(1)

    # Prompt for style if not provided
    if not style:
        style = (
            typer.prompt("Test style (pytest|decorators|dataset)", default="pytest")
            .strip()
            .lower()
        )

    # Generate tests and validate syntax
    output_path = Path(output) if output else None
    output_file = await _emit_tests(
        project,
        style,
        server_name,
        scenarios,
        provider=selected_provider,
        model=generation_model,
        output_path=output_path,
    )

    # Validate generated Python code if it's a Python test file
    if output_file and output_file.suffix == ".py":
        console.print("\n[cyan]Validating generated test file...[/cyan]")
        await _validate_and_fix_test_file(output_file, selected_provider, final_model)

    # Display run command
    _display_run_command(style, output_file)
    # Summary of generated scenarios
    if scenarios:
        console.print("\n[bold]Summary of generated scenarios:[/bold]")
        for s in scenarios[:20]:  # cap for display
            # Prefer description; fall back to a trimmed first line of the prompt
            try:
                summary_text = (s.description or "").strip()
                if not summary_text:
                    prompt_preview = (s.prompt or "").strip().split("\n")[0]
                    summary_text = prompt_preview[:120]
            except Exception:
                summary_text = ""
            console.print(f" - [green]{s.name}[/green]: {summary_text}")
        if len(scenarios) > 20:
            console.print(f" ... and {len(scenarios) - 20} more")


async def update_tests(
    out_dir: str = typer.Option(".", help="Project directory"),
    target_file: str = typer.Option(
        ..., help="Path to an existing test file to append to"
    ),
    server_name: str = typer.Option(
        None, help="Server to generate against (prompted if omitted)"
    ),
    style: str = typer.Option(
        "pytest", help="Test style for new tests: pytest|decorators|dataset"
    ),
    n_examples: int = typer.Option(4, help="Number of new scenarios to generate"),
    provider: str = typer.Option("anthropic", help="LLM provider (anthropic|openai)"),
    model: str | None = typer.Option(None, help="Model id (optional)"),
):
    """Append newly generated tests to an existing test file (non-interactive).

    Examples:

    Append 4 pytest-style tests to a file: $ mcp-eval update --target-file tests/test_fetch_generated.py --style pytest --n-examples 4
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
            console.print(
                "[yellow]No servers configured. Run 'mcp-eval init' first.[/yellow]"
            )
            raise typer.Exit(1)
        console.print("Available servers:")
        for n in sorted(existing_servers.keys()):
            console.print(f" - {n}")
        server_name = typer.prompt(
            "Server name", default=next(iter(sorted(existing_servers.keys())))
        )

    # Create a lightweight context for ModelSelector
    context = Context()

    # Provider + key (reuse flow and respect CLI)
    existing_provider, existing_key, available_providers, settings = (
        _load_existing_provider()
    )
    selected_provider: str
    api_key: str | None
    prompted_model: str | None = None
    if provider:
        selected_provider = provider.strip().lower()
        if available_providers and selected_provider in available_providers:
            api_key = available_providers[selected_provider]
            console.print(
                f"[green]Using API key from environment for {selected_provider}[/green]"
            )
        else:
            api_key = Prompt.ask(f"Enter {selected_provider} API key", password=True)
        if not model:
            prompted_model = (
                Prompt.ask("Model (press Enter to auto-select)", default="").strip()
                or None
            )
    else:
        selected_provider, api_key, prompted_model = _prompt_provider(
            existing_provider, existing_key, available_providers
        )
    final_model = model or prompted_model
    settings = await _write_mcpeval_configs(
        project, settings, selected_provider, api_key, final_model, context=context
    )

    console.print(f"[cyan]Listing tools for '{server_name}'...[/cyan]")
    try:
        tools = await _discover_tools(server_name)
    except Exception as e:
        console.print(f"[red]Failed to list tools:[/] {e}")
        raise typer.Exit(1)
    console.print(f"Discovered {len(tools)} tools")

    # Select an intelligent model for generation if not specified
    generation_model = final_model
    if generation_model is None:
        try:
            selector = ModelSelector(context=context)
            preferences = ModelPreferences(
                costPriority=0.2, speedPriority=0.2, intelligencePriority=0.6
            )
            model_info = selector.select_best_model(
                model_preferences=preferences,
                provider=selected_provider,
                tool_calling=True,
                structured_outputs=True,
            )
            generation_model = model_info.name
            console.print(f"[dim]Selected generation model: {generation_model}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not auto-select generation model: {e}[/yellow]"
            )
            generation_model = final_model

    # Generate and refine new scenarios
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        def progress_reporter(message: str):
            console.print(f"[dim]  {message}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task = progress.add_task("Generating scenarios...", total=None)

            scenarios = await generate_scenarios_with_agent(
                tools=tools,
                n_examples=n_examples,
                provider=selected_provider,
                model=generation_model,
                progress_callback=progress_reporter,
            )
            progress.update(task, description=f"Generated {len(scenarios)} scenarios")

            # Print generated scenarios immediately
            console.print("\n[bold green]Generated Scenarios:[/bold green]")
            for i, scenario in enumerate(scenarios, 1):
                console.print(
                    f"  {i}. [cyan]{scenario.name}[/cyan]: {scenario.description or scenario.prompt[:60]}..."
                )
            console.print()

            progress.update(task, description="Refining assertions...")
            scenarios = await refine_assertions_with_agent(
                scenarios,
                tools,
                provider=selected_provider,
                model=generation_model,
                progress_callback=progress_reporter,
            )
            progress.update(task, description="Completed generation")
    except Exception as e:
        console.print(f"[red]Failed to generate scenarios:[/] {e}")
        raise typer.Exit(1)

    # Render content and append
    if style.strip().lower() == "decorators":
        content = render_decorator_tests(scenarios, server_name)
    elif style.strip().lower() == "dataset":
        content = render_pytest_tests(
            scenarios, server_name
        )  # dataset append not ideal
    else:
        content = render_pytest_tests(scenarios, server_name)

    sep = "\n\n# ---- mcp-eval: additional generated tests ----\n\n"
    appended = sep + content
    file_path.write_text(
        file_path.read_text(encoding="utf-8") + appended, encoding="utf-8"
    )
    console.print(f"[green]✓[/] Appended {len(scenarios)} tests to {file_path}")


add_app = typer.Typer(
    help="Add resources to mcpeval.yaml (servers, agents).\n\nExamples:\n  - Add a server interactively:\n    mcp-eval server add\n\n  - Import servers from mcp.json:\n    mcp-eval server add --from-mcp-json .cursor/mcp.json\n\n  - Add an agent:\n    mcp-eval agent add"
)

# Noun-first groups
server_app = typer.Typer(help="Manage MCP servers (add, list)")
agent_app = typer.Typer(help="Manage agents (add, list)")

# Register noun-first groups for local module CLI execution
app.add_typer(server_app, name="server")
app.add_typer(agent_app, name="agent")
app.add_typer(add_app, name="add")


@server_app.command("add")
@add_app.command("server")
def add_server(
    out_dir: str = typer.Option(".", help="Project directory"),
    from_mcp_json: str | None = typer.Option(
        None, help="Path to mcp.json to import servers from"
    ),
    from_dxt: str | None = typer.Option(
        None, help="Path to DXT file to import servers from"
    ),
):
    """Add a server to mcpeval.yaml, either interactively or from mcp.json/DXT file.

    Examples:

    Interactive add: $ mcp-eval server add

    From mcp.json: $ mcp-eval server add --from-mcp-json .cursor/mcp.json
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
            console.print(
                f"[green]Found {len(imported_servers)} servers in {json_path}[/green]"
            )
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
            console.print(
                f"[green]Found {len(imported_servers)} servers in {dxt_path}[/green]"
            )
        else:
            console.print(
                f"[yellow]No servers found in DXT file: {import_result.error}[/yellow]"
            )

    if imported_servers:
        console.print("[cyan]Imported servers:[/cyan]")
        for n in imported_servers.keys():
            console.print(f" - {n}")

        server_names = list(imported_servers.keys())
        chosen = Prompt.ask(
            "Server to add", choices=server_names, default=server_names[0]
        )

        if chosen in imported_servers:
            write_server_to_mcpeval(project, imported_servers[chosen])
            console.print(f"[green]✓ Added server '{chosen}'[/green]")
            return

    # Interactive add
    server_name, server_config = _prompt_server_settings({}, server_name=None)
    write_server_to_mcpeval(project, server_config)
    console.print(f"[green]✓ Added server '{server_name}'[/green]")

    # If there is a default agent, offer to grant it access to the new server
    cfg_path = ensure_mcpeval_yaml(project)
    cfg = load_yaml(cfg_path)
    default_agent = cfg.get("default_agent")
    if default_agent:
        if Confirm.ask(
            f"Add server '{server_name}' to default agent '{default_agent}'?",
            default=True,
        ):
            # Load existing agent definitions
            agents = cfg.get("agents", {}).get("definitions", [])
            updated = False
            for a in agents:
                if isinstance(a, dict) and a.get("name") == default_agent:
                    servers = a.get("server_names", []) or []
                    if server_name not in servers:
                        servers.append(server_name)
                        a["server_names"] = servers
                    updated = True
                    break
            if updated:
                save_yaml(cfg_path, cfg)
                console.print(
                    f"[green]✓ Added '{server_name}' to default agent '{default_agent}'[/green]"
                )


@agent_app.command("add")
@add_app.command("agent")
def add_agent(
    out_dir: str = typer.Option(".", help="Project directory"),
):
    """Add an AgentSpec to mcpeval.yaml (validates referenced servers exist).

    Examples:

    Add an agent and set as default: $ mcp-eval agent add
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
        if not Confirm.ask(
            f"Agent '{name}' already exists. Replace it?", default=False
        ):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    instruction = Prompt.ask(
        "Instruction",
        default="You are a helpful assistant that can use MCP servers effectively.",
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
        console.print(
            f"[yellow]Warning: Referenced servers not found: {', '.join(missing)}[/yellow]"
        )
        if Confirm.ask("Would you like to add them now?", default=True):
            for s in missing:
                console.print(f"\n[cyan]Adding server '{s}'...[/cyan]")
                # Pre-fill the name
                console.print(f"Server name: {s}")
                _, server_config = _prompt_server_settings({}, server_name=s)
                server_config.name = s  # Ensure name matches
                write_server_to_mcpeval(project, server_config)

    # Create and save agent
    agent = AgentConfig(name=name, instruction=instruction, server_names=server_names)

    set_default = Confirm.ask(
        "Set this as default agent?", default=len(existing_agents) == 0
    )
    write_agent_to_mcpeval(project, agent, set_default=set_default)
    console.print(f"[green]✓ Added agent '{name}'[/green]")


# Noun-first: list commands delegated to existing implementations
@server_app.command("list")
def list_servers_command(
    project_dir: str = typer.Option(".", help="Project directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
):
    """List configured servers (alias of 'list servers')."""
    _list_servers_cmd(project_dir=project_dir, verbose=verbose)


@agent_app.command("list")
def list_agents_command(
    project_dir: str = typer.Option(".", help="Project directory"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full instructions"
    ),
    name: str | None = typer.Option(None, help="Show details for specific agent"),
):
    """List configured agents (alias of 'list agents')."""
    _list_agents_cmd(project_dir=project_dir, verbose=verbose, name=name)


# ---- Sync wrappers for Typer (Click) to execute async commands ----


@app.command("init")
def init_project_cli(
    out_dir: str = typer.Option(".", help="Project directory for configs"),
    template: str = typer.Option(
        "basic",
        help="Bootstrap template: empty (no files), basic (config only), sample (examples + config)",
    ),
):
    return asyncio.run(init_project(out_dir=out_dir, template=template))


@app.command("generate")
def run_generator_cli(
    out_dir: str = typer.Option(".", help="Project directory to write configs/tests"),
    style: str | None = typer.Option(
        None, help="Test style: pytest|decorators|dataset"
    ),
    n_examples: int = typer.Option(6, help="Number of scenarios to generate"),
    provider: str | None = typer.Option(
        None, help="LLM provider (anthropic|openai). If omitted, you'll be prompted."
    ),
    model: str | None = typer.Option(None, help="Model id (optional)"),
    verbose: bool = typer.Option(False, help="Show detailed error messages"),
    output: str | None = typer.Option(
        None, help="Explicit output path for the generated file (.py or .yaml)"
    ),
    update: str | None = typer.Option(
        None,
        "--update",
        help="Append generated tests to an existing file instead of creating a new one",
    ),
):
    """Generate test scenarios and write a test file for an MCP server.

    Discovers server tools, generates test scenarios with assertions,
    and outputs as pytest, decorator, or dataset format.

    Examples:
        $ mcp-eval generate
        $ mcp-eval generate --style pytest --n-examples 10
    """
    return asyncio.run(
        run_generator(
            out_dir=out_dir,
            style=style,
            n_examples=n_examples,
            provider=provider,
            model=model,
            verbose=verbose,
            output=output,
            update=update,
        )
    )
