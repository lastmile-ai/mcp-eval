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
import yaml
from rich.console import Console

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

app = typer.Typer(help="Generate MCP‑Eval tests for an MCP server")
console = Console()


# --------------- helpers -----------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _deep_merge(a: dict, b: dict) -> dict:
    out = a.copy()
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _look_for_mcp_json() -> Optional[Path]:
    for p in (Path(".cursor/mcp.json"), Path(".vscode/mcp.json")):
        if p.exists():
            return p
    return None


def _ensure_mcpeval_yaml(project: Path) -> Path:
    """Ensure mcpeval.yaml exists; create minimal defaults if missing."""
    cfg_path = project / "mcpeval.yaml"
    if not cfg_path.exists():
        console.print(f"[yellow]mcpeval.yaml not found in {project}. Creating minimal config...[/yellow]")
        minimal = {
            "reporting": {"formats": ["json", "markdown"], "output_dir": "./test-reports"},
            "judge": {"min_score": 0.8},
        }
        _save_yaml(cfg_path, minimal)
        console.print(f"[green]✓[/] Created {cfg_path}")
    return cfg_path


def _write_agent_definition(
    project: Path,
    *,
    name: str,
    instruction: str,
    server_names: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Append/merge an AgentSpec under agents.definitions in mcpeval.yaml."""
    cfg_path = _ensure_mcpeval_yaml(project)
    cfg = _load_yaml(cfg_path)
    cfg.setdefault("agents", {}).setdefault("definitions", [])
    # Remove duplicates by name
    existing: List[dict] = [d for d in cfg["agents"]["definitions"] if isinstance(d, dict)]
    existing = [d for d in existing if d.get("name") != name]
    spec: Dict[str, Any] = {
        "name": name,
        "instruction": instruction,
        "server_names": server_names,
    }
    if provider:
        spec["provider"] = provider
    if model:
        spec["model"] = model
    existing.append(spec)
    cfg["agents"]["definitions"] = existing
    _save_yaml(cfg_path, cfg)
    console.print(f"[green]✓[/] Wrote AgentSpec '{name}' to {cfg_path}")


def _set_default_agent(project: Path, agent_name: str) -> None:
    cfg_path = _ensure_mcpeval_yaml(project)
    cfg = _load_yaml(cfg_path)
    cfg["default_agent"] = agent_name
    _save_yaml(cfg_path, cfg)
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


def _import_servers_from_mcp_json(mcp_json_path: Path) -> Dict[str, MCPServerSettings]:
    try:
        data = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        result: Dict[str, MCPServerSettings] = {}
        mcp_servers = data.get("mcpServers") or data.get("servers") or {}
        for name, cfg in mcp_servers.items():
            # Heuristics to map to typed MCPServerSettings
            settings = MCPServerSettings(
                name=name,
                transport=cfg.get("transport")
                or ("stdio" if cfg.get("command") else "sse"),
                command=cfg.get("command"),
                args=cfg.get("args") or [],
                url=cfg.get("url"),
                headers=cfg.get("headers"),
                env=cfg.get("env"),
            )
            result[name] = settings
        return result
    except Exception:
        return {}


def _import_servers_from_dxt(path: Path) -> Dict[str, MCPServerSettings]:
    """Best-effort import of servers from a DXT file.

    Heuristics:
    - If file contains an mcpServers object (emergent MCP JSON standard), parse it just like mcp.json
      (See `mcpServers` format widely used by Cursor/VS Code/Claude Desktop per docs: https://gofastmcp.com/integrations/mcp-json-configuration)
    - Otherwise, return empty and let the caller fall back to interactive add.
    """
    try:
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except Exception:
            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return {}
        if "mcpServers" in data and isinstance(data["mcpServers"], dict):
            # Reuse mcpServers mapping
            servers: Dict[str, MCPServerSettings] = {}
            for name, cfg in data["mcpServers"].items():
                servers[name] = MCPServerSettings(
                    name=name,
                    transport="stdio",  # most mcp-json clients rely on command/args
                    command=cfg.get("command"),
                    args=cfg.get("args") or [],
                    env=cfg.get("env"),
                )
            return servers
    except Exception:
        return {}
    return {}


def _load_existing_provider(project: Path) -> tuple[Optional[str], Optional[str]]:
    sec_path = project / "mcpeval.secrets.yaml"
    sec = _load_yaml(sec_path)
    # Also check .mcp-eval/secrets.yaml
    if not sec:
        sec = _load_yaml(project / ".mcp-eval" / "secrets.yaml")
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
        console.print(f"Using existing provider: {existing_provider}")
        if existing_key:
            console.print("API key already set in secrets; skipping prompt.")
            # Optionally ask for model override
            model = typer.prompt("Model (press Enter to auto-select)", default="").strip() or None
            return existing_provider, existing_key, model
        # Ask only for missing key
        api_key = typer.prompt(f"Enter {existing_provider} API key", hide_input=True)
        model = typer.prompt("Model (press Enter to auto-select)", default="").strip() or None
        return existing_provider, api_key, model
    # No existing provider; prompt fresh
    provider = (
        typer.prompt("LLM provider (anthropic|openai)", default="anthropic")
        .strip()
        .lower()
    )
    if provider not in ("anthropic", "openai"):
        provider = "anthropic"
    api_key = typer.prompt(f"Enter {provider} API key", hide_input=True)
    model = typer.prompt("Model (press Enter to auto-select)", default="").strip() or None
    return provider, api_key, model


def _write_mcpeval_configs(project: Path, provider: str, api_key: str, model: Optional[str] = None) -> None:
    cfg_path = project / "mcpeval.yaml"
    sec_path = project / "mcpeval.secrets.yaml"

    cfg = _load_yaml(cfg_path)
    sec = _load_yaml(sec_path)

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
        except Exception:
            # Fallback if ModelSelector fails
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

    _save_yaml(cfg_path, _deep_merge(cfg, cfg_overlay))
    _save_yaml(sec_path, _deep_merge(sec, sec_overlay))
    console.print(f"[green]✓[/] Wrote {cfg_path} and {sec_path}")


def _load_existing_servers(project: Path) -> Dict[str, MCPServerSettings]:
    servers: Dict[str, MCPServerSettings] = {}
    # mcp-agent.config.yaml
    agent_cfg_path = project / "mcp-agent.config.yaml"
    if agent_cfg_path.exists():
        data = _load_yaml(agent_cfg_path)
        for name, cfg in (data.get("mcp", {}).get("servers", {}) or {}).items():
            try:
                servers[name] = MCPServerSettings(name=name, **cfg)
            except Exception:
                # Best-effort
                servers[name] = MCPServerSettings(
                    name=name,
                    transport=cfg.get("transport") or "stdio",
                    command=cfg.get("command"),
                    args=cfg.get("args") or [],
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                    env=cfg.get("env"),
                )
    # mcpeval.yaml may also define servers
    eval_cfg_path = project / "mcpeval.yaml"
    if eval_cfg_path.exists():
        data = _load_yaml(eval_cfg_path)
        for name, cfg in (data.get("mcp", {}).get("servers", {}) or {}).items():
            if name not in servers:
                try:
                    servers[name] = MCPServerSettings(name=name, **cfg)
                except Exception:
                    servers[name] = MCPServerSettings(
                        name=name,
                        transport=cfg.get("transport") or "stdio",
                        command=cfg.get("command"),
                        args=cfg.get("args") or [],
                        url=cfg.get("url"),
                        headers=cfg.get("headers"),
                        env=cfg.get("env"),
                    )
    # mcpeval.config.yaml or .mcp-eval/config.yaml
    eval_cfg2 = project / "mcpeval.config.yaml"
    if eval_cfg2.exists():
        data = _load_yaml(eval_cfg2)
        for name, cfg in (data.get("mcp", {}).get("servers", {}) or {}).items():
            if name not in servers:
                try:
                    servers[name] = MCPServerSettings(name=name, **cfg)
                except Exception:
                    servers[name] = MCPServerSettings(
                        name=name,
                        transport=cfg.get("transport") or "stdio",
                        command=cfg.get("command"),
                        args=cfg.get("args") or [],
                        url=cfg.get("url"),
                        headers=cfg.get("headers"),
                        env=cfg.get("env"),
                    )
    dot_cfg = project / ".mcp-eval" / "config.yaml"
    if dot_cfg.exists():
        data = _load_yaml(dot_cfg)
        for name, cfg in (data.get("mcp", {}).get("servers", {}) or {}).items():
            if name not in servers:
                try:
                    servers[name] = MCPServerSettings(name=name, **cfg)
                except Exception:
                    servers[name] = MCPServerSettings(
                        name=name,
                        transport=cfg.get("transport") or "stdio",
                        command=cfg.get("command"),
                        args=cfg.get("args") or [],
                        url=cfg.get("url"),
                        headers=cfg.get("headers"),
                        env=cfg.get("env"),
                    )
    return servers


def _prompt_server_settings(
    imported: Dict[str, MCPServerSettings],
) -> tuple[str, MCPServerSettings]:
    if imported:
        console.print("Imported servers:")
        for n in imported.keys():
            console.print(f" - {n}")

    server_name = typer.prompt("Server name (e.g., fetch)")
    if server_name in imported:
        return server_name, imported[server_name]

    transport = typer.prompt(
        "Transport (stdio|sse|streamable_http|websocket)", default="stdio"
    ).strip()
    kwargs: Dict[str, Any] = {"name": server_name, "transport": transport}
    if transport == "stdio":
        kwargs["command"] = typer.prompt("Command (e.g., npx|uvx|python)")
        args = typer.prompt(
            "Args (space-separated, blank for none)", default=""
        ).strip()
        kwargs["args"] = [a for a in args.split(" ") if a]
        if typer.confirm("Add environment variables?", default=False):
            kv = typer.prompt("KEY=VALUE pairs (comma-separated)", default="").strip()
            if kv:
                env: Dict[str, str] = {}
                for pair in kv.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        env[k.strip()] = v.strip()
                kwargs["env"] = env
    else:
        kwargs["url"] = typer.prompt("Server URL")
        if typer.confirm("Add HTTP headers?", default=False):
            kv = typer.prompt(
                "Header:Value pairs (comma-separated)", default=""
            ).strip()
            if kv:
                headers: Dict[str, str] = {}
                for pair in kv.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        headers[k.strip()] = v.strip()
                if headers:
                    kwargs["headers"] = headers

    return server_name, MCPServerSettings(**kwargs)


async def _discover_tools(server_name: str) -> List[ToolSchema]:
    """Connect to the server and return typed tool specs."""
    tools: List[ToolSchema] = []
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
    return tools


def _write_server_to_agent_config_file(
    project: Path, name: str, settings: MCPServerSettings
) -> None:
    cfg_path = project / "mcp-agent.config.yaml"
    cfg = _load_yaml(cfg_path)
    cfg.setdefault("mcp", {}).setdefault("servers", {})
    cfg["mcp"]["servers"][name] = settings.model_dump(exclude_none=True)
    _save_yaml(cfg_path, cfg)
    console.print(f"[green]✓[/] Updated {cfg_path} with server '{name}'")


def _write_server_to_mcpeval_file(
    project: Path, name: str, settings: MCPServerSettings
) -> None:
    """Persist server in mcpeval.yaml under mcp.servers."""
    cfg_path = _ensure_mcpeval_yaml(project)
    cfg = _load_yaml(cfg_path)
    cfg.setdefault("mcp", {}).setdefault("servers", {})

    # Normalize fields we allow in mcpeval.yaml
    src = settings.model_dump(exclude_none=True)
    server_obj: Dict[str, Any] = {}
    for k in ("transport", "command", "args", "url", "headers", "env"):
        if k in src:
            server_obj[k] = src[k]

    cfg["mcp"]["servers"][name] = server_obj
    _save_yaml(cfg_path, cfg)
    console.print(f"[green]✓[/] Updated {cfg_path} with server '{name}'")


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
        _ensure_dir(ds_path)
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
    _ensure_dir(tests_dir)
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
    import_mcp_json: bool = typer.Option(True, help="Import servers from .cursor/.vscode mcp.json if found"),
):
    """Initialize an mcp-eval project.

    - Ensure mcpeval.yaml and mcpeval.secrets.yaml exist (write minimal defaults)
    - Configure provider + API key
    - Import servers from mcp-agent.config.yaml and optionally mcp.json (cursor/vscode)
    - Define/select a default AgentSpec and set default_agent

    Examples:
      - Create a project with auto-import from mcp.json:
        mcp-eval init --import-mcp-json

      - Create without importing from mcp.json:
        mcp-eval init --no-import-mcp-json
    """
    project = Path(out_dir)
    _ensure_dir(project)
    _ensure_mcpeval_yaml(project)

    # Provider + API key
    console.print("[cyan]Configuring LLM provider and secrets...[/cyan]")
    existing_provider, existing_key = _load_existing_provider(project)
    provider, api_key, model = _prompt_provider(existing_provider, existing_key)
    if api_key:
        _write_mcpeval_configs(project, provider, api_key, model)
    else:
        console.print("Using existing secrets without modification.")

    # Servers
    console.print("[cyan]Discovering servers from mcp-agent.config.yaml...[/cyan]")
    existing_servers = _load_existing_servers(project)
    imported: Dict[str, MCPServerSettings] = {}
    if import_mcp_json:
        mcp_json = _look_for_mcp_json()
        if mcp_json:
            console.print(f"[cyan]Importing servers from {mcp_json}...[/cyan]")
            imported = _import_servers_from_mcp_json(mcp_json)
    merged: Dict[str, MCPServerSettings] = {**existing_servers, **imported}
    if merged:
        console.print("Available servers:")
        for n in sorted(merged.keys()):
            console.print(f" - {n}")
    else:
        console.print("[yellow]No servers found; let's add one[/yellow]")
    server_name, server_settings = _prompt_server_settings(merged)
    _write_server_to_mcpeval_file(project, server_name, server_settings)

    # Default AgentSpec
    console.print("[cyan]Define default agent (will be stored in mcpeval.yaml)[/cyan]")
    agent_name = typer.prompt("Agent name", default="default")
    instruction = typer.prompt(
        "Agent instruction",
        default="You are a helpful assistant that can use MCP servers effectively.",
    )
    # Allow multiple servers
    server_list_str = typer.prompt(
        "Server names for this agent (comma-separated)", default=server_name
    )
    server_list = [s.strip() for s in server_list_str.split(",") if s.strip()]
    _write_agent_definition(
        project,
        name=agent_name,
        instruction=instruction,
        server_names=server_list,
        provider=provider,
        model=model,
    )
    _set_default_agent(project, agent_name)
    console.print("[bold green]✓ Project initialized[/bold green]")


@app.command("run")
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
        mcp-eval run

      - Explicit pytest style and 10 scenarios:
        mcp-eval run --style pytest --n-examples 10
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
        console.print("Using existing secrets without modification.")

    # Server capture (only from configs; mcp.json import happens in init)
    existing_servers = _load_existing_servers(project)
    merged: Dict[str, MCPServerSettings] = {**existing_servers}
    if merged:
        console.print("Available servers:")
        for n in sorted(merged.keys()):
            console.print(f" - {n}")
        console.print(
            "Type an existing server name to use it, or type a new name to add one."
        )
        chosen = typer.prompt("Server name", default=next(iter(sorted(merged.keys()))))
        if chosen in merged:
            server_name = chosen
            server_settings = merged[chosen]
        else:
            # Add new server via prompt; prefill imported if present under that name
            server_name, server_settings = _prompt_server_settings({})
    else:
        # No servers known yet; prompt fresh
        server_name, server_settings = _prompt_server_settings({})
    # Persist/ensure presence in mcp-agent config
    # Persist to mcpeval.yaml (source of truth for this tool)
    _write_server_to_mcpeval_file(project, server_name, server_settings)
    console.print(f"[cyan]Server '{server_name}' configured.[/cyan]")

    # Agent selection by name from mcpeval.yaml agents.definitions
    cfg_path = _ensure_mcpeval_yaml(project)
    cfg = _load_yaml(cfg_path)
    agent_defs = [d for d in (cfg.get("agents", {}).get("definitions", []) or []) if isinstance(d, dict)]
    agent_names = [d.get("name") for d in agent_defs if d.get("name")]
    default_agent = cfg.get("default_agent") or (agent_names[0] if agent_names else None)
    if agent_names:
        console.print("Available agents:")
        for n in agent_names:
            console.print(f" - {n}")
        chosen_agent = typer.prompt("Agent to use (by name)", default=default_agent or agent_names[0])
        _set_default_agent(project, chosen_agent)
    else:
        console.print("[yellow]No agents defined. Consider running 'mcp-eval init' first to define a default agent.[/yellow]")

    # Discovery
    try:
        import asyncio

        tools = asyncio.run(_discover_tools(server_name))
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Could not list tools: {e}")
        tools = []
    if tools:
        console.print(
            f"Discovered tools ({len(tools)}): "
            + ", ".join([t.name for t in tools[:10]])
            + (" ..." if len(tools) > 10 else "")
        )
    else:
        console.print("Discovered tools: (none)")

    # Two-stage generation (first: scenarios; second: assertions refinement per scenario)
    try:
        import asyncio

        scenarios = asyncio.run(
            generate_scenarios_with_agent(
                tools=tools, n_examples=n_examples, provider=provider, model=final_model
            )
        )
        console.print(f"[cyan]Generated {len(scenarios)} initial scenarios[/cyan]")
        scenarios = asyncio.run(
            refine_assertions_with_agent(
                scenarios, tools, provider=provider, model=final_model
            )
        )
        console.print("[cyan]Refined assertions for scenarios[/cyan]")
    except Exception as e:
        console.print(f"[red]Failed to generate scenarios:[/] {e}")
        return

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
    existing_servers = _load_existing_servers(project)
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
    _ensure_dir(project)

    imported: Dict[str, MCPServerSettings] = {}
    if from_mcp_json:
        imported = _import_servers_from_mcp_json(Path(from_mcp_json))
        if not imported:
            console.print("[yellow]No servers found in mcp.json[/yellow]")
    elif from_dxt:
        imported = _import_servers_from_dxt(Path(from_dxt))
        if not imported:
            console.print("[yellow]No servers found in DXT file[/yellow]")

    if imported:
        console.print("Imported servers:")
        for n in imported.keys():
            console.print(f" - {n}")
        chosen = typer.prompt("Server to add", default=next(iter(imported.keys())))
        if chosen in imported:
            _write_server_to_mcpeval_file(project, chosen, imported[chosen])
            console.print(f"[green]✓[/] Added server '{chosen}'")
            return

    # Interactive add
    server_name, server_settings = _prompt_server_settings({})
    _write_server_to_mcpeval_file(project, server_name, server_settings)
    console.print(f"[green]✓[/] Added server '{server_name}'")


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
    _ensure_dir(project)
    _ensure_mcpeval_yaml(project)

    # Gather AgentSpec fields
    name = typer.prompt("Agent name")
    instruction = typer.prompt("Instruction", default="You are a helpful assistant that can use MCP servers effectively.")
    server_list_str = typer.prompt("Server names (comma-separated)")
    server_names = [s.strip() for s in server_list_str.split(",") if s.strip()]

    # Validate servers exist
    existing_servers = _load_existing_servers(project)
    missing = [s for s in server_names if s not in existing_servers]
    if missing:
        console.print(f"[yellow]Warning:[/] Referenced servers not found: {', '.join(missing)}")
        if typer.confirm("Would you like to add them now?", default=True):
            for s in missing:
                console.print(f"Adding server '{s}'...")
                n, settings = _prompt_server_settings({})
                # Ensure the name matches intent
                if n != s:
                    console.print(f"[yellow]Note:[/] Entered name '{n}' differs from '{s}'. Using '{n}'.")
                _write_server_to_mcpeval_file(project, n, settings)

    _write_agent_definition(project, name=name, instruction=instruction, server_names=server_names)
    if typer.confirm("Set this as default_agent?", default=False):
        _set_default_agent(project, name)
    console.print(f"[green]✓[/] Added agent '{name}'")
