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

from mcp_eval.generation import (
    generate_scenarios_with_agent,
    refine_assertions_with_agent,
    render_pytest_tests,
    render_decorator_tests,
    dataset_from_scenarios,
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
            model = (
                typer.prompt("Model (press Enter to auto-select)", default="").strip()
                or None
            )
            return existing_provider, existing_key, model
        # Ask only for missing key
        api_key = typer.prompt(f"Enter {existing_provider} API key", hide_input=True)
        model = (
            typer.prompt("Model (press Enter to auto-select)", default="").strip()
            or None
        )
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
    model = (
        typer.prompt("Model (press Enter to auto-select)", default="").strip() or None
    )
    return provider, api_key, model


def _write_mcpeval_configs(
    project: Path, provider: str, api_key: str, model: Optional[str] = None
) -> None:
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
                costPriority=0.4, speedPriority=0.2, intelligencePriority=0.4
            )
            model_info = selector.select_best_model(
                model_preferences=preferences, provider=provider
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


async def _discover_tools(server_name: str) -> List[Dict[str, Any]]:
    """Connect to the server and return tool specs as dicts with name/description/schema."""
    tools: List[Dict[str, Any]] = []
    mcp_app = MCPApp()
    async with mcp_app.run() as running:
        async with gen_client(
            server_name, server_registry=running.context.server_registry
        ) as client:
            result = await client.list_tools()
            try:
                data = result.model_dump()
            except Exception:
                data = getattr(result, "dict", lambda: {})()
            items = data.get("tools") or data.get("items") or []
            for t in items:
                tools.append(
                    {
                        "name": t.get("name") or t.get("tool") or t.get("id"),
                        "description": t.get("description") or "",
                        "input_schema": t.get("inputSchema")
                        or t.get("input_schema")
                        or t.get("input")
                        or {},
                    }
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


@app.command("run")
def run_generator(
    out_dir: str = typer.Option(".", help="Project directory to write configs/tests"),
    style: str = typer.Option("pytest", help="Test style: pytest|decorators|dataset"),
    n_examples: int = typer.Option(6, help="Number of scenarios to generate"),
    provider: str = typer.Option("anthropic", help="LLM provider (anthropic|openai)"),
    model: Optional[str] = typer.Option(None, help="Model id (optional)"),
):
    project = Path(out_dir)
    _ensure_dir(project)

    # Provider + API key (load existing when re-running)
    existing_provider, existing_key = _load_existing_provider(project)
    # Get provider, api_key, and optional model from prompt
    provider, api_key, prompted_model = _prompt_provider(
        existing_provider, existing_key
    )
    # Use CLI model if provided, otherwise use prompted model
    final_model = model or prompted_model
    if api_key:
        _write_mcpeval_configs(project, provider, api_key, final_model)
    else:
        console.print("Using existing secrets without modification.")

    # Server capture (reuse existing servers and optionally import from mcp.json)
    existing_servers = _load_existing_servers(project)
    imported = {}
    mcp_json = _look_for_mcp_json()
    if mcp_json:
        imported = _import_servers_from_mcp_json(mcp_json)
    # Merge existing + imported for user selection
    merged: Dict[str, MCPServerSettings] = {**existing_servers, **imported}
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
            pre = {chosen: imported[chosen]} if chosen in imported else {}
            server_name, server_settings = _prompt_server_settings(pre)
    else:
        # No servers known yet; prompt fresh
        server_name, server_settings = _prompt_server_settings(imported)
    # Persist/ensure presence in mcp-agent config
    _write_server_to_agent_config_file(project, server_name, server_settings)

    # Discovery
    try:
        import asyncio

        tools = asyncio.run(_discover_tools(server_name))
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Could not list tools: {e}")
        tools = []
    console.print(
        f"Discovered tools: {', '.join([t['name'] for t in tools]) if tools else '(none)'}"
    )

    # Two-stage generation (first: scenarios; second: assertions refinement per scenario)
    try:
        import asyncio

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
        return

    _emit_tests(
        project, style, server_name, scenarios, provider=provider, model=final_model
    )
