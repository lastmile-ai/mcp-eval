"""List command for showing configured servers and agents."""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mcp_eval.cli.utils import (
    load_all_servers,
    load_all_agents,
    load_yaml,
)
from mcp_eval.config import find_eval_config

app = typer.Typer(help="List configured resources")
console = Console()


@app.command("servers")
def list_servers(
    project_dir: str = typer.Option(".", help="Project directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
):
    """List all configured MCP servers."""
    project = Path(project_dir)
    servers = load_all_servers(project)

    if not servers:
        console.print("[yellow]No servers configured.[/yellow]")
        console.print("\nTo add servers, run:")
        console.print("  [cyan]mcp-eval server add[/cyan]")
        return

    if verbose:
        # Detailed view
        for name, server in servers.items():
            console.print(f"\n[bold green]{name}[/bold green]")
            console.print(f"  Transport: [yellow]{server.transport}[/yellow]")

            if server.transport == "stdio":
                console.print(f"  Command: {server.command or '(not set)'}")
                if server.args:
                    console.print(f"  Args: {' '.join(server.args)}")
            else:
                console.print(f"  URL: {server.url or '(not set)'}")

            if server.env:
                console.print("  Environment:")
                for key, value in server.env.items():
                    # Mask sensitive values
                    if (
                        "KEY" in key.upper()
                        or "SECRET" in key.upper()
                        or "TOKEN" in key.upper()
                    ):
                        masked = value[:3] + "***" if len(value) > 3 else "***"
                        console.print(f"    {key}: {masked}")
                    else:
                        console.print(f"    {key}: {value}")

            if server.headers:
                console.print("  Headers:")
                for key, value in server.headers.items():
                    # Mask auth headers
                    if "auth" in key.lower() or "token" in key.lower():
                        masked = value[:8] + "***" if len(value) > 8 else "***"
                        console.print(f"    {key}: {masked}")
                    else:
                        console.print(f"    {key}: {value}")
    else:
        # Table view
        table = Table(
            title="Configured MCP Servers", show_header=True, header_style="bold cyan"
        )
        table.add_column("Name", style="green")
        table.add_column("Transport", style="yellow")
        table.add_column("Command/URL")
        table.add_column("Args", style="dim")
        table.add_column("Extra", style="dim")

        for name, server in servers.items():
            if server.transport == "stdio":
                location = server.command or ""
                args = " ".join(server.args) if server.args else ""
            else:
                location = server.url or ""
                args = ""

            # Show if there are env vars or headers
            extras = []
            if server.env:
                extras.append(f"{len(server.env)} env")
            if server.headers:
                extras.append(f"{len(server.headers)} headers")
            extra_str = ", ".join(extras)

            table.add_row(name, server.transport, location, args, extra_str)

        console.print(table)
        console.print("\n[dim]Use --verbose/-v to see full details[/dim]")


@app.command("agents")
def list_agents(
    project_dir: str = typer.Option(".", help="Project directory"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full instructions"
    ),
    name: str | None = typer.Option(None, help="Show details for specific agent"),
):
    """List all configured agents."""
    project = Path(project_dir)
    agents = load_all_agents(project)

    # Get default agent
    config_path = find_eval_config(project)
    if config_path:
        console.print(f"[dim]Using config: {config_path}[/dim]")
    default_agent = None
    if config_path is not None:
        data = load_yaml(config_path)
        default_agent = data.get("default_agent")

    if not agents:
        console.print("[yellow]No agents configured.[/yellow]")
        console.print("\nTo add agents, run:")
        console.print("  [cyan]mcp-eval agent add[/cyan]")
        return

    # If specific agent requested
    if name:
        agent = next((a for a in agents if a.name == name), None)
        if not agent:
            console.print(f"[red]Agent '{name}' not found[/red]")
            console.print("\nAvailable agents:")
            for a in agents:
                console.print(f"  - {a.name}")
            return

        # Show full details for specific agent
        is_default = " [cyan](default)[/cyan]" if agent.name == default_agent else ""
        console.print(f"\n[bold green]{agent.name}[/bold green]{is_default}")
        console.print("\n[bold]Instruction:[/bold]")
        console.print(Panel(agent.instruction, border_style="dim"))
        console.print(
            f"\n[bold]Servers:[/bold] {', '.join(agent.server_names) if agent.server_names else '(none)'}"
        )
        console.print(f"[bold]Provider:[/bold] {agent.provider or '(from settings)'}")
        console.print(f"[bold]Model:[/bold] {agent.model or '(auto-selected)'}")
        return

    # Table view
    table = Table(title="Configured Agents", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Servers", style="yellow")
    table.add_column("Provider", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Default", style="cyan")

    for agent in agents:
        is_default = "✓" if agent.name == default_agent else ""
        servers = ", ".join(agent.server_names) if agent.server_names else "(none)"
        provider = agent.provider or "(settings)"
        model = agent.model or "(auto)"

        table.add_row(agent.name, servers, provider, model, is_default)

    console.print(table)

    if verbose:
        # Show all instructions
        console.print("\n[bold]Agent Instructions:[/bold]\n")
        for agent in agents:
            is_default = (
                " [cyan](default)[/cyan]" if agent.name == default_agent else ""
            )
            console.print(f"[bold green]{agent.name}[/bold green]{is_default}")
            console.print(Panel(agent.instruction, border_style="dim"))
            console.print()
    else:
        # Show help for seeing more
        console.print("\n[dim]Use --verbose/-v to see all instructions[/dim]")
        console.print("[dim]Use --name <agent> to see specific agent details[/dim]")


@app.command("all")
def list_all(
    project_dir: str = typer.Option(".", help="Project directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
):
    """List all configured resources (servers and agents)."""
    list_servers(project_dir=project_dir, verbose=verbose)
    console.print()  # Add spacing
    list_agents(project_dir=project_dir, verbose=verbose, name=None)


if __name__ == "__main__":
    app()
