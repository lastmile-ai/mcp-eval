"""List command for showing configured servers and agents."""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mcp_eval.cli.utils import load_all_servers, load_all_agents, load_yaml

app = typer.Typer(help="List configured resources")
console = Console()


@app.command("servers")
def list_servers(
    project_dir: str = typer.Option(".", help="Project directory"),
):
    """List all configured MCP servers."""
    project = Path(project_dir)
    servers = load_all_servers(project)
    
    if not servers:
        console.print("[yellow]No servers configured.[/yellow]")
        console.print("\nTo add servers, run:")
        console.print("  [cyan]mcp-eval add server[/cyan]")
        return
    
    table = Table(title="Configured MCP Servers", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Transport", style="yellow")
    table.add_column("Command/URL")
    table.add_column("Args", style="dim")
    
    for name, server in servers.items():
        if server.transport == "stdio":
            location = server.command or ""
            args = " ".join(server.args) if server.args else ""
        else:
            location = server.url or ""
            args = ""
        
        table.add_row(name, server.transport, location, args)
    
    console.print(table)


@app.command("agents")
def list_agents(
    project_dir: str = typer.Option(".", help="Project directory"),
):
    """List all configured agents."""
    project = Path(project_dir)
    agents = load_all_agents(project)
    
    # Get default agent
    config_path = project / "mcpeval.yaml"
    default_agent = None
    if config_path.exists():
        data = load_yaml(config_path)
        default_agent = data.get("default_agent")
    
    if not agents:
        console.print("[yellow]No agents configured.[/yellow]")
        console.print("\nTo add agents, run:")
        console.print("  [cyan]mcp-eval add agent[/cyan]")
        return
    
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
        
        table.add_row(
            agent.name,
            servers,
            provider,
            model,
            is_default
        )
    
    console.print(table)
    
    if agents:
        # Show instruction for first agent as example
        first_agent = agents[0]
        instruction_preview = first_agent.instruction[:100] + "..." if len(first_agent.instruction) > 100 else first_agent.instruction
        
        panel = Panel(
            f"[bold]{first_agent.name}[/bold]\n\n{instruction_preview}",
            title="Example Agent Instruction",
            border_style="dim"
        )
        console.print("\n", panel)


@app.command("all")
def list_all(
    project_dir: str = typer.Option(".", help="Project directory"),
):
    """List all configured resources (servers and agents)."""
    list_servers(project_dir)
    console.print()  # Add spacing
    list_agents(project_dir)


if __name__ == "__main__":
    app()