# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich",
#     "typer",
#     "pydantic>=2.10.4",
#     "pydantic-settings>=2.7.0",
#     "mcp-agent>=0.1.13",
#     "pyyaml>=6.0.2"
# ]
# ///
"""
Generate JSON schema for MCP‑Eval configuration (overlays mcp-agent + mcpeval files).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import typer
from rich.console import Console
from pydantic_settings import BaseSettings

# Ensure mcp_eval is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer()
console = Console()


def extract_model_info(content: str) -> Dict[str, Dict[str, str]]:
    models = {}
    current_model = None
    lines = content.splitlines()
    for i, line in enumerate(lines):
        class_match = re.match(r"\s*class\s+(\w+)(?:\([^)]+\))?\s*:", line.strip())
        if class_match:
            current_model = class_match.group(1)
            models[current_model] = {"__doc__": ""}
            for j in range(i + 1, min(i + 4, len(lines))):
                doc_match = re.match(r'\s*"""(.+?)"""', lines[j], re.DOTALL)
                if doc_match:
                    models[current_model]["__doc__"] = doc_match.group(1).strip()
                    break
            continue
        if current_model:
            if line and not line.startswith(" ") and not line.startswith("#"):
                current_model = None
                continue
            field_match = re.match(r"\s+(\w+)\s*:", line)
            if field_match:
                field_name = field_match.group(1)
                if field_name in ("model_config", "Config"):
                    continue
                description = None
                field_desc_match = re.search(r'Field\([^)]*description="([^"]+)"', line)
                if field_desc_match:
                    description = field_desc_match.group(1).strip()
                else:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith('"""'):
                            break
                        doc_match = re.match(r'\s*"""(.+?)"""', lines[j], re.DOTALL)
                        if doc_match:
                            description = doc_match.group(1).strip()
                            break
                if description:
                    models[current_model][field_name] = description
    return models


class MockModule:
    pass


def load_settings_class(
    file_path: Path,
) -> Tuple[type[BaseSettings], Dict[str, Dict[str, str]], Dict[str, Any]]:
    # Simply import the module - it's already installed in the environment
    from mcp_eval.config import MCPEvalSettings

    # Extract model info from the source file for documentation
    content = file_path.read_text(encoding="utf-8")
    model_info = extract_model_info(content)

    # Build namespace with all classes from the module for potential future use
    import mcp_eval.config as config_module

    namespace = {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if not name.startswith("_")
    }

    return MCPEvalSettings, model_info, namespace


def apply_descriptions_to_schema(
    schema: Dict[str, Any], model_info: Dict[str, Dict[str, str]]
) -> None:
    if not isinstance(schema, dict):
        return
    if "$defs" in schema:
        for model_name, model_schema in schema["$defs"].items():
            if model_name in model_info:
                doc = model_info[model_name].get("__doc__", "").strip()
                if doc:
                    model_schema["description"] = doc
                if "properties" in model_schema:
                    for field_name, field_schema in model_schema["properties"].items():
                        if field_name in model_info[model_name]:
                            field_schema["description"] = model_info[model_name][
                                field_name
                            ].strip()
    if "properties" in schema and "MCPEvalSettings" in model_info:
        for field_name, field_schema in schema["properties"].items():
            if field_name in model_info["MCPEvalSettings"]:
                field_schema["description"] = model_info["MCPEvalSettings"][
                    field_name
                ].strip()


@app.command()
def generate(
    config_py: Path = typer.Option(
        Path("src/mcp_eval/config.py"),
        "--config",
        "-c",
        help="Path to the MCP‑Eval config.py file",
    ),
    output: Path = typer.Option(
        Path("schema/mcpeval.config.schema.json"),
        "--output",
        "-o",
        help="Output path for the schema file",
    ),
):
    """Generate JSON schema from MCP‑Eval's typed settings (MCPEvalSettings)."""
    if not config_py.exists():
        console.print(f"[red]Error:[/] File not found: {config_py}")
        raise typer.Exit(1)
    try:
        Settings, model_info, namespace = load_settings_class(config_py)
        # Forward refs are already resolved in load_settings_class
        schema = Settings.model_json_schema()
        schema.update(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "MCP‑Eval Configuration Schema",
                "description": "Configuration schema for MCP‑Eval (overlays mcp-agent settings)",
            }
        )
        apply_descriptions_to_schema(schema, model_info)
        output.parent.mkdir(parents=True, exist_ok=True)
        output = output.absolute()
        with open(output, "w") as f:
            json.dump(schema, f, indent=2)
        console.print(f"[green]✓[/] Schema written to: {output}")
        try:
            rel_path = f"./{output.relative_to(Path.cwd())}"
        except ValueError:
            rel_path = str(output)
        vscode_settings = {
            "yaml.schemas": {
                rel_path: [
                    "mcpeval.yaml",
                    "mcpeval.config.yaml",
                    ".mcp-eval/config.yaml",
                    ".mcp-eval.config.yaml",
                    "mcpeval.secrets.yaml",
                    ".mcp-eval/secrets.yaml",
                    ".mcp-eval.secrets.yaml",
                ]
            }
        }
        console.print("\n[yellow]VS Code Integration:[/]")
        console.print("Add this to .vscode/settings.json:")
        console.print(json.dumps(vscode_settings, indent=2))
    except Exception as e:
        console.print(f"[red]Error generating schema:[/] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
