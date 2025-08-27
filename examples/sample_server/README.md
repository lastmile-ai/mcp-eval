# Sample MCP Server Test Suite

A small, self-contained example MCP server and test suite. This shows how to:

- Run a toy MCP server (`sample_server.py`)
- Evaluate it with mcp-eval using modern decorators (`sample_server.eval.py`)
- Try a lightweight usage example that exercises fetch + analysis (`usage_example.py`)

## Setup

1. Install uv (recommended):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Or with pip:
   ```bash
   pip install -e .
   ```

2. Ensure the MCP Fetch server is available (used by some examples):
   ```bash
   uvx mcp-server-fetch --help
   ```

3. Configure your LLM provider key (example for Anthropic):
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

4. Change into this example directory:
   ```bash
   cd examples/sample_server
   ```

## Running the Sample Server (optional)

You can run the toy server directly if you want to poke it:
```bash
uv run python sample_server.py
```
It provides two tools: `get_current_time` and `summarize_text`.

## Running Tests and Examples

- Run the sample server evaluation (decorator style):
  ```bash
  uv run mcp-eval run sample_server.eval.py
  ```

- Run the usage example (dataset-style helpers + assertions):
  ```bash
  uv run mcp-eval run usage_example.py
  ```

- Generate reports (arguments before test path):
  ```bash
  uv run mcp-eval run --json results.json --markdown results.md sample_server.eval.py
  ```

- With pytest (if you add pytest-style tests here):
  ```bash
  uv run pytest -v
  ```

> Tip: The first `uv run` may take longer while it prepares an isolated environment. Subsequent runs are faster.

## Quick Start

```bash
# 1) Navigate to this example
cd examples/sample_server

# 2) Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# 3) Run a simple evaluation
uv run mcp-eval run sample_server.eval.py
```

## Defining Agents

You can define agents in code (as shown in `sample_server.eval.py`) or via YAML files. Example YAML definition:

```yaml
agents:
  - name: Fetcher
    instruction: You can fetch URLs and summarise content concisely.
    server_names: ["fetch"]
```

Decorator-style usage:

```python
from mcp_eval import task, setup
from mcp_eval.session import TestAgent, TestSession

@task("Basic fetch")
async def test_basic(agent: TestAgent, session: TestSession):
    response = await agent.generate_str("Fetch https://example.com")
```

## Configuration

This example primarily uses programmatic configuration inside `sample_server.eval.py`. If you prefer a file-based setup, bootstrap a project and copy a starter config:

```bash
# From your own project directory
mcp-eval init --template sample
```

This creates `mcpeval.yaml` and a `mcpeval.secrets.yaml.example` you can adapt. The packaged sample templates are also available under the installed package at `mcp_eval.data.sample`.

## Troubleshooting

- MCP server not found: ensure `uvx mcp-server-fetch --help` works (for examples that use fetch) and `uv run python sample_server.py` runs in this directory.
- API key errors: set your LLM provider key (e.g., `ANTHROPIC_API_KEY`).
- Network-related failures: if you add pytest tests with markers, you can re-run with `-m "not network"`.
- For verbose debugging:
  ```bash
  uv run mcp-eval run --verbose sample_server.eval.py
  ```

## Related

- Fetch server test suite for a fuller example: `examples/mcp_server_fetch/`
- Main project documentation: see files in `docs/` and the project `README.md`
