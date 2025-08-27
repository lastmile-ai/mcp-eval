# MCP-Eval Sample Tests

This directory contains example test cases demonstrating how to use the MCP-Eval framework to test MCP servers.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager (for dependency isolation)

## Installation

### Installing uv

If you don't have `uv` installed, you can install it using:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Setting up the environment

1. Navigate to the sample directory:
```bash
cd sample/
```

2. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

This will install mcp-eval from the parent directory along with all its compatible dependencies.

## Running the Example Tests

After installing dependencies with `uv pip install`, you can run commands directly:

```bash
# Run the example test file
mcp_eval run usage_example.py
```

## Defining agents

- Prefer defining agents declaratively via mcp-agent AgentSpecs in your config or in the `subagents.search_paths`, then pass `agent_spec` (name or object) in `test_session`.
- For complex programmatic agents, pass `initial_agent` and optionally `initial_llm` into `test_session`.

Example AgentSpec YAML:

```yaml
agents:
  - name: Fetcher
    instruction: You can fetch URLs and summarise content concisely.
    server_names: ["fetch"]
```

Usage (decorator style):

```python
async with test_session("spec_based", agent_spec="Fetcher") as agent:
    result = await agent.generate_str("Fetch https://example.com")
```

Usage (pytest markers):

```python
import pytest

@pytest.mark.mcp_agent("Fetcher")
async def test_fetch_with_spec_name(mcp_agent, mcp_session):
    resp = await mcp_agent.generate_str("Fetch https://example.com")

@pytest.mark.mcp_agent(custom_agent_object)
async def test_fetch_with_custom_agent(mcp_agent, mcp_session):
    resp = await mcp_agent.generate_str("Fetch https://example.com")
```


