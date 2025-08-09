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