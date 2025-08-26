# MCP-Eval: An Evaluation Framework for MCP Servers

MCP-Eval is a developer-first testing framework for Model Context Protocol (MCP) servers, built on the `mcp-agent` library. It enables you to write clear, concise, and powerful tests to evaluate the performance, reliability, and correctness of your AI agents and the MCP servers they connect to.

## Core Features

- **Task-Based Testing**: Define tests as async functions where an agent performs a task.
- **Automatic Metrics**: Automatically collect detailed metrics on latency, token usage, cost, and tool calls for every test run.
- **Rich Assertions**: A powerful set of assertions designed for AI testing, including:
    - `contains()`: Checks for substrings in responses.
    - `tool_was_called()`: Verifies that a specific tool was used.
    - `tool_arguments_match()`: Checks if a tool was called with the correct arguments.
    - `cost_under()`: Asserts that a test run stays within a defined cost budget.
    - `number_of_steps_under()`: Ensures an agent completes a task efficiently.
    - `objective_succeeded()`: Uses an LLM to verify if the agent's response achieved the overall goal.
    - `plan_is_efficient()`: Uses an LLM to check for redundant or inefficient steps in the agent's execution path.
- **Tool Coverage Reporting**: Automatically calculates the percentage of a server's tools that are exercised by your test suite.
- **Automated Test Generation**: A CLI tool to automatically generate a baseline test suite for any MCP server.
- **Detailed Reports**: Get immediate feedback from rich console reports and generate detailed JSON reports for CI/CD or further analysis.

## Getting Started

### 1. Installation

Install `mcp_eval` and its dependencies. Make sure `mcp-agent` is also installed in your environment.

```bash
pip install -e .
# or with uv
uv pip install -e .
```

## CI/CD with GitHub Actions (uv)

This repo includes ready-to-use GitHub Actions to run MCP‑Eval in CI, publish reports, and comment on pull requests.

- Run action: `.github/actions/mcp-eval/run` (uv-based install + run + reports)
- Upload action: `.github/actions/mcpeval-upload` (artifact, PR comment, optional commit)
- General CI workflow: `.github/workflows/mcpeval.yml`
- Reusable workflow: `.github/workflows/mcpeval-reusable.yml`
- Example workflow: `.github/workflows/mcpeval-example-fetch.yml`

Quick start:

```yaml
name: MCP-Eval CI
on:
  pull_request:
  push:
    branches: [ main ]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/mcp-eval/run
        with:
          python-version: '3.11'
          tests: tests/
          run-args: '-v --max-concurrency 4'
          pr-comment: 'true'
          set-summary: 'true'
          upload-artifacts: 'true'
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Options:

- To publish HTML report, keep the Pages job enabled in `.github/workflows/mcpeval.yml`.
- To commit reports back to the repo, set `commit-report: 'true'` and configure `commit-path`, `commit-branch`, `commit-message`.
- To pass additional environment, use `env`, `env-file`, or `extra-env` inputs.

Troubleshooting in CI:

```bash
uv run mcp-eval doctor --full
```