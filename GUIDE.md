## MCP‑Eval: User and Developer Guide

Think of MCP‑Eval as your “flight simulator” for tool‑using LLMs. You plug in an agent, connect it to real MCP servers (tools), and run realistic scenarios. The framework captures OTEL traces as the single source of truth, turns them into metrics, and gives you expressive assertions for both content and behavior. Two equal goals:

- Evaluate an Agent: Does your agent reason well, call the right tools, recover from errors, stay efficient, and produce quality answers?
- Evaluate an MCP Server: Does your server behave correctly and robustly when driven by an LLM/agent? Are outputs correct, fast, reliable, and easy for the agent to use?

MCP‑Eval is built alongside (and uses) `mcp-agent` for orchestration and tracing. Your MCP servers themselves can be written in any language as long as they implement the MCP spec and can be connected to by the agent.

This guide covers two primary workflows:
1) Evaluating an Agent
2) Evaluating an MCP Server

Along the way, you’ll see multiple test styles (decorator, pytest, dataset, and legacy assertions), configuration options, CLI usage, extensibility, and best practices.


## Installation

- Python 3.10+
- Install in your project (development flow shown):

```bash
pip install -e .
# or using uv
uv pip install -e .
```

API keys (set as needed for your LLM provider):

```bash
export ANTHROPIC_API_KEY=...   # for Anthropic
export OPENAI_API_KEY=...      # for OpenAI
```


## Concepts and Architecture

- **Agent**: An instance of `mcp_agent.agents.agent.Agent` (or an `AugmentedLLM` driving an Agent) that talks to MCP servers/tools
- **MCP Server**: A tool server implementing Model Context Protocol (e.g., `server-fetch`). You attach it by listing its name in an Agent’s `server_names`
- **TestSession**: Unified execution context that configures OTEL tracing, runs the agent, collects spans/metrics, and evaluates assertions
- **Evaluator**: A pluggable check that inspects the response and/or metrics (e.g., content contains X, tool was called Y times, path efficiency, LLM judges)
- **Metrics**: Derived from OTEL traces; include tool calls, latency, iterations, token/cost estimates, success/error rates, etc.
- **Reports**: Console, JSON, Markdown, HTML, plus JSONL OTEL traces saved to disk


## Quick Start (Hello, Fetch!)

Decorator‑style test with the modern unified assertion API:

```python
# tests/test_fetch.py
import mcp_eval
from mcp_eval import task, setup
from mcp_eval import Expect
from mcp_agent.agents.agent import Agent

@setup
def configure():
    # Define servers on the Agent itself (preferred)
    from mcp_agent.agents.agent_spec import AgentSpec
    mcp_eval.use_agent(
        AgentSpec(
            name="fetch_tester",
            instruction="You can fetch URLs and summarize them.",
            server_names=["fetch"],
            # Optionally set per-agent LLM
            # provider="anthropic",
            # model="claude-3-5-haiku-20241022",
        )
    )

@task("Fetch example.com and verify")
async def test_fetch_example(agent, session):
    response = await agent.generate_str("Fetch https://example.com and summarize in one sentence")

    # Immediate content assertion (providing response) + deferred tool usage checks
    await session.assert_that(Expect.content.contains("Example Domain"), name="contains_example_domain", response=response)
    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_called")
    await session.assert_that(Expect.tools.success_rate(min_rate=1.0, tool_name="fetch"), name="fetch_success_rate")
```

Run it:

```bash
mcp-eval run tests/test_fetch.py -v --markdown report.md --html report.html
```


## Part 1) Evaluating an Agent

In this mode you define a single agent configuration and run many scenarios to measure correctness, robustness, and efficiency.

### Defining an Agent (multiple ways)

You can configure the default agent globally via `mcp_eval.use_agent(...)` and refine per‑test via `with_agent(...)`.

1) Programmatic Agent object

```python
from mcp_agent.agents.agent import Agent
import mcp_eval

agent = Agent(name="my_agent", instruction="Be concise.", server_names=["fetch"])  # servers by name
mcp_eval.use_agent(agent)
```

2) Programmatic AugmentedLLM (its `.agent` will be used) — prefer AgentSpec

3) AgentSpec object or by name (discovered by mcp‑agent)

```python
from mcp_agent.agents.agent_spec import AgentSpec
import mcp_eval

spec = AgentSpec(name="Fetcher", instruction="You fetch.", server_names=["fetch"])  # object
mcp_eval.use_agent(spec)

mcp_eval.use_agent("Fetcher")  # by name, resolved from discovered subagents
```

4) (Removed) dict overrides — use AgentSpec or name

5) Per‑test override with `with_agent`

```python
from mcp_eval.core import with_agent
from mcp_agent.agents.agent import Agent
from mcp_eval import task, Expect

@with_agent(Agent(name="custom", instruction="Custom for this test", server_names=["fetch"]))
@task("Custom agent for one test")
async def test_custom(agent, session):
    resp = await agent.generate_str("Fetch https://example.com")
    await session.assert_that(Expect.content.contains("Example Domain"), response=resp)
```

Note: Prefer defining servers on the Agent/AgentSpec via `server_names`. No global project‑wide "use_server" is required.

### Defining AgentSpecs via Configuration (discovery)

In addition to programmatic Agent/LLM/overrides, you can define AgentSpecs declaratively and let `mcp-agent` discover them at app initialization. MCP‑Eval builds on this discovery.

- Root‑level `agents` in `mcp-agent.config.yaml`

```yaml
# mcp-agent.config.yaml
agents:
  - name: Fetcher
    instruction: You can fetch URLs and summarise.
    server_names: ["fetch"]
    # optional: functions: []
```

Reference by name:

```python
import mcp_eval
mcp_eval.use_agent("Fetcher")  # resolves to the discovered AgentSpec
```

- `subagents.inline` in `mcp-agent.config.yaml`

```yaml
subagents:
  enabled: true
  inline:
    - name: Writer
      instruction: You write concise summaries.
      server_names: ["filesystem"]
```

- `subagents.search_paths` (directory discovery)

By default, the app scans these directories for AgentSpec files (YAML/JSON/Markdown):

- .claude/agents
- ~/.claude/agents
- .mcp-agent/agents
- ~/.mcp-agent/agents

Example file `.claude/agents/fetcher.yaml`:

```yaml
name: Fetcher
instruction: You can fetch URLs and summarise.
server_names: ["fetch"]
```

Discovery toggles and patterns:

```yaml
subagents:
  enabled: true
  search_paths: [".claude/agents", "~/.claude/agents"]
  pattern: "**/*.*"  # default
```

Once discovered, reference by name in tests or markers:

```python
mcp_eval.use_agent("Fetcher")
# or in pytest:
# @pytest.mark.mcp_agent("Fetcher")
```

Notes:
- Prefer declaring `server_names` on the AgentSpec so the agent is fully wired without extra code.
- Use discovery (inline/root/dirs) for team‑shared specs and reproducibility.

### Test Styles

You can choose the authoring style that fits your team and codebase.

1) Decorator style (modern evaluators, unified API)

```python
from mcp_eval import task, setup, parametrize, Expect

@setup
def essential_config():
    pass  # optional

@parametrize("url,expect", [
    ("https://example.com", "Example Domain"),
    ("https://httpbin.org/html", "Herman Melville"),
])
@task("Parametrized fetch scenarios")
async def test_fetch(agent, session, url, expect):
    response = await agent.generate_str(f"Fetch {url}")
    await session.assert_that(Expect.tools.was_called("fetch"), name="fetch_called")
    await session.assert_that(Expect.content.contains(expect), name="contains_expected", response=response)
```

2) Pytest style (native fixtures)

```python
# tests/test_pytest_agent.py
import pytest
from mcp_eval import Expect

@pytest.mark.asyncio
async def test_agent_with_pytest(mcp_agent):  # provided by plugin
    response = await mcp_agent.generate_str("Fetch https://example.com")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch"), name="fetch_called")
    await mcp_agent.session.assert_that(Expect.content.contains("Example Domain"), response=response)

# Per-test agent override
from mcp_agent.agents.agent import Agent

@pytest.mark.asyncio
@pytest.mark.mcp_agent(Agent(name="custom", instruction="You fetch", server_names=["fetch"]))
async def test_with_custom_agent(mcp_agent):
    assert "Example" in await mcp_agent.generate_str("Fetch https://example.com")
```

Available pytest markers/fixtures:
- **fixtures**: `mcp_session`, `mcp_agent`
- **markers**: `@pytest.mark.mcp_agent(<Agent|name>)` plus your own (e.g., `network`, `slow`)

3) Dataset style (systematic suites)

```python
from mcp_eval import Case, Dataset
from mcp_eval import ToolWasCalled, ResponseContains, LLMJudge

cases = [
    Case(
        name="fetch_example",
        inputs="Fetch https://example.com",
        expected_output="Example Domain",
        evaluators=[ToolWasCalled("fetch"), ResponseContains("Example Domain")],
    ),
    Case(
        name="fetch_json",
        inputs="Fetch https://httpbin.org/json and summarize",
        evaluators=[ToolWasCalled("fetch"), LLMJudge("Summarizes JSON content")],
    ),
]

dataset = Dataset(name="Agent Basic Suite", cases=cases, server_name="fetch", agent_spec="Fetcher")

async def task_func(inputs: str, agent, session) -> str:
    return await agent.generate_str(inputs)

report = await dataset.evaluate(task_func)
report.print(include_input=True, include_output=True, include_scores=True)
```

4) Legacy assertions style (explicit session)

```python
from mcp_eval import task
from mcp_eval import assertions as A

@task("Legacy assertions example")
async def test_legacy(agent, session):
    response = await agent.generate_str("Fetch https://example.com")
    A.tool_was_called(session, "fetch")
    A.contains(session, response, "Example Domain")
    A.completed_within(session, 3)
```

### Unified Assertion API (Expect + await session.assert_that)

- Immediate vs deferred: If you pass `response=...`, synchronous evaluators run immediately and async ones are scheduled; if you don’t pass a response, the evaluator is deferred to run at session end with full metrics.
- Force timing with `when="now" | "end"` if needed.

#### When does an evaluator run immediately vs at the end?

Each evaluator declares whether it needs the full, final metrics (from OTEL traces) before it can run reliably. The flag is `requires_final_metrics` on the evaluator class.

- Runs immediately (requires_final_metrics=False):
  - `Expect.content.contains`, `Expect.content.not_contains`, `Expect.content.regex`
  - `Expect.judge.llm`, `Expect.judge.multi_criteria`
  These operate on the provided `response` and do not need the final session metrics.

- Defers automatically to the end (requires_final_metrics=True):
  - `Expect.tools.was_called`, `Expect.tools.called_with`, `Expect.tools.count`, `Expect.tools.success_rate`, `Expect.tools.failed`, `Expect.tools.output_matches`, `Expect.tools.sequence`
  - `Expect.performance.max_iterations`, `Expect.performance.response_time_under`
  - `Expect.path.efficiency`
  These rely on the complete session metrics (tool calls, iteration count, latency, etc.).

You can still override with `when="now"` (force immediate) or `when="end"` (force deferral), but in most cases you can let the framework decide.

Examples:

```python
# Immediate content check
await session.assert_that(Expect.content.contains("Example Domain"), response=resp)

# Schedules an async LLM judge now (no await needed); it will finish before deferred checks
await session.assert_that(Expect.judge.llm("Accurate summary", min_score=0.8), response=resp)

# Defers automatically since it needs final metrics
await session.assert_that(Expect.tools.was_called("fetch"))

# Explicitly force timing if you really need to
await session.assert_that(Expect.tools.was_called("fetch"), when="end")
```

Key assertion families (via `from mcp_eval import Expect`):

- **Content**
  - `Expect.content.contains(text, case_sensitive=False)`
  - `Expect.content.not_contains(text, case_sensitive=False)`
  - `Expect.content.regex(pattern, case_sensitive=False)`
- **Tools**
  - `Expect.tools.was_called(tool_name, min_times=1)`
  - `Expect.tools.called_with(tool_name, arguments: dict)`
  - `Expect.tools.count(tool_name, expected_count)`
  - `Expect.tools.success_rate(min_rate, tool_name=None)`
  - `Expect.tools.failed(tool_name)`
  - `Expect.tools.output_matches(tool_name, expected_output, field_path=None, match_type="exact|contains|regex|partial", case_sensitive=True, call_index=-1)`
  - `Expect.tools.sequence(["fetch", "write"], allow_other_calls=False)`
- **Performance**
  - `Expect.performance.max_iterations(max_iterations)`
  - `Expect.performance.response_time_under(ms)`
- **Judge**
  - `Expect.judge.llm(rubric, min_score=0.8, include_input=False, require_reasoning=True)`
  - `Expect.judge.multi_criteria(criteria: dict|EvaluationCriterion[], aggregate_method="weighted", require_all_pass=False, include_confidence=True, use_cot=True, model=None)`
- **Path**
  - `Expect.path.efficiency(optimal_steps=None, expected_tool_sequence=None, allow_extra_steps=0, penalize_backtracking=True, penalize_repeated_tools=True, tool_usage_limits=None, default_tool_limit=1)`

Examples:

```python
# Match nested fields in a tool output (e.g., content[0].text contains a phrase)
await session.assert_that(
    Expect.tools.output_matches(
        tool_name="fetch",
        expected_output=r"use.*examples",
        match_type="regex",
        case_sensitive=False,
        field_path="content.0.text",
    ),
    name="fetch_output_match",
)

# Require an exact tool call sequence
await session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=True))

# Multi-criteria judging
criteria = {
    "accuracy": "Factual correctness",
    "completeness": "Covers key points",
    "clarity": "Clear and well-structured",
}
await session.assert_that(Expect.judge.multi_criteria(criteria, require_all_pass=False), response=response)
```

### Inspecting Metrics and Spans

At any time you can access metrics and the span tree:

```python
metrics = session.get_metrics()
span_tree = session.get_span_tree()

assert metrics.total_duration_ms > 0
assert any(call.name == "fetch" for call in metrics.tool_calls)
```

The span tree supports advanced analyses (e.g., rephrasing loop detection, inefficient paths) for deeper debugging and optimization.


## Part 2) Evaluating an MCP Server

Premise: An MCP server is meant to be used by LLMs/agents. The best way to test it is to connect it to an agent and exercise realistic flows. Your server can be written in any language; as long as it speaks MCP, `mcp-agent` can connect to it and MCP‑Eval can test it.

### 1) Define an Agent for the Server

Use any approach from Part 1. There are multiple ways to connect servers:

- Put your server definition in `mcp-agent.config.yaml` under `mcp.servers` and reference it by name in the Agent’s `server_names` (recommended):

```yaml
mcp:
  servers:
    my_server:
      transport: stdio
      command: "python"
      args: ["path/to/my_server.py"]
      env: { KEY: VALUE }
```

Then either:

```python
from mcp_agent.agents.agent import Agent
import mcp_eval

mcp_eval.use_agent(Agent(name="tester", instruction="You test.", server_names=["my_server"]))
```

or refer to a discovered AgentSpec that already lists `server_names`.

- Use config‑discovered AgentSpecs (inline or from `subagents.search_paths`) that include `server_names`. You can reference them by name with `mcp_eval.use_agent("SpecName")` or `@pytest.mark.mcp_agent("SpecName")`.

- Optional helper: `mcp_eval.use_server("my_server")` sets a default server list used when an Agent is implicitly constructed. Prefer explicit `server_names` on the Agent/Spec in new code.

```python
import mcp_eval

from mcp_agent.agents.agent_spec import AgentSpec
mcp_eval.use_agent(
    AgentSpec(
        name="server_tester",
        instruction="Use the server tools effectively and report results.",
        server_names=["my_server"],
        # Optionally per-agent provider/model
        # provider="anthropic",
        # model="claude-3-5-haiku-20241022",
    )
)
```

### 2) Define Configuration

Configure servers and providers in your `mcp-agent.config.yaml` (preferred). Example:

```yaml
# mcp-agent.config.yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      env: { UV_NO_PROGRESS: "1" }
    my_server:
      command: "python"
      args: ["path/to/my_server.py"]

otel:
  enabled: true
  exporters: ["file"]

# Provider sections (e.g., Anthropic/OpenAI) are read from env/this file
```

MCP‑Eval also has typed evaluation settings layered on top of the mcp‑agent settings, including judge, metrics, reporting, and execution knobs. These are auto‑loaded from your mcp‑agent config; a separate `mcpeval.yaml` is optional and used mainly for displaying config in reports.

Note: Choose the LLM by setting `provider` and `model` globally and/or per AgentSpec. MCP‑Eval prefers per‑AgentSpec provider/model over global defaults.

### 3) Write Tests for Your Server

You can reuse every style from Part 1; here are patterns that focus on server evaluation:

- **Correctness and content**
  - `Expect.content.contains(...)`, `Expect.content.regex(...)`
  - `Expect.judge.llm(...)` or `Expect.judge.multi_criteria(...)`
- **Tool behavior**
  - `Expect.tools.was_called(...)`, `called_with(...)`, `count(...)`
  - `Expect.tools.success_rate(min_rate=...)` and `failed(tool_name)`
  - `Expect.tools.output_matches(...)` for structured outputs
- **Efficiency and performance**
  - `Expect.performance.response_time_under(ms)` and `max_iterations(...)`
  - `Expect.path.efficiency(...)` and `Expect.tools.sequence([...])`

Dataset‑driven server coverage:

```python
from mcp_eval import Case, Dataset, ToolWasCalled, ResponseContains, LLMJudge

cases = [
  Case(
    name="basic_op",
    inputs="Use the server to perform the basic operation",
    evaluators=[ToolWasCalled("my_tool"), ResponseContains("success")],
  ),
]

dataset = Dataset(name="My Server Suite", cases=cases, server_name="my_server")

async def my_server_task(inputs: str, agent, session):
    return await agent.generate_str(inputs)

report = await dataset.evaluate(my_server_task)
report.print(include_input=True, include_output=True)
```

Tip: Add a few golden path tests using `Expect.tools.sequence([...])` and path efficiency checks; then add fuzzier quality checks with LLM judges.


## CLI

Top‑level CLI is exposed as `mcp-eval` (also `mcpeval`/`mcp_eval`). Key commands:

```bash
# Discover and run both decorator tests and datasets under a path
mcp-eval run tests/ -v --json results.json --markdown results.md --html results.html

# Run a single test function (pytest style selector)
mcp-eval run tests/test_fetch.py::test_fetch_example

# Run a dataset file directly
mcp-eval run dataset datasets/basic_fetch_dataset.yaml

# Initialize a new test project (creates tests/ and datasets/)
mcp-eval init . --template basic

# Generate a starter dataset for a server (prototype)
mcp-eval generate my_server --output generated_tests.yaml --n-examples 10
```

Console output includes live progress, pytest‑style summaries, and failure details. Reports can be emitted in JSON/Markdown/HTML. Individual test artifacts (including OTEL traces) are saved under `./test-reports` by default.


## Reporting and Artifacts

- **JSON**: Per‑test details and dataset summaries
- **Markdown**: Human‑readable combined report (optionally embeds configuration)
- **HTML**: Combined interactive report with filters
- **OTEL traces**: JSONL trace files for each test, used as the single source of truth for metrics

Reporting directory and toggles can be configured; default output dir is `./test-reports`.


## Extensibility

### Custom Evaluators

```python
from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult

class MyCustomEvaluator(SyncEvaluator[str, str]):
    def evaluate_sync(self, ctx: EvaluatorContext[str, str]) -> EvaluatorResult:
        ok = "ready" in ctx.output.lower()
        return EvaluatorResult(passed=ok, expected="contains 'ready'", actual=ctx.output)

from mcp_eval.evaluators import register_evaluator
register_evaluator("MyCustomEvaluator", MyCustomEvaluator)
```

Use it anywhere you would use an `Expect.*` evaluator:

```python
await session.assert_that(MyCustomEvaluator(), name="custom_check", response=response)
```

### Custom Metrics

You can register additional metric processors via `mcp_eval.metrics.register_metric(name, processor)` and/or post‑process the OTEL‑derived `TestMetrics` for custom dashboards.


## Configuration Details

MCP‑Eval extends `mcp-agent` settings. It auto‑loads your `mcp-agent.config.yaml` / `mcp_agent.config.yaml` and merges any `mcp-agent.secrets.yaml` if present (either alongside the config or discoverable via standard locations). Key sections:

- **Servers** (`mcp.servers`): CLI command, args, env for each server
- **Agents** (`agents`): Root‑level AgentSpec list
- **Subagents** (`subagents`): Inline AgentSpecs and discovery from directories (e.g., `.claude/agents`)
- **OTEL** (`otel`): Enable and choose exporters; file exporter is used for trace‑backed metrics
- **Providers**: Sections for Anthropic/OpenAI/etc. credentials (env or secrets file)
- **Evaluation knobs** (typed under MCP‑Eval): judge, metrics, reporting, execution

Secrets: Yes, `mcp-agent.secrets.yaml` is merged

```yaml
# mcp-agent.secrets.yaml
anthropic:
  api_key: "sk-ant-..."
openai:
  api_key: "sk-openai-..."
```

- You can use environment variables or the secrets file (or both). MCP‑Eval will merge secrets on top of the base config, mirroring `mcp-agent` behavior.
- Keep secrets out of source control; prefer a local secrets file or CI secrets management.

About mcpeval.yaml

- Optional; used primarily for display in combined reports (Markdown/HTML). Typed configuration is sourced from `mcp-agent` files.
- You may keep metadata (e.g., run labels) here if you want them in reports.

Programmatic helpers (already shown): `use_agent(...)`.

### MCP‑Eval configuration files (mcpeval.*)

In addition to `mcp-agent.config.yaml`, MCP‑Eval can read and overlay its own configuration files. Discovery checks the current directory and parents (with a home fallback under `~/.mcp-eval/`):

- mcpeval.yaml | mcpeval.yml
- mcpeval.config.yaml | mcpeval.config.yml
- .mcp-eval/config.yaml | .mcp-eval/config.yml
- .mcp-eval.config.yaml | .mcp-eval.config.yml

Secrets counterparts (merged similarly to mcp‑agent secrets):

- mcpeval.secrets.yaml | mcpeval.secrets.yml
- .mcp-eval/secrets.yaml | .mcp-eval/secrets.yml
- .mcp-eval.secrets.yaml | .mcp-eval.secrets.yml

Layering/precedence (later overrides earlier fields):

1. mcp-agent.config.yaml (+ mcp-agent.secrets.yaml) — servers, providers, subagents, etc.
2. mcpeval.* (+ mcpeval.secrets.*) — evaluation-only knobs (judge, metrics, reporting, execution), or to override/extend agent defaults
3. Explicit path passed to the loader (highest precedence)

Tip: Keep server/provider definitions in the mcp‑agent config; use mcpeval.* for test-suite specific knobs like reporting directories, judge min scores, execution limits, etc.

### Programmatic configuration

You can configure MCP‑Eval at runtime, bypassing the file discovery:

```python
from mcp_eval.config import set_settings, MCPEvalSettings, load_config

# Minimal overlay via dict
set_settings({
  "reporting": {"output_dir": "./ci-reports", "formats": ["json", "markdown", "html"]},
  "execution": {"timeout_seconds": 120}
})

# Or provide a fully-typed settings object
settings = MCPEvalSettings(reporting={"output_dir": "./local-reports"})
set_settings(settings)

# Or explicitly load a specific config file (path can be mcpeval.yaml or mcp-agent.config.yaml)
load_config(config_path="./.mcp-eval/config.yaml")
```

This is useful for CI pipelines or scenarios where you want to override config at test start without modifying files.


## Best Practices

- Write task‑oriented prompts that clearly call for tools and expected outputs
- Keep tests deterministic: stable prompts, models, and explicit criteria (use LLM judges for fuzzier quality)
- Use parametrization to cover variations (URLs, inputs, error scenarios)
- Separate correctness vs. performance assertions so you can gate them differently (e.g., run perf in nightly)
- Tag network/slow tests with pytest markers
- When a test fails, inspect the per‑test JSON and OTEL trace to understand tool path, errors, and token/cost usage


## Troubleshooting

- No tools are recorded: Ensure the Agent’s `server_names` includes your server and that it’s discoverable in `mcp-agent` config
- LLM provider failures: Confirm `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` and any provider settings in the config
- Long test runs or timeouts: Use `Expect.performance.response_time_under(...)`, check network reachability, consider `execution.timeout_seconds`
- Inconsistent outputs: Tighten prompts, reduce creativity settings in your LLM, or use more objective evaluators (e.g., `EqualsExpected`, `IsInstance`)


## Where To Look in This Repo

- Examples: `examples/mcp_server_fetch/tests` shows decorator, pytest, dataset, and advanced usage
- Discovery‑friendly evaluators catalog: `src/mcp_eval/catalog.py` (import with `from mcp_eval import Expect`)
- Evaluators: `src/mcp_eval/evaluators/*`
- Session and metrics: `src/mcp_eval/session.py`, `src/mcp_eval/metrics.py`
- CLI runner: `src/mcp_eval/runner.py` and `src/mcp_eval/cli/`


## Summary

MCP‑Eval gives you one cohesive way to evaluate both agents and MCP servers:
- For agents: define how the agent should act, run diverse scenarios, and assert behavior, tool usage, and quality
- For servers: connect them to an agent and validate correctness, robustness, and efficiency of tool interactions

Use the style you prefer (decorators, pytest, dataset), lean on `Expect.*` for clear assertions, and let the framework’s OTEL‑backed metrics and reports power your insights.
