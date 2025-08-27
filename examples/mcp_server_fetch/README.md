# MCP Fetch Server Test Suite

A comprehensive test suite for the MCP fetch server using the mcp-eval framework. This project demonstrates all testing approaches supported by mcp-eval: pytest integration, legacy assertions, modern decorators, and dataset-driven evaluation.

## Setup

1. **Install uv (if you haven't already):**
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   > **Note:** With uv, you don't need to manually install dependencies - `uv run` will handle this automatically!

   Alternatively, install with pip:
   ```bash
   pip install -e .
   ```

2. **Ensure MCP fetch server is available:**
   ```bash
   uvx mcp-server-fetch --help
   ```

3. **Configure your LLM API keys** (for Anthropic):
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

## Running Tests

> **Note:** All examples below show both `uv run` (recommended) and direct execution methods. Using `uv run` ensures dependencies are properly managed.

### Pytest Integration Tests
```bash
# Run all pytest tests (with uv)
uv run pytest tests/test_pytest_style.py -v

# Run specific test
uv run pytest tests/test_pytest_style.py::test_basic_fetch_with_pytest -v

# Run with network marker
uv run pytest -m network tests/test_pytest_style.py

# Skip slow tests
uv run pytest -m "not slow" tests/

# Or without uv:
pytest tests/test_pytest_style.py -v
```

### Legacy Assertions Style
```bash
# Run with mcp-eval CLI (using uv)
uv run mcp-eval run tests/test_assertions_style.py

# Or individual test
uv run python -c "
import asyncio
from tests.test_assertions_style import test_basic_fetch_assertions
asyncio.run(test_basic_fetch_assertions())
"

# Without uv:
mcp-eval run tests/test_assertions_style.py
```

### Modern Decorator Style
```bash
# Run with mcp-eval CLI (using uv)
uv run mcp-eval run tests/test_decorator_style.py

# With verbose output
uv run mcp-eval run tests/test_decorator_style.py --verbose

# Without uv:
mcp-eval run tests/test_decorator_style.py --verbose
```

### Dataset Evaluation
```bash
# Run dataset evaluation (with uv)
uv run python tests/test_dataset_style.py

# Run from YAML dataset
uv run mcp-eval run dataset datasets/basic_fetch_dataset.yaml

# Generate reports (note: arguments go before the test path)
uv run mcp-eval run --json results.json --markdown results.md tests/test_dataset_style.py

# Without uv:
python tests/test_dataset_style.py
mcp-eval run --json results.json tests/test_dataset_style.py
```

### Advanced Features
```bash
# Run advanced analysis tests (with uv)
uv run mcp-eval run tests/test_advanced_features.py

# With detailed reporting (arguments before test path)
uv run mcp-eval run --json advanced_results.json tests/test_advanced_features.py

# Without uv:
mcp-eval run --json advanced_results.json tests/test_advanced_features.py
```

### Run All Tests
```bash
# Run everything with mcp-eval (using uv)
uv run mcp-eval run tests/

# Run everything with pytest
uv run pytest tests/ -v

# Generate comprehensive reports
uv run mcp-eval run --json results.json --html report.html tests/

# Mixed approach
uv run mcp-eval run tests/test_decorator_style.py tests/test_dataset_style.py
uv run pytest tests/test_pytest_style.py

# Without uv:
mcp-eval run tests/
pytest tests/ -v
```

## Quick Start Example

Here's a complete example of running a test from scratch:

```bash
# 1. Clone and navigate to this example
cd examples/mcp_server_fetch

# 2. Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Run a simple test (uv will handle dependencies automatically)
uv run mcp-eval run tests/test_decorator_style.py

# Expected output:
# ✓ test_basic_fetch_decorator [4.123s]
#   └─ Test basic URL fetching with decorator style
#      ├─ fetch_tool_called: ✓ Tool 'fetch' was called at least 1 time(s)
#      ├─ contains_domain_text: ✓ Response contains 'Example Domain'
#      └─ fetch_success_rate: ✓ Tool success rate is at least 100.0%
```

> **Tip:** The first run with `uv run` may take a moment as it sets up dependencies. Subsequent runs will be faster!

## Test Categories

### Unified assertions and discovery catalog

The examples use a single assertion entry point so you don't have to choose between immediate vs deferred checks. The framework decides based on whether you provide a `response`.

We also expose a discovery-friendly catalog `Expect` for IntelliSense-driven exploration:

```python
from mcp_eval import Expect

response = await agent.generate_str("Fetch https://example.com")

# Content checks (immediate)
await session.assert_that(Expect.content.contains("Example Domain"), response=response)

# Tool checks (deferred)
await session.assert_that(Expect.tools.was_called("fetch"))

# LLM judge (async immediate; no await required)
await session.assert_that(Expect.judge.llm("Summarizes the page accurately", min_score=0.8), response=response)
```

Optionally override timing with `when="now" | "end"`.

### Basic Functionality
- URL fetching
- Content extraction
- Markdown conversion
- Error handling

### Content Processing
- HTML to markdown conversion
- JSON content handling
- Raw content fetching
- Large content chunking

### Error Scenarios
- Invalid URLs
- Network timeouts
- HTTP errors
- Recovery mechanisms

### Performance Testing
- Response times
- Concurrent fetching
- Resource efficiency
- Tool call optimization

### Advanced Analysis
- Span tree analysis
- LLM rephrasing loop detection
- Tool path efficiency
- Error recovery sequences

## Configuration

The test suite uses `mcpeval.yaml` for configuration ([view file](../../mcpeval.yaml)):

- **Server**: MCP fetch server via uvx
- **Agents**: Different agent configurations for various test types
- **Judge**: Enhanced LLM judge with structured output
- **Metrics**: Comprehensive metrics collection
- **Golden Paths**: Expected tool call sequences

## Results and Reporting

Tests generate multiple output formats:

- **Console output**: Real-time test results
- **JSON reports**: Detailed results for analysis
- **Markdown reports**: Human-readable summaries
- **Trace files**: OpenTelemetry traces for debugging

## Extending Tests

### Adding New Test Cases

1. **Pytest style**: Add to `test_pytest_style.py`
2. **Decorator style**: Add to `test_decorator_style.py` 
3. **Dataset style**: Add cases to `test_dataset_style.py` or YAML files
4. **Custom evaluators**: Create in separate module and register

### Custom Evaluators

```python
from mcp_eval.evaluators.base import SyncEvaluator

class CustomFetchEvaluator(SyncEvaluator):
    def evaluate_sync(self, ctx):
        # Custom evaluation logic
        return True

# Register the evaluator
from mcp_eval.evaluators import register_evaluator
register_evaluator('CustomFetchEvaluator', CustomFetchEvaluator)
```

See [evaluators module](../../src/mcp_eval/evaluators/) for more examples.

### Golden Path Analysis

Update `golden_paths/fetch_paths.json` to define expected tool sequences for different scenarios.

## Troubleshooting

### Common Issues

1. **MCP server not found**: Ensure `uvx mcp-server-fetch` works
2. **API key errors**: Set your LLM provider API key
3. **Network tests failing**: Check internet connectivity
4. **Slow tests**: Use `-m "not slow"` to skip

### Debug Mode

# Inspect specific test with verbose output
uv run mcp-eval run --verbose tests/test_decorator_style.py

# View available commands and options
uv run mcp-eval --help
uv run mcp-eval run --help
```

## Expected Test Results

When running the full test suite, you should see output similar to:

```
✓ test_basic_fetch_decorator [4.123s]
✓ test_content_extraction_decorator [6.287s]
✓ test_error_handling_decorator [3.142s]
✓ test_concurrent_fetching_decorator [8.654s]
✓ test_markdown_conversion_decorator [5.432s]

Results: 5 passed, 0 failed, 0 skipped
```

Note: Some advanced tests may fail depending on network conditions or API availability.

## Related Documentation

- [Assertions Guide](../../docs/assertions.mdx) - Learn about the Expect catalog
- [Common Workflows](../../docs/common-workflows.mdx) - Practical testing patterns
- [Configuration Reference](../../docs/configuration.mdx) - Full configuration options
- [Main README](../../README.md) - Project overview

This test suite serves as both a comprehensive evaluation of the MCP fetch server and a demonstration of mcp-eval capabilities across all testing paradigms.