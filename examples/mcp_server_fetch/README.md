# MCP Fetch Server Test Suite

A comprehensive test suite for the MCP fetch server using the mcp-eval framework. This project demonstrates all testing approaches supported by mcp-eval: pytest integration, legacy assertions, modern decorators, and dataset-driven evaluation.

## Setup

1. **Install dependencies:**
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

### Pytest Integration Tests
```bash
# Run all pytest tests
pytest tests/test_pytest_style.py -v

# Run specific test
pytest tests/test_pytest_style.py::test_basic_fetch_with_pytest -v

# Run with network marker
pytest -m network tests/test_pytest_style.py

# Skip slow tests
pytest -m "not slow" tests/
```

### Legacy Assertions Style
```bash
# Run with mcp-eval CLI
mcp-eval run tests/test_assertions_style.py

# Or individual test
python -c "
import asyncio
from tests.test_assertions_style import test_basic_fetch_assertions
asyncio.run(test_basic_fetch_assertions())
"
```

### Modern Decorator Style
```bash
# Run with mcp-eval CLI
mcp-eval run tests/test_decorator_style.py

# With verbose output
mcp-eval run tests/test_decorator_style.py --verbose
```

### Dataset Evaluation
```bash
# Run dataset evaluation
python tests/test_dataset_style.py

# Run from YAML dataset
mcp-eval dataset datasets/basic_fetch_dataset.yaml

# Generate reports
mcp-eval run tests/test_dataset_style.py --json=results.json --markdown=results.md
```

### Advanced Features
```bash
# Run advanced analysis tests
mcp-eval run tests/test_advanced_features.py

# With detailed reporting
mcp-eval run tests/test_advanced_features.py --json=advanced_results.json
```

### Run All Tests
```bash
# Run everything with mcp-eval
mcp-eval run tests/

# Run everything with pytest
pytest tests/ -v

# Mixed approach
mcp-eval run tests/test_decorator_style.py tests/test_dataset_style.py
pytest tests/test_pytest_style.py
```

## Test Categories

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

The test suite uses `mcp-eval.yaml` for configuration:

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
from mcp-eval.evaluators.base import SyncEvaluator

class CustomFetchEvaluator(SyncEvaluator):
    def evaluate_sync(self, ctx):
        # Custom evaluation logic
        return True

# Register the evaluator
from mcp-eval.evaluators.builtin import register_evaluator
register_evaluator('CustomFetchEvaluator', CustomFetchEvaluator)
```

### Golden Path Analysis

Update `golden_paths/fetch_paths.json` to define expected tool sequences for different scenarios.

## Troubleshooting

### Common Issues

1. **MCP server not found**: Ensure `uvx mcp-server-fetch` works
2. **API key errors**: Set your LLM provider API key
3. **Network tests failing**: Check internet connectivity
4. **Slow tests**: Use `-m "not slow"` to skip

### Debug Mode

```bash
# Enable debug logging
mcp-eval_LOG_LEVEL=DEBUG mcp-eval run tests/

# Inspect specific test
mcp-eval run tests/test_decorator_style.py::test_basic_fetch_decorator --verbose
```

This test suite serves as both a comprehensive evaluation of the MCP fetch server and a demonstration of mcp-eval's capabilities across all testing paradigms.