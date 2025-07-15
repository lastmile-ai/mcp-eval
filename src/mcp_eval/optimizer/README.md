# Tool Docstring Optimizer

Optimizes tool docstrings using DSPy based on trace data to improve tool selection accuracy.

## Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files --output results.json
```

## Output

- `results.json`: Main optimization results
- `results_report.json`: Detailed report with original/optimized docstrings for each tool

The tool shows before/after docstrings during optimization and generates a JSON report containing the original docstring, input schema, successful/failed examples, and optimized docstring for each tool.