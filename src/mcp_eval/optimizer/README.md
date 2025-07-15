# Tool Docstring Optimizer

Optimizes tool docstrings using DSPy based on trace data to improve tool selection accuracy.

## Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Required Arguments
- `--trace-directory`: Path to directory containing `*_trace.jsonl` and `*.json` file pairs

### Basic Run
```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files
```

### With Output Files
```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files --output results.json
```

### Common Options
- `--model`: Model to use (default: `openai/gpt-4o-mini`)
- `--limit`: Limit examples processed (default: 50)
- `--train-ratio`: Training data ratio (default: 0.8)
- `--optimizer`: Optimizer type (default: `bootstrap`)

### What Happens When You Run
1. Loads trace file pairs from specified directory
2. Creates training/test split from examples
3. Extracts available tools from trace files
4. Optimizes docstrings using DSPy with successful/failed examples
5. Displays original vs optimized docstrings in console
6. Saves results to JSON files (if `--output` specified)

## Output

### Console Output
- Training/validation example counts
- Original and optimized docstrings for each tool (with `--- ORIGINAL/OPTIMIZED DOCSTRING ---` markers)
- Optimization progress and completion status

### File Output
If `--output` is specified, generates two JSON files:

1. **Main Results** (`results.json`):
   ```json
   {
     "optimizer": "bootstrap",
     "train_count": 7,
     "val_count": 2,
     "optimization_report": {...}
   }
   ```

2. **Detailed Report** (`results_report.json`):
   ```json
   {
     "tool_name": {
       "original_docstring": "...",
       "input_schema": {...},
       "successful_examples": [...],
       "failed_examples": [...],
       "optimized_docstring": "...",
       "optimization_attempted": true
     }
   }
   ```