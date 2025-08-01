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

### Server-Specific Processing
Process only a specific server:
```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files --server-name "fetch_server" --output results.json
```

Process all servers separately (default behavior):
```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files --output results.json
```

### Common Options
- `--model`: Model to use (default: `openai/gpt-4o-mini`)
- `--limit`: Limit examples processed (default: 50)
- `--train-ratio`: Training data ratio (default: 0.8)
- `--optimizer`: Optimizer type (default: `bootstrap`)
- `--server-name`: Process only specific server (optional)

### What Happens When You Run
1. Loads trace file pairs from specified directory
2. **Server Processing**: 
   - If `--server-name` is specified: processes only that server
   - If no server name: automatically detects and processes all servers separately
3. **LLM-as-Judge Task Success Evaluation**: Each trace is evaluated using LLMJudgeSuccess evaluator to determine if the user's task was successfully completed (score ≥ 0.7)
4. Creates training/test split from examples for each server
5. Extracts available tools from trace files for each server
6. Optimizes docstrings using DSPy with successful/failed examples per server
7. Displays original vs optimized docstrings in console
8. Saves results to JSON files (if `--output` specified)

## Server Name Detection

The optimizer automatically extracts server names from trace files by analyzing:

1. **Span Attributes**: Looks for keys containing "server" (e.g., `server_name`, `server.name`, `*.server`)
2. **Span Names**: Extracts from patterns like `ServerName.method`
3. **Filename**: Falls back to trace filename if no server info found in spans

If no server name can be determined, traces are grouped under "unknown".

To see available servers in your trace directory:
```bash
uv run python optimizer_cli.py --trace-directory /path/to/trace/files --server-name "nonexistent"
# This will show: Available servers: [list of detected servers]
```

## Output

### Console Output
- Training/validation example counts
- Original and optimized docstrings for each tool (with `--- ORIGINAL/OPTIMIZED DOCSTRING ---` markers)
- Optimization progress and completion status

### File Output
If `--output` is specified, generates JSON files based on server processing mode:

#### Single Server Mode (with `--server-name`)
1. **Main Results** (`results_<server_name>.json`):
   ```json
   {
     "server_name": "fetch_server",
     "optimizer": "bootstrap", 
     "train_count": 7,
     "val_count": 2,
     "optimization_report": {...}
   }
   ```

2. **Detailed Report** (`<server_name>_optimization_report.json`):
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

#### Multi-Server Mode (no `--server-name`)
1. **Individual Server Results** (`results_<server_name>.json` for each server)
2. **Individual Server Reports** (`<server_name>_optimization_report.json` for each server)  
3. **Combined Results** (`results_combined.json`):
   ```json
   {
     "server1": {
       "server_name": "server1",
       "optimizer": "bootstrap",
       "train_count": 5,
       "val_count": 1,
       "optimization_report": {...}
     },
     "server2": {
       "server_name": "server2", 
       "optimizer": "bootstrap",
       "train_count": 3,
       "val_count": 1,
       "optimization_report": {...}
     }
   }
   ```

## LLM-as-Judge Task Success Evaluation

The optimizer uses the `LLMJudgeSuccess` evaluator to determine whether each trace represents a successfully completed user task. This evaluation:

- **Purpose**: Automatically classifies traces as successful/failed based on task completion
- **Scoring**: Uses a 0.0-1.0 scale with minimum threshold of 0.7 for "success"
- **Evaluation Criteria**: 
  - Did the agent understand the user's original request?
  - Were appropriate actions taken to address the request?
  - Was the task completed without critical errors?
  - Did the conversation conclude with the user's needs met?

### Evaluation Output

Each trace evaluation returns:
```json
{
  "passed": true,
  "score": 0.9,
  "expected": "Task successfully addressed (score >= 0.7)",
  "actual": "score = 0.9, task_success = True",
  "details": {
    "reasoning": "User requested to fetch content from URL. Agent successfully...",
    "confidence": 0.85,
    "user_task_identified": "Fetch content from specific URL",
    "completion_evidence": "Agent retrieved and provided the requested content"
  },
  "error": null
}

```

## Prompt Optimization

MCP-Eval now includes advanced prompt optimization capabilities that can analyze trace data and improve tool descriptions and interactions.

### Trace-Based Analysis

The optimizer can process OTEL trace files to extract:
- User queries from conversation history
- Tool calls and responses 
- Server interactions and performance metrics

### Example: Tool Description Optimization

The optimizer can improve tool descriptions based on actual usage patterns. For example:

**Original Description:**
```
Fetches a URL from the internet and optionally extracts its contents as markdown.

Although originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.
```

**Optimized Description:**
```
Fetches the content of a specified URL from the internet and can extract and format it as markdown. This tool allows users to access the most up-to-date information from web pages, making it useful for retrieving articles, news, or any other web content. Simply provide a valid URL, and the tool will return the content in a markdown format for easy readability.
```

### Usage

```bash
# Run optimization on a directory of trace files
python -m mcp_eval.optimizer.optimizer_cli \
  --trace-directory examples/mcp_server_fetch/test-reports \
  --output optimization_results.json \
  --server-name fetch \
  --model openai/gpt-4o-mini

# Process all servers in the directory
python -m mcp_eval.optimizer.optimizer_cli \
  --trace-directory path/to/trace/files \
  --output results.json \
  --optimizer bootstrap \
  --limit 50

# Use different optimization strategies
python -m mcp_eval.optimizer.optimizer_cli \
  --trace-directory path/to/traces \
  --optimizer mipro \
  --num-epochs 5 \
  --batch-size 16
```

### Command Line Options

- `--trace-directory`: Directory containing trace files (default: examples/mcp_server_fetch/test-reports)
- `--output`: Path to save results JSON file
- `--server-name`: Specific server name to optimize (default: fetch)
- `--model`: Model to use for optimization (default: openai/gpt-4o-mini)
- `--optimizer`: Optimizer type - fewshot, bootstrap, or mipro (default: bootstrap)
- `--limit`: Maximum number of examples to process (default: 50)
- `--train-ratio`: Ratio of data for training vs validation (default: 0.8)
- `--k`: Number of examples for few-shot learning (default: 5)
- `--num-bootstrapped`: Number of bootstrapped example sets (default: 5)
- `--max-demos`: Maximum demonstrations per bootstrapped set (default: 3)
- `--num-epochs`: Number of epochs for MIPRO optimizer (default: 3)
- `--batch-size`: Batch size for MIPRO optimizer (default: 8)
