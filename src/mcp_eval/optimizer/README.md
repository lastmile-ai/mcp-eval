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
3. Creates training/test split from examples for each server
4. Extracts available tools from trace files for each server
5. Optimizes docstrings using DSPy with successful/failed examples per server
6. Displays original vs optimized docstrings in console
7. Saves results to JSON files (if `--output` specified)

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