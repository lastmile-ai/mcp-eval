"""
Core utilities for trace evaluation.

This module provides foundational functions for loading and working with trace files.
"""

import json
from typing import Dict, List, Any
from dataloader import DataExample


def read_trace_file(trace_file_path: str) -> List[Dict[str, Any]]:
    """
    Read a trace file and return a list of span dictionaries.
    
    Args:
        trace_file_path: Path to the trace file (JSONL format)
        
    Returns:
        List of span dictionaries from the trace file
    """
    spans = []
    with open(trace_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    span = json.loads(line)
                    spans.append(span)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    return spans



def get_tools_info(trace_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract tool information from trace file.
    
    Args:
        trace_file_path: Path to the trace file containing tool spans
        
    Returns:
        List of dictionaries containing tool information including:
        - name: Tool name
        - description: Tool description/docstring
        - input_schema: Tool input arguments schema
    """
    traces = read_trace_file(trace_file_path)
    filtered_items = list(filter(lambda x: "list_tool" in x.get("name", ""), traces))
    tools_info = []
    
    for span in filtered_items:
        attributes = span.get('attributes', {})
        
        # Extract tool information from attributes
        for key, value in attributes.items():
            if key.startswith('tool.') and key.endswith('.description'):
                # Extract tool name from key (e.g., "tool.fetch_fetch.description" -> "fetch_fetch")
                tool_name = key[5:-12]  # Remove "tool." prefix and ".description" suffix
                
                # Look for corresponding input schema
                schema_key = f"tool.{tool_name}.inputSchema"
                input_schema = attributes.get(schema_key, None)
                
                # Parse input schema JSON if it exists
                parsed_schema = None
                if input_schema:
                    try:
                        parsed_schema = json.loads(input_schema)
                    except json.JSONDecodeError:
                        parsed_schema = input_schema  # Keep as string if not valid JSON
                
                tool_info = {
                    'name': tool_name,
                    'description': value,
                    'input_schema': parsed_schema
                }
                tools_info.append(tool_info)
                
    
    print(f"Total spans read: {len(traces)}")
    print(f"Tools found: {len(tools_info)}")
    return tools_info

def extract_trace_dataset(trace_raw_file_path: str, processed_file_path: str) -> DataExample:
    """
    Extract dataset from raw trace file and processed results file.
    
    Args:
        trace_raw_file_path: Path to raw trace file (JSONL format)
        processed_file_path: Path to processed results file (JSON format)
        
    Returns:
        DataExample instance containing extracted dataset with user query and metrics
    """
    # Read processed file for metrics
    with open(processed_file_path, 'r') as f:
        processed_data = json.load(f)
    
    # Read raw trace file to find user query
    traces = read_trace_file(trace_raw_file_path)
    
    # Extract user query from trace data
    user_query = None
    for span in traces:
        attributes = span.get('attributes', {})
        # Look for user prompt in chat completions
        if 'gen_ai.prompt.1.content' in attributes:
            user_query = attributes['gen_ai.prompt.1.content']
            break
    
    # Extract metrics from processed data
    metrics = processed_data.get('metrics', {})
    
    # Get available tools from trace file
    available_tools = get_tools_info(trace_raw_file_path)
    
    # Extract tool calls and unique tools
    tool_calls = metrics.get('tool_calls', [])
    unique_tools = metrics.get('unique_tools_used', [])
    
    # Create updated metrics dictionary with available tools
    updated_metrics = {
        'tool_calls': tool_calls,
        'unique_tools_used': unique_tools,
        'list_of_available_tools': available_tools,
        'iteration_count': metrics.get('iteration_count', 0),
        'total_duration_ms': metrics.get('total_duration_ms', 0.0),
        'latency_ms': metrics.get('latency_ms', 0.0),
        'error_count': metrics.get('error_count', 0),
        'success_rate': metrics.get('success_rate', 0.0),
        'cost_estimate': metrics.get('cost_estimate', 0.0)
    }
    
    # Create and return DataExample instance
    return DataExample(
        user_query=user_query,
        metrics=updated_metrics
    )


def create_trace_dataset(trace_files: List[str], processed_files: List[str]) -> List[DataExample]:
    """
    Create a dataset from multiple trace and processed files.
    
    Args:
        trace_files: List of paths to raw trace files
        processed_files: List of paths to processed result files
        
    Returns:
        List of DataExample instances
    """
    if len(trace_files) != len(processed_files):
        raise ValueError("Number of trace files must match number of processed files")
    
    dataset = []
    for trace_file, processed_file in zip(trace_files, processed_files):
        try:
            entry = extract_trace_dataset(trace_file, processed_file)
            dataset.append(entry)
        except Exception as e:
            print(f"Error processing {trace_file} and {processed_file}: {e}")
            continue
    
    return dataset



