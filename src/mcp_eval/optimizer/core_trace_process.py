"""
Core utilities for trace evaluation.

This module provides foundational functions for loading and working with trace files.
"""
import os
import json
from typing import Dict, List, Any
from dataloader import DataExample
from mcp_eval.metrics import TestMetrics, process_spans, TraceSpan, extract_comprehensive_trace_information, TraceInformation
from mcp_eval.evaluators.builtin import LLMJudgeSuccess, EvaluatorContext
import asyncio


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


def extract_server_name_from_trace(trace_file_path: str) -> str:
    """
    Extract server name from trace file by analyzing span attributes.
    
    Args:
        trace_file_path: Path to the trace file
        
    Returns:
        Server name if found, otherwise "unknown"
    """
    traces = read_trace_file(trace_file_path)
    
    # Look for server name in various span attributes
    for span in traces:
        attributes = span.get('attributes', {})
        
        # Check for server name in MCP-related attributes
        for key, value in attributes.items():
            if 'server' in key.lower() and isinstance(value, str):
                # Extract server name from various formats
                if 'server_name' in key.lower():
                    return value
                elif 'server.name' in key.lower():
                    return value
                elif key.endswith('.server'):
                    return value
        
        # Check span name for server information
        span_name = span.get('name', '')
        if span_name and 'server' in span_name.lower():
            # Try to extract server name from span name like "ServerName.method"
            parts = span_name.split('.')
            if len(parts) > 1:
                return parts[0]
    
    # Try to extract from filename if no server name found in traces
    filename = os.path.basename(trace_file_path)
    if '_trace.jsonl' in filename:
        base_name = filename.replace('_trace.jsonl', '')
        # If base name contains server info, use it
        if 'server' in base_name.lower():
            return base_name
    
    return "unknown"


async def evaluate_task_success(trace_info_list: List[TraceInformation], metrics: TestMetrics) -> Dict[str, Any]:
    """
    Evaluate if the user's task was successfully addressed using LLMJudgeSuccess.
    
    Args:
        trace_info_list: List of TraceInformation objects with message content
        metrics: TestMetrics object containing tool calls and other metrics
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create a custom EvaluatorContext with trace information
    class TraceEvaluatorContext(EvaluatorContext):
        def __init__(self, trace_info, metrics):
            self.trace_info = trace_info
            self.inputs = {}
            self.output = ""
            self.expected_output = None
            self.tool_calls = metrics.tool_calls
            self.metrics = metrics
    
    # Create the evaluator context
    ctx = TraceEvaluatorContext(trace_info_list, metrics)
    
    # Create and run the LLMJudgeSuccess evaluator
    evaluator = LLMJudgeSuccess(min_score=0.7)  # Set minimum score threshold
    
    try:
        result = await evaluator.evaluate(ctx)
        
        # Convert EvaluatorResult to dictionary
        return {
            'passed': result.passed,
            'score': result.score,
            'expected': result.expected,
            'actual': result.actual,
            'details': result.details,
            'error': result.error
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'reasoning': 'Failed to evaluate task success with LLM judge'
        }


def separate_traces_by_server(trace_files: List[str]) -> Dict[str, List[str]]:
    """
    Separate trace files by server name.
    
    Args:
        trace_files: List of trace file paths
        
    Returns:
        Dictionary mapping server names to lists of trace files
    """
    server_traces = {}
    
    for trace_file in trace_files:
        server_name = extract_server_name_from_trace(trace_file)
        if server_name not in server_traces:
            server_traces[server_name] = []
        server_traces[server_name].append(trace_file)
    
    return server_traces



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
    tools_info = []
    
    # Method 1: Look for "tools/list" spans with complete tool definitions
    tools_list_spans = [span for span in traces if span.get("name", "") == "MCPAgentClientSession.send_request" and 
                       span.get("attributes", {}).get("mcp.method.name") == "tools/list"]
    
    for span in tools_list_spans:
        attributes = span.get('attributes', {})
        
        # Extract tools from result.tools.X.* attributes
        tool_indices = set()
        for key in attributes.keys():
            if key.startswith('result.tools.') and '.' in key[13:]:
                # Extract tool index (e.g., "result.tools.0.name" -> "0")
                parts = key.split('.')
                if len(parts) >= 3:
                    tool_indices.add(parts[2])
        
        for tool_idx in tool_indices:
            tool_name = attributes.get(f'result.tools.{tool_idx}.name')
            tool_description = attributes.get(f'result.tools.{tool_idx}.description')
            
            if tool_name and tool_description:
                # Build input schema from individual properties
                input_schema = {}
                schema_base = f'result.tools.{tool_idx}.inputSchema'
                
                # Look for schema properties
                for key, value in attributes.items():
                    if key.startswith(schema_base + '.'):
                        schema_path = key[len(schema_base) + 1:]
                        
                        # Parse nested schema structure
                        current = input_schema
                        path_parts = schema_path.split('.')
                        
                        for i, part in enumerate(path_parts[:-1]):
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        
                        # Set the final value
                        final_key = path_parts[-1]
                        current[final_key] = value
                
                tool_info = {
                    'name': tool_name,
                    'description': tool_description,
                    'input_schema': input_schema if input_schema else None
                }
                tools_info.append(tool_info)
    
    # Method 2: Look for Agent.*.list_tools spans with JSON schema
    if not tools_info:
        list_tools_spans = [span for span in traces if ".list_tools" in span.get("name", "")]
        
        for span in list_tools_spans:
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
    
    # Method 3: Look for MCPAggregator.load_server spans with basic tool info
    if not tools_info:
        load_server_spans = [span for span in traces if span.get("name", "") == "MCPAggregator.load_server"]
        
        for span in load_server_spans:
            attributes = span.get('attributes', {})
            
            # Extract tool information from attributes
            for key, value in attributes.items():
                if key.startswith('tool.') and not key.endswith('.description') and not key.endswith('.inputSchema'):
                    # Extract tool name from key (e.g., "tool.fetch" -> "fetch")
                    tool_name = key[5:]  # Remove "tool." prefix
                    
                    tool_info = {
                        'name': tool_name,
                        'description': value,
                        'input_schema': None  # No schema available in this format
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
    # Convert raw trace dictionaries to TraceSpan objects
    trace_spans = []
    for span_dict in traces:
        try:
            # Convert dict to JSON string and then to TraceSpan
            span_json = json.dumps(span_dict)
            trace_spans.append(TraceSpan.from_json(span_json))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error converting span to TraceSpan: {e}")
            continue
    
    # Process spans to get metrics
    metrics = process_spans(trace_spans)
    
    # Extract comprehensive trace information 
    comprehensive_info = extract_comprehensive_trace_information(trace_spans)
    
    # Evaluate task success using LLMJudgeSuccess
    success_evaluation = None
    try:
        success_evaluation = asyncio.run(evaluate_task_success(comprehensive_info, metrics))
    except Exception as e:
        print(f"Error evaluating task success: {e}")
        success_evaluation = {
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'reasoning': 'Failed to evaluate task success'
        }
    
    # Get available tools from trace file
    available_tools = get_tools_info(trace_raw_file_path)
    
    # Extract tool calls and unique tools
    tool_calls = metrics.tool_calls
    unique_tools = metrics.unique_tools_used
    
    # Create updated metrics dictionary with available tools and comprehensive trace info
    updated_metrics = {
        'tool_calls': tool_calls,
        'unique_tools_used': unique_tools,
        'list_of_available_tools': available_tools,
        'iteration_count': metrics.iteration_count,
        'total_duration_ms': metrics.total_duration_ms,
        'latency_ms': metrics.latency_ms,
        'error_count': metrics.error_count,
        'success_rate': metrics.success_rate,
        'cost_estimate': metrics.cost_estimate,
        'comprehensive_trace_info': comprehensive_info,
        'task_success_evaluation': success_evaluation
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



