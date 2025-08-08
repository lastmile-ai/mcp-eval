"""
Core utilities for trace evaluation.

This module provides foundational functions for loading and working with trace files.
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from mcp_eval.optimizer.dataloader import DataExample
from mcp_eval.metrics import TestMetrics, process_spans, TraceSpan, extract_comprehensive_trace_information, TraceInformation, extract_user_query, extract_tool_info_from_llm_span, _is_tool_call_span, _extract_tool_call, _is_trace_get_capabilities_span, _is_trace_list_tools_span, _is_trace_user_query_span, _is_response_span, _is_the_shutdown_span, _extract_response_from_span
from mcp_eval.evaluators.builtin import LLMJudgeSuccess, EvaluatorContext
from mcp_eval.evaluators.base import EvaluatorContext

from mcp_eval.optimizer.trace_grouping import group_trace_information_by_span_name
from mcp_eval.optimizer.trace_prompt_generator import TracePromptGenerator
from mcp_eval.otel.span_tree import SpanTree, SpanNode
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


async def evaluate_task_success(trace_info_list: List[TraceInformation], evaluation_prompts: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Evaluate if the user's task was successfully addressed using LLMJudgeSuccess.
    
    Args:
        trace_info_list: List of TraceInformation objects with message content
        evaluation_prompts: Dictionary containing structured prompts for evaluation
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create a custom EvaluatorContext with trace information
    class TraceEvaluatorContext(EvaluatorContext):
        def __init__(self, trace_info, prompts):
            self.trace_info = trace_info
            self.evaluation_prompts = prompts
    
    # Create the evaluator context
    ctx = TraceEvaluatorContext(trace_info_list, evaluation_prompts)
    
    # Create and run the LLMJudgeSuccess evaluator
    evaluator = LLMJudgeSuccess(min_score=0.7)  # Set minimum score threshold
    
    try:
        result = await evaluator.evaluate(ctx)
        print(f"Evaluator result: {result}")
        
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
    
    return tools_info


def filter_traces_by_server(traces: List[Dict[str, Any]], server_name: str) -> List[Dict[str, Any]]:
    """
    Filter traces to return only those associated with the specified server.
    
    Args:
        traces: List of raw trace dictionaries
        server_name: Server name to filter for
        
    Returns:
        List of traces associated with the specified server
    """
    if not server_name:
        return traces
    
    filtered_traces = []
    
    for trace in traces:
        # Check if this trace is related to the specified server
        if _is_trace_for_server(trace, server_name):
            filtered_traces.append(trace)
    
    return filtered_traces


def _is_trace_for_server(trace: Dict[str, Any], server_name: str) -> bool:
    """
    Check if a trace is associated with the specified server.
    
    Args:
        trace: Raw trace dictionary
        server_name: Server name to check for
        
    Returns:
        True if trace is associated with the server, False otherwise
    """
    attributes = trace.get('attributes', {})
    span_name = trace.get('name', '')
    
    # Check for server name in various attribute patterns
    for key, value in attributes.items():
        if 'server' in key.lower() and isinstance(value, str):
            if server_name.lower() in value.lower():
                return True
    
    # Check span name for server information
    if server_name.lower() in span_name.lower():
        return True
    
    # Check for tool calls that might be from this server
    tool_name = attributes.get('mcp.tool.name') or attributes.get('tool.name')
    if tool_name and server_name.lower() in tool_name.lower():
        return True
    
    # Check events for server-related information
    events = trace.get('events', [])
    for event in events:
        event_attrs = event.get('attributes', {})
        for key, value in event_attrs.items():
            if isinstance(value, str) and server_name.lower() in value.lower():
                return True
    
    return False


def create_span_tree_from_raw_file(trace_raw_file_path: str) -> Optional[SpanTree]:
    """
    Create a SpanTree from raw trace file.
    
    Args:
        trace_raw_file_path: Path to raw trace file (JSONL format)
        
    Returns:
        SpanTree instance or None if no spans found
    """
    traces = read_trace_file(trace_raw_file_path)
    if not traces:
        return None
    
    # Convert raw trace dictionaries to SpanNode objects
    span_nodes = {}
    root_spans = []
    
    for span_dict in traces:
        try:
            # Extract span information
            span_id = span_dict.get('span_id') or span_dict.get('spanId', '')
            name = span_dict.get('name', '')
            
            # Parse timestamps
            start_time_str = span_dict.get('start_time') or span_dict.get('startTime', '')
            end_time_str = span_dict.get('end_time') or span_dict.get('endTime', '')
            
            if isinstance(start_time_str, (int, float)):
                # Nanoseconds to seconds
                start_time = datetime.fromtimestamp(start_time_str / 1_000_000_000)
            else:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            
            if isinstance(end_time_str, (int, float)):
                # Nanoseconds to seconds  
                end_time = datetime.fromtimestamp(end_time_str / 1_000_000_000)
            else:
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Extract attributes and events
            attributes = span_dict.get('attributes', {})
            events = span_dict.get('events', [])
            parent_id = span_dict.get('parent_span_id') or span_dict.get('parentSpanId')
            
            # Create SpanNode
            span_node = SpanNode(
                span_id=span_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                attributes=attributes,
                events=events,
                parent_id=parent_id
            )
            
            span_nodes[span_id] = span_node
            
            # Track root spans (no parent)
            if not parent_id:
                root_spans.append(span_node)
                
        except Exception as e:
            print(f"Error converting span to SpanNode: {e}")
            continue
    
    # Build parent-child relationships
    for span_node in span_nodes.values():
        if span_node.parent_id and span_node.parent_id in span_nodes:
            parent = span_nodes[span_node.parent_id]
            parent.children.append(span_node)
    
    # Create SpanTree with the first root span, or create artificial root if multiple roots
    if not root_spans:
        return None
    elif len(root_spans) == 1:
        return SpanTree(root_spans[0])
    else:
        # Create artificial root to hold multiple root spans
        artificial_root = SpanNode(
            span_id="artificial_root",
            name="Root",
            start_time=min(span.start_time for span in root_spans),
            end_time=max(span.end_time for span in root_spans),
            attributes={},
            events=[],
            children=root_spans
        )
        return SpanTree(artificial_root)


async def extract_trace_dataset(trace_raw_file_path: str) -> List[DataExample]:
    """Extract dataset from raw trace file.
    
    Args:
        trace_raw_file_path: Path to raw trace file (JSONL format)
        
    Returns:
        List of DataExample instances containing extracted dataset with user query and metrics
    """
    # Read raw trace file
    traces = read_trace_file(trace_raw_file_path)
    list_of_available_tools = []
    trace_spans = []
    data_examples = []
    current_server_name, current_user_query, current_available_tools = None, None, None
    current_response = ""
    current_tool_info = []
    session_active = False
    
    print(f"Processing {len(traces)} spans from trace file")
    
    while len(traces) > 0:
        if current_user_query and current_tool_info and session_active and current_response:
            print("Processing collected spans and creating data example")
            
            # Process spans to get metrics
            metrics = process_spans(trace_spans)
            comprehensive_info = extract_comprehensive_trace_information(trace_spans)
            
            # Generate prompts from trace information
            prompt_generator = TracePromptGenerator()
            trace_events = prompt_generator.extract_trace_events(comprehensive_info)
            evaluation_prompts = prompt_generator.generate_evaluation_prompt(trace_events)
            
            # Evaluate task success using LLMJudgeSuccess
            success_evaluation = None
            try:
                success_evaluation = await evaluate_task_success(comprehensive_info, evaluation_prompts)
            except Exception as e:
                print(f"Error evaluating task success: {e}")
                success_evaluation = {
                    'passed': False,
                    'score': 0.0,
                    'error': str(e),
                    'reasoning': 'Failed to evaluate task success'
                }
            
            # Create comprehensive metrics
            updated_metrics = {
                'tool_calls': current_tool_info,
                'response': current_response,
                'list_of_available_tools': current_available_tools or [],
                'iteration_count': metrics.iteration_count,
                'total_duration_ms': metrics.total_duration_ms,
                'latency_ms': metrics.latency_ms,
                'error_count': metrics.error_count,
                'success_rate': metrics.success_rate,
                'cost_estimate': metrics.cost_estimate,
                'comprehensive_trace_info': comprehensive_info,
                'is_successful': success_evaluation.get('passed', False),
                'score': success_evaluation.get('score', 0.0),
                'task_success_evaluation': success_evaluation['details'].get('reasoning', 'No evaluation performed'),
            }
            
            # Create DataExample instance
            data_example = DataExample(user_query=current_user_query, metrics=updated_metrics)
            data_examples.append(data_example)
            print(f"Created data example for query: {current_user_query[:50]}...")
            list_of_available_tools.extend(current_available_tools)
            # Reset for next iteration
            trace_spans = []
            current_user_query, current_response, current_tool_info = None, "", []
        span = traces.pop(0)
        # TODO: the trace_spans needed to be carefully constructed to avoid duplicates, data leakage, and comprehensive information
        try:
            # Convert dict to JSON string and then to TraceSpan
            span_json = json.dumps(span)
            trace_span = TraceSpan.from_json(span_json)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error converting span to TraceSpan: {e}")
            continue
            
        # Check if this is server capabilities/initialization
        if _is_trace_get_capabilities_span(trace_span) and not session_active:
            current_server_name = extract_server_name_from_trace(trace_raw_file_path)
            session_active = True
            trace_spans.append(trace_span)
            continue
            
        # Check if this is tools listing
        if _is_trace_list_tools_span(trace_span):
            current_available_tools = get_tools_info(trace_raw_file_path)
            trace_spans.append(trace_span)
            continue
            
        # Check if this contains user query
        if _is_trace_user_query_span(trace_span):
            user_query = extract_user_query(span)
            tool_info = extract_tool_info_from_llm_span(span)
            trace_spans.append(trace_span)
            if user_query:
                current_user_query = user_query
                current_tool_info.append(tool_info)
                response = _extract_response_from_span(trace_span)
                if response and current_user_query:
                    current_response = response
            continue
                        
            
        # Check if this is shutdown - end of session
        if _is_the_shutdown_span(trace_span):
            session_active = False
            trace_spans.append(trace_span)
            
        # Process collected data when we have enough information or session ends

    
    print(f"Generated {len(data_examples)} data examples")
    return data_examples, list_of_available_tools


async def create_trace_dataset(trace_files: List[str]) -> List[DataExample]:
    """
    Create a dataset from multiple trace files.
    
    Args:
        trace_files: List of paths to raw trace files
        
    Returns:
        List of DataExample instances
    """
    # for testing I am going to run the optimization with 3 trace files
    trace_files = trace_files[:5]  
    dataset = []
    list_of_available_tools = []
    for trace_file in trace_files:
        try:
            # extract_trace_dataset now returns a list of DataExample instances
            examples, available_tools = await extract_trace_dataset(trace_file)
            dataset.extend(examples)
            list_of_available_tools.extend(available_tools)
        except Exception as e:
            print(f"Error processing {trace_file}: {e}")
            continue
    
    return dataset, list_of_available_tools



