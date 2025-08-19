"""
Core utilities for trace evaluation.

This module provides foundational functions for loading and working with trace files.
"""

import json
from typing import Dict, List, Any
from dataloader import DataExample
from mcp_eval.metrics import process_spans, TraceSpan


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
    tools_info = []

    # Method 1: Look for "tools/list" spans with complete tool definitions
    tools_list_spans = [
        span
        for span in traces
        if span.get("name", "") == "MCPAgentClientSession.send_request"
        and span.get("attributes", {}).get("mcp.method.name") == "tools/list"
    ]

    for span in tools_list_spans:
        attributes = span.get("attributes", {})

        # Extract tools from result.tools.X.* attributes
        tool_indices = set()
        for key in attributes.keys():
            if key.startswith("result.tools.") and "." in key[13:]:
                # Extract tool index (e.g., "result.tools.0.name" -> "0")
                parts = key.split(".")
                if len(parts) >= 3:
                    tool_indices.add(parts[2])

        for tool_idx in tool_indices:
            tool_name = attributes.get(f"result.tools.{tool_idx}.name")
            tool_description = attributes.get(f"result.tools.{tool_idx}.description")

            if tool_name and tool_description:
                # Build input schema from individual properties
                input_schema = {}
                schema_base = f"result.tools.{tool_idx}.inputSchema"

                # Look for schema properties
                for key, value in attributes.items():
                    if key.startswith(schema_base + "."):
                        schema_path = key[len(schema_base) + 1 :]

                        # Parse nested schema structure
                        current = input_schema
                        path_parts = schema_path.split(".")

                        for i, part in enumerate(path_parts[:-1]):
                            if part not in current:
                                current[part] = {}
                            current = current[part]

                        # Set the final value
                        final_key = path_parts[-1]
                        current[final_key] = value

                tool_info = {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema if input_schema else None,
                }
                tools_info.append(tool_info)

    # Method 2: Look for Agent.*.list_tools spans with JSON schema
    if not tools_info:
        list_tools_spans = [
            span for span in traces if ".list_tools" in span.get("name", "")
        ]

        for span in list_tools_spans:
            attributes = span.get("attributes", {})

            # Extract tool information from attributes
            for key, value in attributes.items():
                if key.startswith("tool.") and key.endswith(".description"):
                    # Extract tool name from key (e.g., "tool.fetch_fetch.description" -> "fetch_fetch")
                    tool_name = key[
                        5:-12
                    ]  # Remove "tool." prefix and ".description" suffix

                    # Look for corresponding input schema
                    schema_key = f"tool.{tool_name}.inputSchema"
                    input_schema = attributes.get(schema_key, None)

                    # Parse input schema JSON if it exists
                    parsed_schema = None
                    if input_schema:
                        try:
                            parsed_schema = json.loads(input_schema)
                        except json.JSONDecodeError:
                            parsed_schema = (
                                input_schema  # Keep as string if not valid JSON
                            )

                    tool_info = {
                        "name": tool_name,
                        "description": value,
                        "input_schema": parsed_schema,
                    }
                    tools_info.append(tool_info)

    # Method 3: Look for MCPAggregator.load_server spans with basic tool info
    if not tools_info:
        load_server_spans = [
            span
            for span in traces
            if span.get("name", "") == "MCPAggregator.load_server"
        ]

        for span in load_server_spans:
            attributes = span.get("attributes", {})

            # Extract tool information from attributes
            for key, value in attributes.items():
                if (
                    key.startswith("tool.")
                    and not key.endswith(".description")
                    and not key.endswith(".inputSchema")
                ):
                    # Extract tool name from key (e.g., "tool.fetch" -> "fetch")
                    tool_name = key[5:]  # Remove "tool." prefix

                    tool_info = {
                        "name": tool_name,
                        "description": value,
                        "input_schema": None,  # No schema available in this format
                    }
                    tools_info.append(tool_info)

    print(f"Total spans read: {len(traces)}")
    print(f"Tools found: {len(tools_info)}")
    return tools_info


def extract_trace_dataset(
    trace_raw_file_path: str, processed_file_path: str
) -> DataExample:
    """
    Extract dataset from raw trace file and processed results file.

    Args:
        trace_raw_file_path: Path to raw trace file (JSONL format)
        processed_file_path: Path to processed results file (JSON format)

    Returns:
        DataExample instance containing extracted dataset with user query and metrics
    """
    # Read processed file for metrics
    with open(processed_file_path, "r") as f:
        _processed_data = json.load(f)

    # Read raw trace file to find user query
    traces = read_trace_file(trace_raw_file_path)

    # Extract user query from trace data
    user_query = None
    for span in traces:
        attributes = span.get("attributes", {})
        # Look for user prompt in chat completions
        if "gen_ai.prompt.1.content" in attributes:
            user_query = attributes["gen_ai.prompt.1.content"]
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
    # Extract metrics from processed data

    # Get available tools from trace file
    available_tools = get_tools_info(trace_raw_file_path)

    # Extract tool calls and unique tools
    tool_calls = metrics.tool_calls
    unique_tools = metrics.unique_tools_used

    # Create updated metrics dictionary with available tools
    updated_metrics = {
        "tool_calls": tool_calls,
        "unique_tools_used": unique_tools,
        "list_of_available_tools": available_tools,
        "iteration_count": metrics.iteration_count,
        "total_duration_ms": metrics.total_duration_ms,
        "latency_ms": metrics.latency_ms,
        "error_count": metrics.error_count,
        "success_rate": metrics.success_rate,
        "cost_estimate": metrics.cost_estimate,
    }

    # Create and return DataExample instance
    return DataExample(user_query=user_query, metrics=updated_metrics)


def create_trace_dataset(
    trace_files: List[str], processed_files: List[str]
) -> List[DataExample]:
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
