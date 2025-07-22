"""Functions to group TraceInformation objects by various span criteria."""

from typing import List, Dict
from collections import defaultdict
from ..metrics import TraceInformation


def group_trace_information_by_span(trace_info_list: List[TraceInformation]) -> Dict[str, List[TraceInformation]]:
    """Group TraceInformation objects by their span.
    
    Args:
        trace_info_list: List of TraceInformation objects to group
        
    Returns:
        Dictionary where keys are span identifiers and values are lists of TraceInformation objects
        that belong to the same span
    """
    grouped = defaultdict(list)
    
    for trace_info in trace_info_list:
        # Use span ID if available, otherwise use span name as grouping key
        if trace_info.original_span and trace_info.original_span.context.get("span_id"):
            span_key = trace_info.original_span.context["span_id"]
        elif trace_info.original_span:
            # If no span_id, use combination of span name and start time for uniqueness
            span_key = f"{trace_info.original_span.name}_{trace_info.original_span.start_time}"
        else:
            # Fallback to using metadata span name if original_span is not available
            span_name = trace_info.message_info.metadata.get("span_name", "unknown")
            span_key = f"{span_name}_{trace_info.message_info.timestamp}"
        
        grouped[span_key].append(trace_info)
    
    return dict(grouped)


def group_trace_information_by_span_name(trace_info_list: List[TraceInformation]) -> Dict[str, List[TraceInformation]]:
    """Group TraceInformation objects by their span name.
    
    Args:
        trace_info_list: List of TraceInformation objects to group
        
    Returns:
        Dictionary where keys are span names and values are lists of TraceInformation objects
        that have the same span name
    """
    grouped = defaultdict(list)
    
    for trace_info in trace_info_list:
        # Get span name from original span or from metadata
        if trace_info.original_span:
            span_name = trace_info.original_span.name
        else:
            span_name = trace_info.message_info.metadata.get("span_name", "unknown")
        
        grouped[span_name].append(trace_info)
    
    return dict(grouped)


def group_trace_information_by_parent_span(trace_info_list: List[TraceInformation]) -> Dict[str, List[TraceInformation]]:
    """Group TraceInformation objects by their parent span.
    
    Args:
        trace_info_list: List of TraceInformation objects to group
        
    Returns:
        Dictionary where keys are parent span IDs (or 'root' for spans without parent) 
        and values are lists of TraceInformation objects that share the same parent
    """
    grouped = defaultdict(list)
    
    for trace_info in trace_info_list:
        # Get parent span ID if available
        parent_key = "root"  # Default for root spans
        
        if (trace_info.original_span and 
            trace_info.original_span.parent and 
            trace_info.original_span.parent.get("span_id")):
            parent_key = trace_info.original_span.parent["span_id"]
        
        grouped[parent_key].append(trace_info)
    
    return dict(grouped)