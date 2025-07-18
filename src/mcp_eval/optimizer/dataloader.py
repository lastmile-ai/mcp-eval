"""Data loader for trace datasets."""

import dspy
from typing import Dict, Any, Optional


class DataExample(dspy.Example):
    """DSPy Example class for trace dataset."""
    
    def __init__(self, user_query: str, metrics: Dict[str, Any]) -> None:
        """
        Initialize a DataExample instance.
        
        Args:
            user_query: The user's query/prompt
            metrics: Dictionary containing performance metrics and tool usage data
        """
        unique_tools = metrics.get('unique_tools_used', [])
        correct_tool: Optional[str] = unique_tools[0] if unique_tools else None
        
        super().__init__(
            user_query=user_query,
            query=user_query,
            tool_calls=metrics.get('tool_calls', []),
            unique_tools_used=unique_tools,
            correct_tool=correct_tool,
            iteration_count=metrics.get('iteration_count', 0),
            total_duration_ms=metrics.get('total_duration_ms', 0.0),
            latency_ms=metrics.get('latency_ms', 0.0),
            error_count=metrics.get('error_count', 0),
            success_rate=metrics.get('success_rate', 0.0),
            cost_estimate=metrics.get('cost_estimate', 0.0)
        )

