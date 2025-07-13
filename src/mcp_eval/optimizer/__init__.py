"""
MCP Eval Optimizer Package

This package provides tools for optimizing tool selection using DSPy.
"""

from .core_trace_process import create_trace_dataset, get_tools_info
from .dataloader import DataExample
from .predictors import ToolPredictor

__all__ = [
    "create_trace_dataset",
    "get_tools_info", 
    "DataExample",
    "ToolPredictor"
]

__version__ = "0.1.0"