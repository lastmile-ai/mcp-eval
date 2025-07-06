"""MCP-Eval: A comprehensive testing framework for MCP servers built on mcp-agent."""

# Core testing paradigms (primary API)
from mcp_eval.core import task, setup, teardown, parametrize
from mcp_eval.datasets import Case, Dataset, generate_test_cases
from mcp_eval.session import TestAgent, TestSession, test_session

# Configuration
from mcp_eval.config import use_server, use_agent, use_llm_factory

# Modern Evaluator System (preferred approach)
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.builtin import (
    ToolWasCalled,
    ToolSequence,
    ResponseContains,
    MaxIterations,
    ToolSuccessRate,
    LLMJudge,
    IsInstance,
    EqualsExpected,
)

# Dataset generation
from mcp_eval.generation import generate_dataset

# Extensibility
from mcp_eval.evaluators.builtin import register_evaluator
from mcp_eval.metrics import register_metric

__all__ = [
    # Core testing paradigms
    "task",
    "setup",
    "teardown",
    "parametrize",
    # Configuration
    "use_server",
    "use_agent",
    "use_llm_factory",
    # Dataset API
    "Case",
    "Dataset",
    "generate_test_cases",
    "generate_dataset",
    # Modern Evaluator System (preferred)
    "Evaluator",
    "EvaluatorContext",
    "ToolWasCalled",
    "ToolSequence",
    "ResponseContains",
    "MaxIterations",
    "ToolSuccessRate",
    "LLMJudge",
    "IsInstance",
    "EqualsExpected",
    # Extensibility
    "register_evaluator",
    "register_metric",
    # Session management
    "TestSession",
    "TestAgent",
    "test_session",
]
