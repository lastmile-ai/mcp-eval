"""MCP-Eval: A comprehensive testing framework for MCP servers built on mcp-agent."""

# Core testing paradigms (primary API)
from mcp_eval.core import task, setup, teardown, parametrize, with_agent
from mcp_eval.datasets import Case, Dataset, generate_test_cases
from mcp_eval.session import TestAgent, TestSession, test_session
from mcp_eval.catalog import Expect

# Configuration
from mcp_eval.config import use_agent, use_agent_factory, use_config, MCPEvalSettings

# Modern Evaluator System (preferred approach)
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators import (
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
from mcp_eval.evaluators import register_evaluator
from mcp_eval.metrics import register_metric, TestMetrics

__all__ = [
    # Core testing paradigms
    "task",
    "setup",
    "teardown",
    "parametrize",
    "with_agent",
    # Configuration
    "use_agent",
    "use_agent_factory",
    "use_config",
    "MCPEvalSettings",
    # Dataset API
    "Case",
    "Dataset",
    "generate_test_cases",
    "generate_dataset",
    # Modern Evaluator System (preferred)
    "Evaluator",
    "EvaluatorContext",
    "EvaluatorResult",
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
    # Metrics
    "TestMetrics",
    # Session management
    "TestSession",
    "TestAgent",
    "test_session",
    # Discovery catalog
    "Expect",
]
