"""Unified session management with OTEL as the single source of truth."""

import os
import json
import time
import tempfile
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import (
    get_settings,
    TracePathSettings,
    MCPServerSettings,
    MCPSettings,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from .metrics import TestMetrics, process_spans, TraceSpan
from .otel.span_tree import SpanTree, SpanNode
from .evaluators.base import Evaluator, EvaluatorContext


# LLM Factory Registry
LLM_FACTORIES = {
    "AnthropicAugmentedLLM": AnthropicAugmentedLLM,
    "OpenAIAugmentedLLM": OpenAIAugmentedLLM,
}


def resolve_llm_factory(factory_name: Union[str, type]) -> type:
    """Resolve LLM factory name to actual class."""
    if isinstance(factory_name, str):
        if factory_name not in LLM_FACTORIES:
            raise ValueError(
                f"Unknown LLM factory: {factory_name}. "
                f"Available: {list(LLM_FACTORIES.keys())}"
            )
        return LLM_FACTORIES[factory_name]
    return factory_name


class TestAgent:
    """Clean wrapper around mcp_agent.Agent for testing interface.

    This is a thin wrapper that provides convenience methods and maintains
    reference to the session for evaluation context. All core functionality
    delegates to the underlying Agent.
    """

    def __init__(self, agent: Agent, session: "TestSession"):
        self._agent = agent
        self._session = session
        self._llm: Optional[AugmentedLLM] = None

    async def attach_llm(self, llm_factory: Union[str, type]) -> AugmentedLLM:
        """Attach LLM to the underlying agent."""
        llm_factory_class = resolve_llm_factory(llm_factory)
        self._llm = await self._agent.attach_llm(llm_factory_class)
        return self._llm

    async def generate_str(self, prompt: str, **kwargs) -> str:
        """Generate string response - delegates to underlying agent LLM."""
        if not self._llm:
            raise RuntimeError("No LLM attached. Call attach_llm() first.")

        # Direct delegation to real agent - no re-implementation
        return await self._llm.generate_str(prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs):
        """Generate response - delegates to underlying agent LLM."""
        if not self._llm:
            raise RuntimeError("No LLM attached. Call attach_llm() first.")

        return await self._llm.generate(prompt, **kwargs)

    # Evaluation methods that use session context
    def evaluate_now(self, evaluator: Evaluator, response: str, name: str):
        """Immediately evaluate with current session context."""
        self._session.evaluate_now(evaluator, response, name)

    async def evaluate_now_async(self, evaluator: Evaluator, response: str, name: str):
        """Immediately evaluate with current session context (async)."""
        await self._session.evaluate_now_async(evaluator, response, name)

    def add_deferred_evaluator(self, evaluator: Evaluator, name: str):
        """Add evaluator to run at session end."""
        self._session.add_deferred_evaluator(evaluator, name)

    # Convenience properties
    @property
    def agent(self) -> Agent:
        """Access underlying agent if needed."""
        return self._agent

    @property
    def session(self) -> "TestSession":
        """Access session for advanced use cases."""
        return self._session


class TestSession:
    """Unified session manager - single source of truth for all test execution.

    This is the heart of the execution context, responsible for:
    - Setting up MCPApp and Agent with OTEL tracing
    - Managing the trace file as the single source of truth
    - Processing OTEL spans into metrics and span trees
    - Running evaluators with proper context
    - Collecting and reporting results
    """

    def __init__(
        self,
        server_name: str,
        test_name: str,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        self.server_name = server_name
        self.test_name = test_name
        self.agent_config = agent_config or {}

        # Core objects
        self.app: Optional[MCPApp] = None
        self.agent: Optional[Agent] = None
        self.test_agent: Optional[TestAgent] = None

        # OTEL as single source of truth
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = os.path.join(self.temp_dir.name, f"{test_name}_trace.jsonl")

        # Results tracking
        self._evaluators: List[tuple] = []  # (evaluator, context_or_response, name)
        self._start_time = time.time()

        # Cached data (computed from OTEL traces)
        self._metrics: Optional[TestMetrics] = None
        self._span_tree: Optional[SpanTree] = None
        self._results: List[Dict[str, Any]] = []

    async def __aenter__(self) -> TestAgent:
        """Initialize the test session with OTEL tracing as source of truth."""
        # Configure OpenTelemetry tracing (single source of truth)
        settings = get_settings()
        settings.otel.enabled = True
        settings.otel.exporters = ["file"]
        settings.otel.path_settings = TracePathSettings(path_pattern=self.trace_file)

        # Ensure LLM provider settings exist based on the llm_factory
        llm_factory = self.agent_config.get("llm_factory")
        if llm_factory:
            if llm_factory == "AnthropicAugmentedLLM" and settings.anthropic is None:
                from mcp_agent.config import AnthropicSettings

                settings.anthropic = AnthropicSettings()
                # API key will be picked up from environment variable ANTHROPIC_API_KEY
            elif llm_factory == "OpenAIAugmentedLLM" and settings.openai is None:
                from mcp_agent.config import OpenAISettings

                settings.openai = OpenAISettings()
                # API key will be picked up from environment variable OPENAI_API_KEY

        # Load mcp-eval config to get server definitions
        from .config import get_current_config

        mcp_eval_config = get_current_config()

        # Configure servers from mcp-eval config
        if "servers" in mcp_eval_config and mcp_eval_config["servers"]:
            # Ensure mcp settings exist
            if settings.mcp is None:
                settings.mcp = MCPSettings(servers={})

            # Register each server from mcp-eval config
            for server_name, server_config in mcp_eval_config["servers"].items():
                # Convert mcp-eval server config to MCPServerSettings
                mcp_server_settings = MCPServerSettings(
                    name=server_name,
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    transport=server_config.get("transport", "stdio"),
                )
                settings.mcp.servers[server_name] = mcp_server_settings

        # Initialize MCP app (sets up OTEL instrumentation automatically)
        self.app = MCPApp(settings=settings)
        await self.app.initialize()

        # Create agent with configuration
        self.agent = Agent(
            name=self.agent_config.get("name", f"test_agent_{self.test_name}"),
            instruction=self.agent_config.get(
                "instruction", "Complete the task as requested."
            ),
            server_names=[self.server_name],
            context=self.app.context,
        )
        await self.agent.initialize()

        # Create clean test agent wrapper
        self.test_agent = TestAgent(self.agent, self)

        # Configure LLM if specified (do this after creating TestAgent)
        llm_factory = self.agent_config.get("llm_factory")
        if llm_factory:
            await self.test_agent.attach_llm(llm_factory)

        return self.test_agent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session and process final metrics."""
        if self.agent:
            await self.agent.shutdown()

        if self.app:
            await self.app.cleanup()

        # Small delay to ensure OTEL traces are fully written
        await asyncio.sleep(0.1)

        # Process all deferred evaluators using OTEL-derived metrics
        await self._process_deferred_evaluators()

    def add_deferred_evaluator(self, evaluator: Evaluator, name: str):
        """Add evaluator to run at session end with full metrics context."""
        self._evaluators.append((evaluator, None, name))

    def evaluate_now(self, evaluator: Evaluator, response: str, name: str):
        """Evaluate immediately with current response."""
        try:
            # Create minimal context for immediate evaluation
            ctx = EvaluatorContext(
                inputs="",  # Would be set by caller if needed
                output=response,
                expected_output=None,
                metadata={},
                metrics=self.get_metrics(),  # Get current metrics from OTEL
                span_tree=self.get_span_tree(),
            )

            if hasattr(evaluator, "evaluate_sync"):
                result = evaluator.evaluate_sync(ctx)
            else:
                raise ValueError(
                    "Cannot evaluate async evaluator immediately. Use evaluate_now_async() or add_deferred_evaluator()"
                )

            self._record_evaluation_result(name, result, None)

        except Exception as e:
            self._record_evaluation_result(name, False, str(e))
            raise

    async def evaluate_now_async(self, evaluator: Evaluator, response: str, name: str):
        """Evaluate immediately with async evaluator."""
        try:
            ctx = EvaluatorContext(
                inputs="",
                output=response,
                expected_output=None,
                metadata={},
                metrics=self.get_metrics(),
                span_tree=self.get_span_tree(),
            )

            result = await evaluator.evaluate(ctx)
            self._record_evaluation_result(name, result, None)

        except Exception as e:
            self._record_evaluation_result(name, False, str(e))
            raise

    async def _process_deferred_evaluators(self):
        """Process all deferred evaluators using final OTEL metrics."""
        metrics = self.get_metrics()  # Final metrics from OTEL traces
        span_tree = self.get_span_tree()

        for evaluator, context_or_response, name in self._evaluators:
            try:
                # Create evaluation context
                if isinstance(context_or_response, str):
                    ctx = EvaluatorContext(
                        inputs="",
                        output=context_or_response,
                        expected_output=None,
                        metadata={},
                        metrics=metrics,
                        span_tree=span_tree,
                    )
                elif context_or_response is None:
                    # Use session-level context
                    ctx = EvaluatorContext(
                        inputs="",
                        output="",  # Would be filled by specific evaluator
                        expected_output=None,
                        metadata={},
                        metrics=metrics,
                        span_tree=span_tree,
                    )
                else:
                    ctx = context_or_response

                # Run evaluator
                if hasattr(evaluator, "evaluate_sync"):
                    result = evaluator.evaluate_sync(ctx)
                else:
                    result = await evaluator.evaluate(ctx)

                self._record_evaluation_result(name, result, None)

            except Exception as e:
                self._record_evaluation_result(name, False, str(e))

    def _record_evaluation_result(
        self, name: str, result: Union[bool, float, Dict], error: Optional[str]
    ):
        """Record an evaluation result."""
        # Determine if passed
        if isinstance(result, bool):
            passed = result
        elif isinstance(result, (int, float)):
            passed = result > 0.5
        elif isinstance(result, dict):
            passed = result.get("passed", result.get("score", 0) > 0.5)
        else:
            passed = True

        self._results.append(
            {"name": name, "result": result, "passed": passed, "error": error}
        )

    def get_metrics(self) -> TestMetrics:
        """Get test metrics from OTEL traces (single source of truth)."""
        if self._metrics is None:
            self._metrics = self._process_otel_traces()
        return self._metrics

    def get_span_tree(self) -> Optional[SpanTree]:
        """Get span tree for advanced analysis."""
        if self._span_tree is None:
            self._process_otel_traces()  # This sets both metrics and span tree
        return self._span_tree

    def get_duration_ms(self) -> float:
        """Get session duration."""
        return (time.time() - self._start_time) * 1000

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results."""
        return self._results.copy()

    def all_passed(self) -> bool:
        """Check if all evaluations passed."""
        return all(r["passed"] for r in self._results)

    def _process_otel_traces(self) -> TestMetrics:
        """Process OTEL traces into metrics and span tree (single source of truth)."""
        spans = []
        if os.path.exists(self.trace_file):
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        spans.append(TraceSpan.from_json(line))
                    except json.JSONDecodeError:
                        continue

        # Process spans into metrics (OTEL is the source of truth)
        metrics = process_spans(spans)
        self._metrics = metrics

        # Build span tree for advanced analysis
        if spans:
            span_nodes = {}
            for span in spans:
                node = SpanNode(
                    span_id=span.context.get("span_id", ""),
                    name=span.name,
                    start_time=datetime.fromtimestamp(span.start_time / 1e9),
                    end_time=datetime.fromtimestamp(span.end_time / 1e9),
                    attributes=span.attributes,
                    events=span.events,
                    parent_id=span.parent.get("span_id") if span.parent else None,
                )
                span_nodes[node.span_id] = node

            # Build parent-child relationships
            root_node = None
            for node in span_nodes.values():
                if node.parent_id and node.parent_id in span_nodes:
                    parent = span_nodes[node.parent_id]
                    parent.children.append(node)
                else:
                    root_node = node

            if root_node:
                self._span_tree = SpanTree(root_node)

        return metrics

    def cleanup(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()


@asynccontextmanager
async def test_session(
    server_name: str, test_name: str, agent_config: Optional[Dict[str, Any]] = None
) -> TestAgent:
    """Context manager for creating test sessions."""
    session = TestSession(server_name, test_name, agent_config)
    try:
        agent = await session.__aenter__()
        yield agent
    finally:
        await session.__aexit__(None, None, None)
        session.cleanup()
