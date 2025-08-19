"""Unified session management with OTEL as the single source of truth."""

import os
import json
import time
import asyncio
import tempfile
import shutil
import re
import logging
from pathlib import Path
from typing import Any, List, Literal, Dict, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, MessageTypes
from mcp_agent.workflows.factory import agent_from_spec as _agent_from_spec_factory

from mcp_eval.config import get_settings, ProgrammaticDefaults, MCPEvalSettings, get_current_config
from mcp_eval.metrics import TestMetrics, process_spans, TraceSpan, ToolCoverage
from mcp_eval.otel.span_tree import SpanTree, SpanNode
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators import EvaluatorResult, EvaluationRecord
from mcp_eval.utils import get_test_artifact_paths


logger = logging.getLogger(__name__)


# Legacy LLM factory resolution removed. Use provider/model via mcp-agent factory.


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

    # Explicit attach_llm by factory/class is removed. LLMs are attached by session configuration.
    async def attach_llm(self, *args, **kwargs) -> AugmentedLLM:  # type: ignore[override]
        raise NotImplementedError(
            "attach_llm is no longer supported on TestAgent. Configure settings.provider/model instead."
        )

    async def generate_str(self, prompt: str, **kwargs) -> str:
        """Generate string response - delegates to underlying agent LLM."""
        if not self._llm:
            raise RuntimeError("No LLM attached. Call attach_llm() first.")

        # Direct delegation to real agent - no re-implementation
        response = await self._llm.generate_str(prompt, **kwargs)
        await self._session._ensure_traces_flushed()
        return response

    async def generate(self, prompt: str, **kwargs):
        """Generate response - delegates to underlying agent LLM."""
        if not self._llm:
            raise RuntimeError("No LLM attached. Call attach_llm() first.")

        response = await self._llm.generate(prompt, **kwargs)
        await self._session._ensure_traces_flushed()
        return response

    def set_llm(self, llm: AugmentedLLM) -> AugmentedLLM:
        """Set an already-constructed AugmentedLLM on this agent."""
        self._llm = llm
        return llm

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

    async def assert_that(
        self,
        evaluator: Evaluator,
        name: Optional[str] = None,
        response: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Unified async assertion API delegated to the session."""
        await self._session.assert_that(
            evaluator,
            name=name,
            response=response,
            **kwargs,
        )

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
        test_name: str,
        verbose: bool = False,
        *,
        agent_override: Optional[Union[Agent, AugmentedLLM, AgentSpec, str]] = None,
    ):
        self.test_name = test_name
        self.verbose = verbose
        self._agent_override = agent_override

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
        self._results: List[EvaluationRecord] = []
        self._available_tools_by_server: Dict[str, List[str]] = {}

    async def __aenter__(self) -> TestAgent:
        """Initialize the test session with OTEL tracing as source of truth."""
        # Clear any cached metrics for fresh session
        self._metrics = None
        self._span_tree = None

        # Configure OpenTelemetry tracing (single source of truth)
        # IMPORTANT: clone settings to avoid cross-test mutation when running in parallel
        _global_settings = get_settings()
        settings = MCPEvalSettings(**_global_settings.model_dump())
        settings.otel.enabled = True
        settings.otel.exporters = ["file"]
        settings.otel.path = self.trace_file
        settings.logger.transports = ["console"] if self.verbose else ["none"]

        # No legacy server merging: servers should be defined in mcp-agent config

        # Initialize MCP app (sets up OTEL instrumentation automatically)
        self.app = MCPApp(settings=settings)
        await self.app.initialize()

        # Construct agent by precedence:
        # 1) Per-call agent_override (Agent | AugmentedLLM | AgentSpec | name)
        # 2) Global default_agent_spec or default_agent_spec_name
        # 3) Minimal default AgentSpec with default_servers
        eval_settings = get_settings()
        # Prefer provider/model from AgentSpec if present; fallback to settings
        spec_provider: Optional[str] = None
        spec_model: Optional[str] = None
        pre_attached_llm: Optional[AugmentedLLM] = None

        async def _agent_from_spec(spec: AgentSpec) -> Agent:
            return _agent_from_spec_factory(spec, context=self.app.context)

        # Global programmatic agent or LLM
        def _effective_servers(existing: List[str] | None) -> List[str]:
            if existing:
                return existing
            # Defaults from config
            default_servers = getattr(eval_settings, "default_servers", None)
            if default_servers:
                return list(default_servers)
            # No defaults
            return []

        # 1) Per-call override
        if self._agent_override is not None:
            override = self._agent_override
            if isinstance(override, AugmentedLLM):
                if override.agent is None:
                    override.agent = Agent(
                        name=f"test_agent_{self.test_name}",
                        instruction="Complete the task as requested.",
                        server_names=_effective_servers(None),
                        context=self.app.context,
                    )
                self.agent = override.agent
                if getattr(override, "context", None) is None:
                    override.context = self.app.context
                await self.agent.attach_llm(llm=override)
                pre_attached_llm = override
            elif isinstance(override, Agent):
                if not override.server_names:
                    override.server_names = _effective_servers(None)
                if override.context is None:
                    override.context = self.app.context
                self.agent = override
            elif isinstance(override, AgentSpec):
                # Capture per-spec provider/model if present
                spec_provider = getattr(override, "provider", None)
                spec_model = getattr(override, "model", None)
                # Build spec kwargs, only including attributes that exist
                spec_kwargs = {
                    "name": override.name,
                    "instruction": override.instruction,
                    "server_names": _effective_servers(override.server_names),
                    "connection_persistence": override.connection_persistence,
                }
                # Add optional attributes if they exist
                if hasattr(override, "functions"):
                    spec_kwargs["functions"] = override.functions
                if hasattr(override, "human_input_callback"):
                    spec_kwargs["human_input_callback"] = override.human_input_callback
                if spec_provider:
                    spec_kwargs["provider"] = spec_provider
                if spec_model:
                    spec_kwargs["model"] = spec_model
                    
                self.agent = await _agent_from_spec(AgentSpec(**spec_kwargs))
            elif isinstance(override, str):
                loaded_specs = getattr(self.app.context, "loaded_subagents", []) or []
                matched = next(
                    (s for s in loaded_specs if getattr(s, "name", None) == override),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"AgentSpec named '{override}' not found in loaded subagents."
                    )
                # Normalize servers
                matched.server_names = _effective_servers(matched.server_names)
                # Capture provider/model extras if provided in spec
                spec_provider = getattr(matched, "provider", None)
                spec_model = getattr(matched, "model", None)
                self.agent = await _agent_from_spec(matched)
            elif isinstance(override, dict):
                raise TypeError("Dict overrides removed. Pass AgentSpec or name.")
            else:
                raise TypeError("Unsupported agent_override type")
        else:
            # 2) Global default Programmatic (Agent | AugmentedLLM) or schema default (AgentSpec | name)
            # Context-local programmatic default allows parallel tests in separate tasks
            factory = ProgrammaticDefaults.get_default_agent_factory()
            programmatic_default = (
                factory()
                if factory is not None
                else ProgrammaticDefaults.get_default_agent()
            )
            default_agent = programmatic_default or getattr(
                eval_settings, "default_agent", None
            )
            if isinstance(default_agent, AugmentedLLM):
                if default_agent.agent is None:
                    default_agent.agent = Agent(
                        name=f"test_agent_{self.test_name}",
                        instruction="Complete the task as requested.",
                        server_names=_effective_servers(None),
                        context=self.app.context,
                    )
                self.agent = default_agent.agent
                if getattr(default_agent, "context", None) is None:
                    default_agent.context = self.app.context
                await self.agent.attach_llm(llm=default_agent)
                pre_attached_llm = default_agent
            elif isinstance(default_agent, Agent):
                if not default_agent.server_names:
                    default_agent.server_names = _effective_servers(None)
                if default_agent.context is None:
                    default_agent.context = self.app.context
                self.agent = default_agent
            elif isinstance(default_agent, AgentSpec):
                # Extract provider/model if specified in the AgentSpec
                spec_provider = getattr(default_agent, "provider", None) or spec_provider
                spec_model = getattr(default_agent, "model", None) or spec_model
                self.agent = await _agent_from_spec(default_agent)
            elif isinstance(default_agent, str):
                loaded_specs = getattr(self.app.context, "loaded_subagents", []) or []
                matched = next(
                    (
                        s
                        for s in loaded_specs
                        if getattr(s, "name", None) == default_agent
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"AgentSpec named '{default_agent}' not found in loaded subagents."
                    )
                self.agent = await _agent_from_spec(matched)
            else:
                # 3) Minimal fallback
                from mcp_agent.agents.agent_spec import AgentSpec as AS

                spec = AS(
                    name=f"test_agent_{self.test_name}",
                    instruction="Complete the task as requested.",
                    server_names=_effective_servers(None),
                )
                self.agent = await _agent_from_spec(spec)

        await self.agent.initialize()
        
        # Fetch available tools from configured servers for coverage tracking
        await self._fetch_available_tools()

        # Create clean test agent wrapper
        self.test_agent = TestAgent(self.agent, self)

        # If an AugmentedLLM was supplied programmatically, use it
        if pre_attached_llm is not None:
            self.test_agent.set_llm(pre_attached_llm)
            # Also attach to the underlying agent
            await self.agent.attach_llm(llm=pre_attached_llm)

        # Configure LLM via provider/model, preferring AgentSpec-level if present
        provider = spec_provider or getattr(settings, "provider", None)
        model = spec_model or getattr(settings, "model", None)
        if provider and pre_attached_llm is None:
            from mcp_agent.workflows.factory import _llm_factory

            llm_factory = _llm_factory(
                provider=provider, model=model, context=self.app.context
            )
            # Build the AugmentedLLM bound to this agent
            augmented_llm = llm_factory(self.agent)
            self.test_agent.set_llm(augmented_llm)
            # Also attach to the underlying agent
            await self.agent.attach_llm(llm=augmented_llm)

        return self.test_agent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session and process final metrics."""
        logger.info(f"TestSession.__aexit__ called for test {self.test_name}")
        try:
            # Process deferred evaluators before cleanup to ensure traces are available
            await self._process_deferred_evaluators()

            # Shutdown agent first
            if self.agent:
                await self.agent.shutdown()

            # Save traces if configured
            logger.info(f"About to save test artifacts for {self.test_name}")
            await self._save_test_artifacts()
            logger.info(f"Completed saving test artifacts for {self.test_name}")

            # Now we can safely cleanup the app - the new mcp-agent version
            # handles OTEL cleanup properly without affecting other apps
            if self.app:
                await self.app.cleanup()
                
            # Give subprocess transports time to close properly
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")
            # Continue with cleanup even if there's an error
    
    async def _fetch_available_tools(self):
        """Fetch list of available tools from configured servers."""
        if not self.agent:
            return
            
        try:
            # Get list of configured servers
            servers = self.agent.server_names if hasattr(self.agent, 'server_names') else []
            
            for server_name in servers:
                try:
                    # Use the agent's list_tools method to get tools for this specific server
                    tools_result = await self.agent.list_tools(server_name=server_name)
                    if tools_result and hasattr(tools_result, 'tools'):
                        # Extract tool names (remove server prefix if present)
                        tool_names = []
                        for tool in tools_result.tools:
                            tool_name = tool.name
                            # Remove server prefix if it exists (format: server_toolname)
                            if tool_name.startswith(f"{server_name}_"):
                                tool_name = tool_name[len(server_name) + 1:]
                            tool_names.append(tool_name)
                        self._available_tools_by_server[server_name] = tool_names
                        logger.info(f"Found {len(tool_names)} tools for server {server_name}")
                except Exception as e:
                    logger.warning(f"Failed to fetch tools for server {server_name}: {e}")
                    self._available_tools_by_server[server_name] = []
        except Exception as e:
            logger.warning(f"Failed to fetch available tools: {e}")
    
    def _calculate_tool_coverage(self, metrics: TestMetrics):
        """Calculate tool coverage metrics per server."""
        # Group used tools by server
        tools_used_by_server: Dict[str, set] = {}
        
        for tool_call in metrics.tool_calls:
            if tool_call.server_name:
                server = tool_call.server_name
                if server not in tools_used_by_server:
                    tools_used_by_server[server] = set()
                tools_used_by_server[server].add(tool_call.name)
        
        # Create coverage metrics for each server
        for server_name, available_tools in self._available_tools_by_server.items():
            used_tools = list(tools_used_by_server.get(server_name, set()))
            coverage = ToolCoverage(
                server_name=server_name,
                available_tools=available_tools,
                used_tools=used_tools
            )
            metrics.tool_coverage[server_name] = coverage

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
            error_result = EvaluatorResult(
                passed=False,
                expected="evaluation to complete",
                actual="error occurred",
                score=0.0,
                error=str(e),
            )
            self._record_evaluation_result(name, error_result, str(e))
            raise

    async def evaluate_now_async(
        self,
        evaluator: Evaluator,
        response: str,
        name: str,
        inputs: MessageTypes | None = None,
    ):
        """Evaluate immediately with async evaluator.

        Args:
            evaluator (Evaluator): The evaluator to use for assessment.
            response (str): The response generated by the agent.
            name (str): Name identifier for this evaluation.
            inputs (MessageTypes | None): the original prompt/messages given to the agent.
        """
        try:
            ctx = EvaluatorContext(
                inputs=inputs if inputs is not None else "",
                output=response,
                expected_output=None,
                metadata={},
                metrics=self.get_metrics(),
                span_tree=self.get_span_tree(),
            )

            result = await evaluator.evaluate(ctx)
            self._record_evaluation_result(name, result, None)

        except Exception as e:
            error_result = EvaluatorResult(
                passed=False,
                expected="evaluation to complete",
                actual="error occurred",
                score=0.0,
                error=str(e),
            )
            self._record_evaluation_result(name, error_result, str(e))
            raise

    async def assert_that(
        self,
        evaluator: Evaluator,
        name: Optional[str] = None,
        response: Optional[str] = None,
        *,
        inputs: MessageTypes | None = None,
        when: Literal["auto", "now", "end"] = "auto",
    ) -> None:
        """Unified API to record an assertion without worrying about timing.

        Behavior:
        - If response is provided:
            - Sync evaluators run immediately and record results.
            - Async evaluators are scheduled immediately and recorded automatically
              without requiring explicit await; completion is awaited at session end.
        - If response is not provided:
            - The evaluator is deferred and will run at session end with full metrics.

        Args:
            evaluator: Evaluator instance
            name: Optional name for the evaluation (defaults to class name)
            response: Optional response/output to evaluate against
            input: Optional input/prompt that produced the response
            when: "auto" (default), "now", or "end" to override scheduling
        """
        eval_name = name or evaluator.__class__.__name__

        # Simplified logic: defer if any of these conditions are true
        should_defer = (
            when == "end"  # Explicitly requested deferral
            or (
                when != "now"
                and (
                    response is None  # No response provided, need full trace
                    or bool(evaluator.requires_final_metrics)  # Needs final metrics
                )
            )
        )

        if should_defer:
            # Defer evaluation to session end
            context = (
                {"inputs": inputs, "output": response}
                if (inputs is not None or response is not None)
                else None
            )
            self._evaluators.append((evaluator, context, eval_name))
            return

        # At this point we should evaluate "now" (immediate)
        if hasattr(evaluator, "evaluate_sync"):
            # Synchronous immediate evaluation
            self.evaluate_now(evaluator, response or "", eval_name)
            return

        # Async evaluator: run immediately and await
        await self.evaluate_now_async(
            evaluator, response or "", eval_name, inputs=inputs
        )

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
                elif isinstance(context_or_response, dict):
                    ctx = EvaluatorContext(
                        inputs=context_or_response.get("inputs", ""),
                        output=context_or_response.get("output", ""),
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
                error_result = EvaluatorResult(
                    passed=False,
                    expected="evaluation to complete",
                    actual="error occurred",
                    score=0.0,
                    error=str(e),
                )
                self._record_evaluation_result(name, error_result, str(e))

    def _record_evaluation_result(
        self, name: str, result: "EvaluatorResult", error: Optional[str]
    ):
        """Record an evaluation result."""
        self._results.append(
            EvaluationRecord(
                name=name,
                result=result,
                passed=result.passed,
                error=error,
            )
        )

    def get_metrics(self) -> TestMetrics:
        """Get test metrics from OTEL traces (single source of truth)."""
        if self._metrics is None:
            self._metrics = self._process_otel_traces()
        return self._metrics

    def cleanup(self):
        """Cleanup session resources."""
        try:
            # Clear cached data
            self._metrics = None
            self._span_tree = None

            # Close temp directory
            if hasattr(self, "temp_dir"):
                self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")

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
        return all(r.passed for r in self._results)

    async def _ensure_traces_flushed(self):
        """Enhanced trace flushing to ensure complete isolation between tests."""
        try:
            # Flush app-specific tracing config
            if self.app and self.app._context and self.app._context.tracing_config:
                await self.app._context.tracing_config.flush()
        except Exception as e:
            logger.warning(f"Error during trace flushing for {self.test_name}: {e}")

    def _process_otel_traces(self) -> TestMetrics:
        """Process OTEL traces into metrics and span tree (single source of truth)."""

        spans: list[TraceSpan] = []
        if os.path.exists(self.trace_file):
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        spans.append(TraceSpan.from_json(line))
                    except json.JSONDecodeError:
                        continue

        # Process spans into metrics (OTEL is the source of truth)
        metrics = process_spans(spans)
        
        # Calculate tool coverage per server
        self._calculate_tool_coverage(metrics)
        
        self._metrics = metrics

        # Build span tree for advanced analysis
        if spans:
            span_nodes: dict[str, SpanNode] = {}
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
            orphaned_nodes: list[SpanNode] = []
            for node in span_nodes.values():
                if node.parent_id and node.parent_id in span_nodes:
                    parent = span_nodes[node.parent_id]
                    parent.children.append(node)
                else:
                    orphaned_nodes.append(node)

            # Create synthetic root to connect all orphaned spans (including the actual root)
            if orphaned_nodes:
                synthetic_root = SpanNode(
                    span_id="synthetic_root",
                    name="Execution Root",
                    start_time=min(node.start_time for node in orphaned_nodes),
                    end_time=max(node.end_time for node in orphaned_nodes),
                    attributes={},
                    events=[],
                    parent_id=None,
                    children=orphaned_nodes,
                )
                self._span_tree = SpanTree(synthetic_root)

        return metrics

    async def _save_test_artifacts(self):
        """Save test artifacts (traces, reports) based on configuration."""
        config = get_current_config()
        reporting_config = config.get("reporting", {})

        # Check if we should save traces
        if not reporting_config.get("include_traces", True):
            logger.info("Skipping trace save - include_traces is False")
            return

        output_dir = Path(reporting_config.get("output_dir", "./test-reports"))
        logger.info(f"Saving test artifacts to {output_dir} for test {self.test_name}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Absolute output path: {output_dir.resolve()}")

        # Get standardized artifact paths
        trace_dest, results_dest = get_test_artifact_paths(self.test_name, output_dir)
        logger.info(f"Artifact paths: trace={trace_dest}, results={results_dest}")

        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save trace file if it exists
            if os.path.exists(self.trace_file):
                shutil.copy2(self.trace_file, trace_dest)
                logger.info(f"Saved trace file to {trace_dest}")
            else:
                logger.warning(
                    f"Trace file not found at {self.trace_file} for test {self.test_name}"
                )

            # Save test results/metrics as JSON
            test_data = {
                "test_name": self.test_name,
                "server_name": ",".join(self.agent.server_names)
                if self.agent and getattr(self.agent, "server_names", None)
                else "",
                "timestamp": self._start_time,
                "duration_ms": self.get_duration_ms(),
                "results": self.get_results(),
                "metrics": self.get_metrics().__dict__ if self._metrics else {},
                "all_passed": self.all_passed(),
            }

            # Convert metrics to serializable format
            if test_data["metrics"]:
                # Handle nested objects
                if "llm_metrics" in test_data["metrics"] and hasattr(
                    test_data["metrics"]["llm_metrics"], "__dict__"
                ):
                    test_data["metrics"]["llm_metrics"] = test_data["metrics"][
                        "llm_metrics"
                    ].__dict__
                if "tool_calls" in test_data["metrics"]:
                    test_data["metrics"]["tool_calls"] = [
                        tc.__dict__ if hasattr(tc, "__dict__") else tc
                        for tc in test_data["metrics"]["tool_calls"]
                    ]

            with open(results_dest, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2, default=str)
            logger.info(f"Saved test results to {results_dest}")

        except Exception as e:
            logger.warning(f"Failed to save test artifacts: {e}", exc_info=True)


@asynccontextmanager
async def test_session(
    test_name: str,
    agent: Optional[Union[Agent, AugmentedLLM, AgentSpec, str]] = None,
):
    """Context manager for creating test sessions.

    Supports programmatic initialization of `Agent` and `AugmentedLLM`, as well as
    declarative initialization from `AgentSpec` or a named AgentSpec discovered by
    the mcp-agent app from configured search paths.
    """
    session = TestSession(
        test_name=test_name,
        agent_override=agent,
    )
    try:
        agent = await session.__aenter__()
        yield agent
    finally:
        await session.__aexit__(None, None, None)
        session.cleanup()
