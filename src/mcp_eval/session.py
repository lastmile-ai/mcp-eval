"""Unified session management with OTEL as the single source of truth."""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, List, Literal, Dict, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio


from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from mcp_eval.config import get_settings
from mcp_eval.metrics import TestMetrics, process_spans, TraceSpan
from mcp_eval.otel.span_tree import SpanTree, SpanNode
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators import EvaluatorResult, EvaluationRecord

import logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency

    def load_dotenv():
        return None


logger = logging.getLogger(__name__)


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
        agent_config: Optional[Dict[str, object]] = None,
        verbose: bool = False,
        *,
        agent_override: Optional[
            Union[Agent, AugmentedLLM, AgentSpec, str, Dict[str, object]]
        ] = None,
    ):
        self.test_name = test_name
        self.agent_config = agent_config or {}
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
        # Track async evaluations scheduled to run immediately (no explicit awaits in tests)
        self._pending_async_evaluations: list[asyncio.Task] = []

    async def __aenter__(self) -> TestAgent:
        """Initialize the test session with OTEL tracing as source of truth."""
        # Clear any cached metrics for fresh session
        self._metrics = None
        self._span_tree = None

        # Configure OpenTelemetry tracing (single source of truth)
        settings = get_settings()
        settings.otel.enabled = True
        settings.otel.exporters = ["file"]
        settings.otel.path = self.trace_file
        settings.logger.transports = ["console"] if self.verbose else ["none"]

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

        # No legacy server merging: servers should be defined in mcp-agent config

        # Initialize MCP app (sets up OTEL instrumentation automatically)
        self.app = MCPApp(settings=settings)
        await self.app.initialize()

        # Construct agent based on precedence:
        # 1) Per-call agent_override (Agent | AugmentedLLM | AgentSpec | name | dict)
        # 2) Global programmatic_agent from settings (use_agent/use_agent_object)
        # 3) Fallback to agent_config lightweight dict
        from mcp_eval.config import get_settings as get_eval_settings

        eval_settings = get_eval_settings()

        async def _agent_from_spec(spec: AgentSpec) -> Agent:
            return Agent(
                name=spec.name,
                instruction=spec.instruction,
                server_names=spec.server_names or [],
                functions=spec.functions or [],
                connection_persistence=spec.connection_persistence,
                human_input_callback=spec.human_input_callback,
                context=self.app.context,
            )

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
            elif isinstance(override, Agent):
                if not override.server_names:
                    override.server_names = _effective_servers(None)
                if override.context is None:
                    override.context = self.app.context
                self.agent = override
            elif isinstance(override, AgentSpec):
                self.agent = await _agent_from_spec(
                    AgentSpec(
                        name=override.name,
                        instruction=override.instruction,
                        server_names=_effective_servers(override.server_names),
                        functions=override.functions,
                        connection_persistence=override.connection_persistence,
                        human_input_callback=override.human_input_callback,
                    )
                )
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
                self.agent = await _agent_from_spec(matched)
            elif isinstance(override, dict):
                self.agent = Agent(
                    name=str(override.get("name", f"test_agent_{self.test_name}")),
                    instruction=str(
                        override.get("instruction", "Complete the task as requested.")
                    ),
                    server_names=_effective_servers(
                        list(override.get("server_names", [])) or None
                    ),
                    context=self.app.context,
                )
            else:
                raise TypeError("Unsupported agent_override type")
        else:
            # 2) Global programmatic config
            pa_cfg = getattr(eval_settings, "programmatic_agent", None)
            if pa_cfg is not None:
                if pa_cfg.kind == "llm_object" and pa_cfg.llm is not None:
                    pa = pa_cfg.llm
                    if getattr(pa, "agent", None) is None:
                        pa.agent = Agent(
                            name=f"test_agent_{self.test_name}",
                            server_names=_effective_servers(None),
                            context=self.app.context,
                        )
                    self.agent = pa.agent
                    if getattr(pa, "context", None) is None:
                        pa.context = self.app.context
                    await self.agent.attach_llm(llm=pa)
                elif pa_cfg.kind == "agent_object" and pa_cfg.agent is not None:
                    agent_obj = pa_cfg.agent
                    if not agent_obj.server_names:
                        agent_obj.server_names = _effective_servers(None)
                    if getattr(agent_obj, "context", None) is None:
                        agent_obj.context = self.app.context
                    self.agent = agent_obj
                elif pa_cfg.kind == "agent_spec" and pa_cfg.agent_spec is not None:
                    spec = pa_cfg.agent_spec
                    self.agent = await _agent_from_spec(
                        AgentSpec(
                            name=spec.name,
                            instruction=spec.instruction,
                            server_names=_effective_servers(spec.server_names),
                            functions=spec.functions,
                            connection_persistence=spec.connection_persistence,
                            human_input_callback=spec.human_input_callback,
                        )
                    )
                elif (
                    pa_cfg.kind == "agent_spec_name"
                    and pa_cfg.agent_spec_name is not None
                ):
                    loaded_specs = (
                        getattr(self.app.context, "loaded_subagents", []) or []
                    )
                    matched = next(
                        (
                            s
                            for s in loaded_specs
                            if getattr(s, "name", None) == pa_cfg.agent_spec_name
                        ),
                        None,
                    )
                    if matched is None:
                        raise ValueError(
                            f"AgentSpec named '{pa_cfg.agent_spec_name}' not found in loaded subagents."
                        )
                    matched.server_names = _effective_servers(matched.server_names)
                    self.agent = await _agent_from_spec(matched)
                elif pa_cfg.kind == "overrides" and pa_cfg.overrides is not None:
                    overrides = pa_cfg.overrides
                    self.agent = Agent(
                        name=str(overrides.get("name", f"test_agent_{self.test_name}")),
                        instruction=str(
                            overrides.get(
                                "instruction", "Complete the task as requested."
                            )
                        ),
                        server_names=_effective_servers(
                            list(overrides.get("server_names", [])) or None
                        ),
                        context=self.app.context,
                    )
                else:
                    raise ValueError("Invalid programmatic_agent configuration")
            else:
                # 3) Fallback to lightweight dict
                self.agent = Agent(
                    name=self.agent_config.get("name", f"test_agent_{self.test_name}"),
                    instruction=self.agent_config.get(
                        "instruction", "Complete the task as requested."
                    ),
                    server_names=_effective_servers(
                        self.agent_config.get("server_names")  # type: ignore[arg-type]
                        if isinstance(self.agent_config.get("server_names"), list)
                        else None
                    ),
                    context=self.app.context,
                )
            if pa_cfg and pa_cfg.kind == "llm_object" and pa_cfg.llm is not None:
                pa = pa_cfg.llm
                # Use LLM's agent
                if getattr(pa, "agent", None) is None:
                    pa.agent = Agent(
                        name=f"test_agent_{self.test_name}",
                        server_names=[],
                        context=self.app.context,
                    )
                self.agent = pa.agent
                if getattr(pa, "context", None) is None:
                    pa.context = self.app.context
                await self.agent.attach_llm(llm=pa)
            elif pa_cfg and pa_cfg.kind == "agent_object" and pa_cfg.agent is not None:
                agent_obj = pa_cfg.agent
                if getattr(agent_obj, "context", None) is None:
                    agent_obj.context = self.app.context
                self.agent = agent_obj
            elif (
                pa_cfg and pa_cfg.kind == "agent_spec" and pa_cfg.agent_spec is not None
            ):
                self.agent = await _agent_from_spec(pa_cfg.agent_spec)
            elif (
                pa_cfg
                and pa_cfg.kind == "agent_spec_name"
                and pa_cfg.agent_spec_name is not None
            ):
                loaded_specs = getattr(self.app.context, "loaded_subagents", []) or []
                matched = next(
                    (
                        s
                        for s in loaded_specs
                        if getattr(s, "name", None) == pa_cfg.agent_spec_name
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"AgentSpec named '{pa_cfg.agent_spec_name}' not found in loaded subagents."
                    )
                self.agent = await _agent_from_spec(matched)
            elif pa_cfg and pa_cfg.kind == "overrides" and pa_cfg.overrides is not None:
                overrides = pa_cfg.overrides
                self.agent = Agent(
                    name=str(overrides.get("name", f"test_agent_{self.test_name}")),
                    instruction=str(
                        overrides.get("instruction", "Complete the task as requested.")
                    ),
                    server_names=list(overrides.get("server_names", [])),
                    context=self.app.context,
                )
            else:
                if pa_cfg is not None:
                    raise ValueError("Invalid programmatic_agent configuration")

        await self.agent.initialize()

        # Create clean test agent wrapper
        self.test_agent = TestAgent(self.agent, self)

        # Configure LLM from factory if provided via overrides (dict path)
        llm_factory = (
            self.agent_config.get("llm_factory")
            if isinstance(self._agent_override, dict) or self._agent_override is None
            else None
        )
        if llm_factory:
            await self.test_agent.attach_llm(llm_factory)

        return self.test_agent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session and process final metrics."""
        logger.info(f"TestSession.__aexit__ called for test {self.test_name}")
        try:
            # First ensure any immediately-scheduled async evaluations are completed
            if self._pending_async_evaluations:
                await asyncio.gather(
                    *self._pending_async_evaluations, return_exceptions=True
                )

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

        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")
            # Continue with cleanup even if there's an error

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
        input_text: str | None = None,
    ):
        """Evaluate immediately with async evaluator.

        Args:
            evaluator (Evaluator): The evaluator to use for assessment.
            response (str): The response generated by the agent.
            name (str): Name identifier for this evaluation.
            input (str | None, optional): input prompt that was given to the agent. Defaults to None.
        """
        try:
            ctx = EvaluatorContext(
                inputs=input_text if input_text is not None else "",
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
        input_text: str | None = None,
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

        # Decide if we should defer evaluator execution to session end. Done if:
        # 1) the caller explicitly requested it, or
        # 2) the evaluator requires final metrics (i.e., after the full trace is processed)
        # 3) the caller didn't provide a response (i.e., we need to evaluate against the full trace).
        force_defer = when == "end"
        requires_final = bool(evaluator.requires_final_metrics)
        missing_response = response is None
        auto_defer = when == "auto" and missing_response

        if force_defer:
            # Defer evaluation to session end
            self._evaluators.append(
                (evaluator, response if response is not None else None, eval_name)
            )
            return

        if requires_final and when != "now":
            # Defer evaluation to session end
            self._evaluators.append(
                (evaluator, response if response is not None else None, eval_name)
            )
            return

        if auto_defer and when != "now":
            # Defer evaluation to session end
            self._evaluators.append(
                (evaluator, response if response is not None else None, eval_name)
            )
            return

        # At this point we should evaluate "now" (immediate)
        if hasattr(evaluator, "evaluate_sync"):
            # Synchronous immediate evaluation
            self.evaluate_now(evaluator, response or "", eval_name)
            return

        # Async evaluator: run immediately and await
        await self.evaluate_now_async(
            evaluator, response or "", eval_name, input_text=input_text
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
        from mcp_eval.config import get_current_config
        import re

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

        # Sanitize test name for filesystem
        safe_test_name = re.sub(r'[<>:"/\\|?*\[\]]', "_", self.test_name)
        logger.info(f"Sanitized test name: {safe_test_name}")

        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save trace file if it exists
            if os.path.exists(self.trace_file):
                trace_dest = output_dir / f"{safe_test_name}_trace.jsonl"
                shutil.copy2(self.trace_file, trace_dest)
                logger.info(f"Saved trace file to {trace_dest}")
            else:
                logger.warning(
                    f"Trace file not found at {self.trace_file} for test {self.test_name}"
                )

            # Also save test results/metrics as JSON
            results_dest = output_dir / f"{safe_test_name}.json"
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

            with open(results_dest, "w") as f:
                json.dump(test_data, f, indent=2, default=str)
            logger.info(f"Saved test results to {results_dest}")

        except Exception as e:
            logger.warning(f"Failed to save test artifacts: {e}", exc_info=True)


@asynccontextmanager
async def test_session(
    test_name: str,
    agent_config: Optional[Dict[str, Any]] = None,
    *,
    initial_agent: Optional[Agent] = None,
    initial_llm: Optional[AugmentedLLM] = None,
    agent_spec: Optional[AgentSpec] = None,
    agent_spec_name: Optional[str] = None,
):
    """Context manager for creating test sessions.

    Supports programmatic initialization of `Agent` and `AugmentedLLM`, as well as
    declarative initialization from `AgentSpec` or a named AgentSpec discovered by
    the mcp-agent app from configured search paths.
    """
    session = TestSession(
        test_name=test_name,
        agent_config=agent_config,
        agent_override=initial_llm or initial_agent or agent_spec or agent_spec_name,
    )
    try:
        agent = await session.__aenter__()
        yield agent
    finally:
        await session.__aexit__(None, None, None)
        session.cleanup()
