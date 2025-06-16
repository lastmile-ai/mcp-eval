import asyncio
import os
import json
import tempfile
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import Settings, get_settings, TracePathSettings
from mcp_eval.metrics import TestMetrics, process_spans, SimpleSpan
from typing import Optional, List, Callable, Coroutine, Any
from dataclasses import dataclass, field

@dataclass
class AssertionResult:
    name: str
    passed: bool
    error: Optional[str] = None

@dataclass
class TestResult:
    test_name: str
    description: str
    metrics: TestMetrics
    passed: bool = True
    assertions: List[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None


class TestSession:
    def __init__(self, server_name: str, test_name: str, description: str):
        self.server_name = server_name
        self.test_name = test_name
        self.description = description
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = os.path.join(self.temp_dir.name, "trace.jsonl")

        settings = get_settings()
        settings.otel.enabled = True
        settings.otel.exporters = ["file"]
        settings.otel.path_settings = TracePathSettings(path_pattern=self.trace_file)
        
        self.app = MCPApp(settings=settings)
        self.agent: Optional[Agent] = None
        self.result: Optional[TestResult] = None
        self._assertions: List[AssertionResult] = []
        self._deferred_assertions: List[Callable[[TestMetrics], None]] = []
        self._async_deferred_assertions: List[Callable[[TestMetrics], Coroutine[Any, Any, None]]] = []


    async def __aenter__(self):
        await self.app.initialize()
        
        self.agent = Agent(
            name=self.test_name,
            server_names=[self.server_name],
            context=self.app.context
        )
        await self.agent.initialize()

        mcp_server_config = self.app.config.mcp.servers.get(self.server_name)
        if not mcp_server_config:
            raise ValueError(f"Server '{self.server_name}' not found in config.")
        
        if hasattr(mcp_server_config, 'llm') and mcp_server_config.llm:
             llm_instance_name = mcp_server_config.llm
        else:
             llm_instance_name = next(iter(self.app.config.llm_providers))

        def llm_factory(agent_instance):
            return self.app.context.llm_provider.get_llm(llm_instance_name, agent=agent_instance)

        await self.agent.attach_llm(llm_factory=llm_factory)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.agent.shutdown()
        await self.app.cleanup()
        
        await asyncio.sleep(0.1)

        spans = []
        if os.path.exists(self.trace_file):
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        spans.append(SimpleSpan.from_json(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode trace line: {line}")
        
        metrics = process_spans(spans)
        
        for assertion_func in self._deferred_assertions:
            try:
                assertion_func(metrics)
                self.add_assertion(assertion_func.__name__, passed=True)
            except AssertionError as e:
                self.add_assertion(assertion_func.__name__, passed=False, error=str(e))

        for async_assertion_func in self._async_deferred_assertions:
            try:
                await async_assertion_func(metrics)
                self.add_assertion(async_assertion_func.__name__, passed=True)
            except AssertionError as e:
                self.add_assertion(async_assertion_func.__name__, passed=False, error=str(e))

        passed = not exc_type and all(a.passed for a in self._assertions)
        error_message = str(exc_val) if exc_val else None

        self.result = TestResult(
            test_name=self.test_name,
            description=self.description,
            metrics=metrics,
            passed=passed,
            assertions=self._assertions,
            error=error_message,
        )
        self.temp_dir.cleanup()

    def add_assertion(self, name: str, passed: bool, error: Optional[str] = None):
        self._assertions.append(AssertionResult(name=name, passed=passed, error=error))
    
    def add_deferred_assertion(self, assertion_func: Callable[[TestMetrics], None]):
        self._deferred_assertions.append(assertion_func)

    def add_async_deferred_assertion(self, assertion_func: Callable[[TestMetrics], Coroutine[Any, Any, None]]):
        self._async_deferred_assertions.append(assertion_func)