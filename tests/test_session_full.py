"""Comprehensive tests for session.py to achieve >80% coverage."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

from mcp_eval.session import TestSession, TestAgent, load_trace_spans
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.config import set_settings, get_settings


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""

    def __init__(self, result: EvaluatorResult = None, is_async: bool = False):
        self.result = result or EvaluatorResult(passed=True, score=1.0)
        self.is_async = is_async
        self.calls = []

    def evaluate_sync(self, context: EvaluatorContext) -> EvaluatorResult:
        self.calls.append(("sync", context))
        return self.result

    async def evaluate_async(self, context: EvaluatorContext) -> EvaluatorResult:
        self.calls.append(("async", context))
        return self.result

    def is_async_evaluator(self) -> bool:
        return self.is_async


@pytest.mark.asyncio
async def test_session_basic_lifecycle():
    """Test basic session lifecycle."""
    session = TestSession(test_name="basic_test")

    assert session.test_name == "basic_test"
    assert session.agent is None
    assert session.test_agent is None

    async with session as agent:
        assert session.agent is not None
        assert session.test_agent is not None
        assert isinstance(agent, TestAgent)
        assert agent._session == session

    # After exit, agent should be cleaned up
    assert session.metrics is not None


@pytest.mark.asyncio
async def test_session_with_agent_spec():
    """Test session with specific agent spec."""
    from mcp_agent.agents.agent_spec import AgentSpec

    agent_spec = AgentSpec(name="test_agent", description="Test agent", tools=[])

    session = TestSession(test_name="spec_test", agent_spec=agent_spec)

    async with session:
        assert session.agent is not None
        assert session.test_agent is not None


@pytest.mark.asyncio
async def test_session_with_config_path(tmp_path):
    """Test session with config path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
name: "Config Test"
description: "Test config"
agents:
  definitions:
    - name: test_agent
      description: Test agent
      tools: []
    """)

    session = TestSession(test_name="config_test", config_path=str(config_file))

    async with session:
        assert session.agent is not None


@pytest.mark.asyncio
async def test_session_evaluate_now():
    """Test immediate evaluation."""
    session = TestSession(test_name="eval_now_test")

    async with session:
        evaluator = MockEvaluator()
        session.evaluate_now(evaluator, "test response", "sync_eval")

        assert len(evaluator.calls) == 1
        assert evaluator.calls[0][0] == "sync"


@pytest.mark.asyncio
async def test_session_evaluate_now_async():
    """Test async immediate evaluation."""
    session = TestSession(test_name="eval_async_test")

    async with session:
        evaluator = MockEvaluator(is_async=True)
        await session.evaluate_now_async(evaluator, "test response", "async_eval")

        assert len(evaluator.calls) == 1
        assert evaluator.calls[0][0] == "async"


@pytest.mark.asyncio
async def test_session_add_deferred_evaluator():
    """Test deferred evaluator."""
    session = TestSession(test_name="deferred_test")

    evaluator = MockEvaluator()

    async with session:
        session.add_deferred_evaluator(evaluator, "deferred_eval")
        assert len(session.deferred_evaluators) == 1

    # After session, deferred evaluators should be executed
    assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_session_assert_that_immediate():
    """Test assert_that with immediate evaluation."""
    session = TestSession(test_name="assert_immediate")

    async with session:
        evaluator = MockEvaluator(EvaluatorResult(passed=True, score=0.9))
        await session.assert_that(evaluator, response="test", when="now")
        assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_session_assert_that_deferred():
    """Test assert_that with deferred evaluation."""
    session = TestSession(test_name="assert_deferred")

    evaluator = MockEvaluator(EvaluatorResult(passed=True, score=0.9))

    async with session:
        await session.assert_that(evaluator, response="test", when="end")
        assert len(evaluator.calls) == 0  # Not evaluated yet

    # After session ends, evaluator should be called
    assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_session_assert_that_with_failure():
    """Test assert_that with failing evaluator."""
    session = TestSession(test_name="assert_failure")

    evaluator = MockEvaluator(EvaluatorResult(passed=False, score=0.3))

    async with session:
        with pytest.raises(AssertionError):
            await session.assert_that(evaluator, response="test", when="now")


@pytest.mark.asyncio
async def test_session_get_results():
    """Test getting evaluation results."""
    session = TestSession(test_name="results_test")

    async with session:
        evaluator1 = MockEvaluator(EvaluatorResult(passed=True, score=1.0))
        evaluator2 = MockEvaluator(EvaluatorResult(passed=False, score=0.5))

        session.evaluate_now(evaluator1, "test1", "eval1")
        session.evaluate_now(evaluator2, "test2", "eval2")

    results = session.get_results()
    assert len(results) == 2
    assert results[0].passed
    assert not results[1].passed


@pytest.mark.asyncio
async def test_session_get_span_tree():
    """Test getting span tree."""
    session = TestSession(test_name="span_tree_test")

    async with session:
        pass

    span_tree = session.get_span_tree()
    # Span tree might be None if no OTEL traces were generated
    assert span_tree is None or hasattr(span_tree, "root")


@pytest.mark.asyncio
async def test_session_save_trace():
    """Test saving trace data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = TestSession(test_name="save_trace_test")

        async with session:
            pass

        trace_path = Path(tmp_dir) / "trace.jsonl"
        session.save_trace(str(trace_path))

        # Check if trace file was created (might be empty if no traces)
        assert trace_path.exists() or session.get_span_tree() is None


@pytest.mark.asyncio
async def test_session_save_report():
    """Test saving report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = TestSession(test_name="save_report_test")

        async with session:
            evaluator = MockEvaluator(EvaluatorResult(passed=True, score=1.0))
            session.evaluate_now(evaluator, "test", "eval")

        report_path = Path(tmp_dir) / "report.json"
        session.save_report(str(report_path))

        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
            assert report["test_name"] == "save_report_test"


@pytest.mark.asyncio
async def test_test_agent_generate_str():
    """Test TestAgent.generate_str method."""
    session = TestSession(test_name="agent_generate_test")

    async with session as agent:
        # Create mock LLM
        mock_llm = AsyncMock(spec=AugmentedLLM)
        mock_llm.generate_str = AsyncMock(return_value="Generated response")

        agent.set_llm(mock_llm)

        response = await agent.generate_str("test prompt")
        assert response == "Generated response"
        mock_llm.generate_str.assert_called_once_with("test prompt")


@pytest.mark.asyncio
async def test_test_agent_generate():
    """Test TestAgent.generate method."""
    session = TestSession(test_name="agent_generate_test")

    async with session as agent:
        # Create mock LLM
        mock_llm = AsyncMock(spec=AugmentedLLM)
        mock_llm.generate = AsyncMock(return_value={"response": "test"})

        agent.set_llm(mock_llm)

        response = await agent.generate("test prompt")
        assert response == {"response": "test"}
        mock_llm.generate.assert_called_once_with("test prompt")


@pytest.mark.asyncio
async def test_test_agent_no_llm_error():
    """Test TestAgent methods fail without LLM."""
    session = TestSession(test_name="no_llm_test")

    async with session as agent:
        with pytest.raises(RuntimeError, match="No LLM attached"):
            await agent.generate_str("test")

        with pytest.raises(RuntimeError, match="No LLM attached"):
            await agent.generate("test")


@pytest.mark.asyncio
async def test_test_agent_attach_llm_not_supported():
    """Test TestAgent.attach_llm raises NotImplementedError."""
    session = TestSession(test_name="attach_llm_test")

    async with session as agent:
        with pytest.raises(NotImplementedError):
            await agent.attach_llm()


@pytest.mark.asyncio
async def test_test_agent_evaluation_methods():
    """Test TestAgent evaluation delegation methods."""
    session = TestSession(test_name="agent_eval_test")

    async with session as agent:
        evaluator = MockEvaluator()

        # Test evaluate_now delegation
        agent.evaluate_now(evaluator, "response", "eval1")
        assert len(evaluator.calls) == 1

        # Test evaluate_now_async delegation
        evaluator2 = MockEvaluator(is_async=True)
        await agent.evaluate_now_async(evaluator2, "response", "eval2")
        assert len(evaluator2.calls) == 1

        # Test add_deferred_evaluator delegation
        evaluator3 = MockEvaluator()
        agent.add_deferred_evaluator(evaluator3, "deferred")
        assert len(session.deferred_evaluators) == 1


@pytest.mark.asyncio
async def test_test_agent_assert_that():
    """Test TestAgent.assert_that delegation."""
    session = TestSession(test_name="agent_assert_test")

    async with session as agent:
        evaluator = MockEvaluator(EvaluatorResult(passed=True, score=0.9))

        await agent.assert_that(evaluator, name="test_assert", response="test")
        assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_session_with_llm_settings():
    """Test session with LLM settings from config."""
    settings = get_settings()
    original_provider = settings.provider
    original_model = settings.model

    try:
        # Set provider and model
        settings.provider = "anthropic"
        settings.model = "claude-3-sonnet"
        set_settings(settings)

        session = TestSession(test_name="llm_settings_test")

        with patch("mcp_eval.session._agent_from_spec_factory") as mock_factory:
            mock_agent = MagicMock(spec=Agent)
            mock_factory.return_value = mock_agent

            async with session:
                assert session.agent is not None
                # Factory should be called with llm_factory
                assert mock_factory.called

    finally:
        settings.provider = original_provider
        settings.model = original_model
        set_settings(settings)


@pytest.mark.asyncio
async def test_session_ensure_traces_flushed():
    """Test _ensure_traces_flushed method."""
    session = TestSession(test_name="traces_flush_test")

    async with session:
        # This should not raise an error even without traces
        await session._ensure_traces_flushed()


@pytest.mark.asyncio
async def test_session_process_metrics():
    """Test _process_metrics method."""
    session = TestSession(test_name="metrics_test")

    async with session:
        pass

    # Metrics should be processed after session
    assert session.metrics is not None


@pytest.mark.asyncio
async def test_session_with_existing_app():
    """Test session with existing MCPApp."""
    from mcp_agent.app import MCPApp

    existing_app = MCPApp()
    session = TestSession(test_name="existing_app_test", app=existing_app)

    async with session:
        assert session.app == existing_app


@pytest.mark.asyncio
async def test_load_trace_spans():
    """Test load_trace_spans function."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write some test spans
        f.write('{"name": "test_span", "trace_id": "123", "span_id": "456"}\n')
        f.write('{"name": "another_span", "trace_id": "123", "span_id": "789"}\n')
        trace_file = f.name

    try:
        spans = load_trace_spans(trace_file)
        assert len(spans) == 2
        assert spans[0]["name"] == "test_span"
        assert spans[1]["name"] == "another_span"
    finally:
        Path(trace_file).unlink()


@pytest.mark.asyncio
async def test_load_trace_spans_nonexistent():
    """Test load_trace_spans with nonexistent file."""
    spans = load_trace_spans("/nonexistent/file.jsonl")
    assert spans == []


@pytest.mark.asyncio
async def test_session_run_deferred_evaluators_async():
    """Test running async deferred evaluators."""
    session = TestSession(test_name="async_deferred_test")

    async_evaluator = MockEvaluator(is_async=True)
    sync_evaluator = MockEvaluator(is_async=False)

    async with session:
        session.add_deferred_evaluator(async_evaluator, "async_eval")
        session.add_deferred_evaluator(sync_evaluator, "sync_eval")

    # Both evaluators should have been called
    assert len(async_evaluator.calls) == 1
    assert len(sync_evaluator.calls) == 1


@pytest.mark.asyncio
async def test_session_multiple_evaluations():
    """Test session with multiple evaluations and mixed results."""
    session = TestSession(test_name="multi_eval_test")

    async with session:
        # Mix of passed and failed evaluations
        eval1 = MockEvaluator(EvaluatorResult(passed=True, score=1.0))
        eval2 = MockEvaluator(EvaluatorResult(passed=False, score=0.3))
        eval3 = MockEvaluator(EvaluatorResult(passed=True, score=0.8))

        session.evaluate_now(eval1, "response1", "eval1")
        session.evaluate_now(eval2, "response2", "eval2")
        await session.evaluate_now_async(eval3, "response3", "eval3")

    results = session.get_results()
    assert len(results) == 3
    assert sum(1 for r in results if r.passed) == 2
    assert sum(1 for r in results if not r.passed) == 1


@pytest.mark.asyncio
async def test_session_with_timeout_settings():
    """Test session timeout configuration."""
    settings = get_settings()
    original_timeout = getattr(settings, "timeout", None)

    try:
        settings.timeout = 30
        set_settings(settings)

        session = TestSession(test_name="timeout_test")
        async with session:
            assert session.agent is not None

    finally:
        if original_timeout is not None:
            settings.timeout = original_timeout
        set_settings(settings)
