"""Comprehensive tests for datasets.py to achieve >80% coverage."""

import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from mcp_eval.datasets import Case, Dataset, load_dataset
from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult
from mcp_eval.evaluators import EqualsExpected
from mcp_eval.session import TestSession, TestAgent
from mcp_agent.agents.agent_spec import AgentSpec


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""

    def __init__(self, result: EvaluatorResult = None):
        self.result = result or EvaluatorResult(passed=True, score=1.0)
        self.calls = []

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        self.calls.append(context)
        return self.result

    async def evaluate_async(self, context: EvaluatorContext) -> EvaluatorResult:
        return await self.evaluate(context)

    def evaluate_sync(self, context: EvaluatorContext) -> EvaluatorResult:
        self.calls.append(context)
        return self.result


def test_case_basic():
    """Test basic Case creation."""
    case = Case(
        name="test_case",
        inputs={"input": "value"},
        expected_output={"output": "result"},
    )

    assert case.name == "test_case"
    assert case.inputs == {"input": "value"}
    assert case.expected_output == {"output": "result"}
    assert case.metadata is None
    assert len(case.evaluators) == 1  # EqualsExpected added automatically
    assert isinstance(case.evaluators[0], EqualsExpected)


def test_case_without_expected_output():
    """Test Case without expected output."""
    case = Case(name="no_expected", inputs={"input": "value"})

    assert case.expected_output is None
    assert len(case.evaluators) == 0  # No automatic evaluator


def test_case_with_custom_evaluators():
    """Test Case with custom evaluators."""
    evaluator1 = MockEvaluator()
    evaluator2 = MockEvaluator()

    case = Case(
        name="custom_eval",
        inputs={"input": "value"},
        expected_output={"output": "result"},
        evaluators=[evaluator1, evaluator2],
    )

    assert len(case.evaluators) == 3  # 2 custom + 1 automatic EqualsExpected
    assert evaluator1 in case.evaluators
    assert evaluator2 in case.evaluators


def test_case_with_equals_expected_already():
    """Test Case doesn't duplicate EqualsExpected."""
    equals_eval = EqualsExpected()

    case = Case(
        name="has_equals",
        inputs={"input": "value"},
        expected_output={"output": "result"},
        evaluators=[equals_eval],
    )

    assert len(case.evaluators) == 1  # No duplicate added
    assert case.evaluators[0] == equals_eval


def test_case_with_metadata():
    """Test Case with metadata."""
    case = Case(
        name="with_metadata",
        inputs={"input": "value"},
        metadata={"key": "value", "priority": "high"},
    )

    assert case.metadata == {"key": "value", "priority": "high"}


def test_dataset_basic():
    """Test basic Dataset creation."""
    dataset = Dataset(name="Test Dataset")

    assert dataset.name == "Test Dataset"
    assert dataset.cases == []
    assert dataset.evaluators == []
    assert dataset.server_name is None
    assert dataset.metadata == {}
    assert dataset.agent_spec is None


def test_dataset_with_params():
    """Test Dataset with all parameters."""
    case1 = Case(name="case1", inputs={"in": 1})
    case2 = Case(name="case2", inputs={"in": 2})
    evaluator = MockEvaluator()
    agent_spec = AgentSpec(name="test_agent", description="Test", tools=[])

    dataset = Dataset(
        name="Full Dataset",
        cases=[case1, case2],
        evaluators=[evaluator],
        server_name="test_server",
        metadata={"version": "1.0"},
        agent_spec=agent_spec,
    )

    assert dataset.name == "Full Dataset"
    assert len(dataset.cases) == 2
    assert len(dataset.evaluators) == 1
    assert dataset.server_name == "test_server"
    assert dataset.metadata == {"version": "1.0"}
    assert dataset.agent_spec == agent_spec


def test_dataset_add_case():
    """Test adding cases to dataset."""
    dataset = Dataset(name="Test")
    case = Case(name="new_case", inputs={"test": 1})

    dataset.add_case(case)

    assert len(dataset.cases) == 1
    assert dataset.cases[0] == case


def test_dataset_add_evaluator():
    """Test adding evaluators to dataset."""
    dataset = Dataset(name="Test")
    evaluator = MockEvaluator()

    dataset.add_evaluator(evaluator)

    assert len(dataset.evaluators) == 1
    assert dataset.evaluators[0] == evaluator


@pytest.mark.asyncio
async def test_dataset_evaluate_basic():
    """Test basic dataset evaluation."""
    case = Case(name="test_case", inputs={"value": 42}, expected_output={"result": 42})
    dataset = Dataset(name="Test Dataset", cases=[case])

    async def task_func(inputs, agent, session):
        assert isinstance(agent, TestAgent)
        assert isinstance(session, TestSession)
        return {"result": inputs["value"]}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        report = await dataset.evaluate(task_func)

        assert report.dataset_name == "Test Dataset"
        assert len(report.case_results) == 1
        assert report.case_results[0].case_name == "test_case"
        assert report.case_results[0].passed


@pytest.mark.asyncio
async def test_dataset_evaluate_with_concurrency():
    """Test dataset evaluation with concurrency limit."""
    cases = [Case(name=f"case_{i}", inputs={"value": i}) for i in range(5)]
    dataset = Dataset(name="Concurrent Test", cases=cases)

    call_times = []

    async def task_func(inputs, agent, session):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.1)
        return {"result": inputs["value"]}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        report = await dataset.evaluate(task_func, max_concurrency=2)

        assert len(report.case_results) == 5


@pytest.mark.asyncio
async def test_dataset_evaluate_with_global_evaluators():
    """Test dataset evaluation with global evaluators."""
    case = Case(name="test", inputs={"in": 1})
    global_evaluator = MockEvaluator()
    dataset = Dataset(
        name="Global Eval Test", cases=[case], evaluators=[global_evaluator]
    )

    async def task_func(inputs, agent, session):
        return {"out": inputs["in"]}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        await dataset.evaluate(task_func)

        assert len(global_evaluator.calls) == 1


@pytest.mark.asyncio
async def test_dataset_evaluate_with_error():
    """Test dataset evaluation with error in task."""
    case = Case(name="error_case", inputs={})
    dataset = Dataset(name="Error Test", cases=[case])

    async def task_func(inputs, agent, session):
        raise ValueError("Task error")

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        MockSession.return_value = mock_session

        report = await dataset.evaluate(task_func)

        assert not report.case_results[0].passed
        assert report.case_results[0].error == "Task error"


@pytest.mark.asyncio
async def test_dataset_evaluate_with_evaluator_error():
    """Test dataset evaluation with error in evaluator."""
    case = Case(name="eval_error", inputs={})

    class ErrorEvaluator(Evaluator):
        async def evaluate(self, context):
            raise RuntimeError("Evaluator failed")

    case.evaluators = [ErrorEvaluator()]
    dataset = Dataset(name="Eval Error Test", cases=[case])

    async def task_func(inputs, agent, session):
        return {"output": "success"}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        report = await dataset.evaluate(task_func)

        assert not report.case_results[0].passed
        assert "Evaluator failed" in str(
            report.case_results[0].evaluation_results[0].error
        )


@pytest.mark.asyncio
async def test_dataset_evaluate_with_progress_callback():
    """Test dataset evaluation with progress callback."""
    cases = [Case(name=f"case_{i}", inputs={}) for i in range(3)]
    dataset = Dataset(name="Progress Test", cases=cases)

    progress_calls = []

    def progress_callback(passed, is_final):
        progress_calls.append((passed, is_final))

    async def task_func(inputs, agent, session):
        return {}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        await dataset.evaluate(task_func, progress_callback=progress_callback)

        # Should have calls for each case
        assert len(progress_calls) >= 3


def test_load_dataset_json():
    """Test loading dataset from JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "name": "JSON Dataset",
            "cases": [
                {"name": "case1", "inputs": {"x": 1}, "expected_output": {"y": 2}}
            ],
        }
        json.dump(data, f)
        json_file = f.name

    try:
        dataset = load_dataset(json_file)
        assert dataset.name == "JSON Dataset"
        assert len(dataset.cases) == 1
        assert dataset.cases[0].name == "case1"
    finally:
        Path(json_file).unlink()


def test_load_dataset_yaml():
    """Test loading dataset from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        data = """
name: YAML Dataset
cases:
  - name: case1
    inputs:
      x: 1
    expected_output:
      y: 2
        """
        f.write(data)
        yaml_file = f.name

    try:
        dataset = load_dataset(yaml_file)
        assert dataset.name == "YAML Dataset"
        assert len(dataset.cases) == 1
        assert dataset.cases[0].name == "case1"
    finally:
        Path(yaml_file).unlink()


def test_load_dataset_with_evaluators():
    """Test loading dataset with evaluators configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "name": "Eval Dataset",
            "evaluators": [{"type": "ResponseContains", "substring": "test"}],
            "cases": [
                {
                    "name": "case1",
                    "inputs": {},
                    "evaluators": [{"type": "ToolWasCalled", "tool_name": "test_tool"}],
                }
            ],
        }
        json.dump(data, f)
        json_file = f.name

    try:
        with patch("mcp_eval.datasets.get_evaluator_by_name") as mock_get_eval:
            mock_get_eval.return_value = MockEvaluator()

            dataset = load_dataset(json_file)

            assert len(dataset.evaluators) == 1
            assert len(dataset.cases[0].evaluators) == 1
            assert mock_get_eval.call_count == 2
    finally:
        Path(json_file).unlink()


def test_load_dataset_invalid_file():
    """Test loading dataset from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/file.json")


def test_load_dataset_unsupported_format():
    """Test loading dataset with unsupported format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Not a dataset")
        txt_file = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(txt_file)
    finally:
        Path(txt_file).unlink()


def test_load_dataset_with_metadata():
    """Test loading dataset with metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "name": "Meta Dataset",
            "metadata": {"version": "1.0", "author": "test"},
            "cases": [
                {"name": "case1", "inputs": {}, "metadata": {"priority": "high"}}
            ],
        }
        json.dump(data, f)
        json_file = f.name

    try:
        dataset = load_dataset(json_file)
        assert dataset.metadata == {"version": "1.0", "author": "test"}
        assert dataset.cases[0].metadata == {"priority": "high"}
    finally:
        Path(json_file).unlink()


def test_load_dataset_with_server_name():
    """Test loading dataset with server name."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {"name": "Server Dataset", "server_name": "test_server", "cases": []}
        json.dump(data, f)
        json_file = f.name

    try:
        dataset = load_dataset(json_file)
        assert dataset.server_name == "test_server"
    finally:
        Path(json_file).unlink()


@pytest.mark.asyncio
async def test_dataset_evaluate_with_agent_spec_string():
    """Test dataset evaluation with agent spec as string."""
    case = Case(name="test", inputs={})
    dataset = Dataset(
        name="Agent Spec Test",
        cases=[case],
        agent_spec="test_agent",  # String agent spec
    )

    async def task_func(inputs, agent, session):
        return {}

    with patch("mcp_eval.datasets.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_metrics = Mock(return_value=None)
        mock_session.get_span_tree = Mock(return_value=None)
        MockSession.return_value = mock_session

        await dataset.evaluate(task_func)

        # Verify TestSession was called with agent_override
        MockSession.assert_called_with(
            test_name="Agent Spec Test::test", agent_override="test_agent"
        )
