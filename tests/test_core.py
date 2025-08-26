import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from mcp_eval.core import (
    TestResult,
    generate_test_id,
    setup,
    teardown,
    parametrize,
    task,
    _metrics_to_dict,
    _setup_functions,
    _teardown_functions,
)
from mcp_eval.session import TestSession, TestAgent
from mcp_eval.evaluators import EvaluationRecord
from mcp_eval.metrics import TestMetrics, ToolCall, ToolCoverage, LLMMetrics


def test_generate_test_id():
    """Test test ID generation."""
    test_id = generate_test_id("test_file.py", "test_name")
    assert isinstance(test_id, str)
    assert len(test_id) == 41  # 20 chars + hyphen + 20 chars
    assert "-" in test_id

    # Same inputs should produce same ID
    test_id2 = generate_test_id("test_file.py", "test_name")
    assert test_id == test_id2

    # Different inputs should produce different IDs
    test_id3 = generate_test_id("other_file.py", "test_name")
    assert test_id != test_id3


def test_setup_decorator():
    """Test setup decorator registration."""
    # Clear setup functions first
    _setup_functions.clear()

    @setup
    def my_setup():
        pass

    assert my_setup in _setup_functions
    assert len(_setup_functions) == 1


def test_teardown_decorator():
    """Test teardown decorator registration."""
    # Clear teardown functions first
    _teardown_functions.clear()

    @teardown
    def my_teardown():
        pass

    assert my_teardown in _teardown_functions
    assert len(_teardown_functions) == 1


def test_parametrize_single_param():
    """Test parametrize decorator with single parameter."""

    @parametrize("value", [1, 2, 3])
    def test_func():
        pass

    assert hasattr(test_func, "_mcpeval_param_combinations")
    assert len(test_func._mcpeval_param_combinations) == 3
    assert test_func._mcpeval_param_combinations[0] == {"value": 1}
    assert test_func._mcpeval_param_combinations[1] == {"value": 2}
    assert test_func._mcpeval_param_combinations[2] == {"value": 3}


def test_parametrize_multiple_params():
    """Test parametrize decorator with multiple parameters."""

    @parametrize("a, b", [(1, 2), (3, 4), (5, 6)])
    def test_func():
        pass

    assert hasattr(test_func, "_mcpeval_param_combinations")
    assert len(test_func._mcpeval_param_combinations) == 3
    assert test_func._mcpeval_param_combinations[0] == {"a": 1, "b": 2}
    assert test_func._mcpeval_param_combinations[1] == {"a": 3, "b": 4}
    assert test_func._mcpeval_param_combinations[2] == {"a": 5, "b": 6}


def test_parametrize_invalid_values():
    """Test parametrize decorator with invalid values."""

    with pytest.raises(ValueError, match="Parameter count mismatch"):

        @parametrize("a, b", [1, 2, 3])  # Single values for multiple params
        def test_func():
            pass


def test_parametrize_tuple_list_values():
    """Test parametrize decorator with list values."""

    @parametrize("a, b, c", [[1, 2, 3], [4, 5, 6]])
    def test_func():
        pass

    assert hasattr(test_func, "_mcpeval_param_combinations")
    assert len(test_func._mcpeval_param_combinations) == 2
    assert test_func._mcpeval_param_combinations[0] == {"a": 1, "b": 2, "c": 3}


def test_metrics_to_dict_none():
    """Test _metrics_to_dict with None input."""
    result = _metrics_to_dict(None)
    assert result is None


def test_metrics_to_dict_with_tool_calls():
    """Test _metrics_to_dict with tool calls."""
    metrics = TestMetrics(
        total_iterations=5,
        total_llm_calls=3,
        total_tool_calls=2,
        tool_calls=[
            ToolCall(tool_name="test_tool", arguments={"arg": "value"}),
            {"tool_name": "already_dict", "arguments": {}},  # Already a dict
        ],
        tool_coverage={},
        llm_metrics=None,
    )

    result = _metrics_to_dict(metrics)
    assert result is not None
    assert result["total_iterations"] == 5
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["tool_name"] == "test_tool"
    assert result["tool_calls"][1]["tool_name"] == "already_dict"


def test_metrics_to_dict_with_tool_coverage():
    """Test _metrics_to_dict with tool coverage."""
    metrics = TestMetrics(
        total_iterations=5,
        total_llm_calls=3,
        total_tool_calls=2,
        tool_calls=[],
        tool_coverage={
            "server1": ToolCoverage(
                total_tools=10,
                tools_called=5,
                coverage_percentage=50.0,
                tools_list=["tool1", "tool2"],
                called_tools=["tool1"],
            )
        },
        llm_metrics=None,
    )

    result = _metrics_to_dict(metrics)
    assert result is not None
    assert "tool_coverage" in result
    assert "server1" in result["tool_coverage"]
    assert result["tool_coverage"]["server1"]["total_tools"] == 10


def test_metrics_to_dict_with_llm_metrics():
    """Test _metrics_to_dict with LLM metrics."""
    llm_metrics = LLMMetrics(
        total_tokens=1000, prompt_tokens=600, completion_tokens=400, total_cost=0.01
    )

    metrics = TestMetrics(
        total_iterations=5,
        total_llm_calls=3,
        total_tool_calls=2,
        tool_calls=[],
        tool_coverage={},
        llm_metrics=llm_metrics,
    )

    result = _metrics_to_dict(metrics)
    assert result is not None
    assert "llm_metrics" in result
    assert result["llm_metrics"]["total_tokens"] == 1000


def test_test_result_dataclass():
    """Test TestResult dataclass."""
    eval_record = EvaluationRecord(
        name="test_eval",
        evaluator_type="TestEvaluator",
        passed=True,
        score=1.0,
        reasoning="Good",
        confidence=0.9,
    )

    result = TestResult(
        id="test-id",
        test_name="test",
        description="Test description",
        server_name="server1",
        servers=["server1", "server2"],
        agent_name="test_agent",
        parameters={"param": "value"},
        passed=True,
        evaluation_results=[eval_record],
        metrics={"metric": 1},
        duration_ms=100.5,
        file="test_file.py",
        error=None,
    )

    assert result.test_name == "test"
    assert result.passed is True
    assert len(result.servers) == 2
    assert result.duration_ms == 100.5
    assert result.error is None


@pytest.mark.asyncio
async def test_task_decorator_basic():
    """Test task decorator basic functionality."""

    @task(description="Test task")
    async def test_task_func(agent: TestAgent, session: TestSession):
        return "success"

    assert hasattr(test_task_func, "__wrapped__")
    assert asyncio.iscoroutinefunction(test_task_func)


@pytest.mark.asyncio
async def test_task_decorator_with_session():
    """Test task decorator with TestSession."""
    _setup_functions.clear()
    _teardown_functions.clear()

    setup_called = False
    teardown_called = False

    @setup
    async def my_setup():
        nonlocal setup_called
        setup_called = True

    @teardown
    async def my_teardown():
        nonlocal teardown_called
        teardown_called = True

    @task(description="Test task")
    async def test_task_func(agent: TestAgent, session: TestSession):
        assert agent is not None
        assert session is not None
        return "test_result"

    # Mock the TestSession context manager
    with patch("mcp_eval.core.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_results = Mock(return_value=[])
        mock_session.get_span_tree = Mock(return_value=None)
        mock_session.metrics = None
        MockSession.return_value = mock_session

        # Call the task
        await test_task_func()

        # Check setup and teardown were called
        assert setup_called
        assert teardown_called

        # Check session was created properly
        MockSession.assert_called_once()


@pytest.mark.asyncio
async def test_task_decorator_with_error():
    """Test task decorator error handling."""
    _setup_functions.clear()
    _teardown_functions.clear()

    @task(description="Error task")
    async def test_error_task(agent: TestAgent, session: TestSession):
        raise ValueError("Test error")

    with patch("mcp_eval.core.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)

        # Make the task raise an error
        async def raise_error(*args, **kwargs):
            raise ValueError("Test error")

        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        MockSession.return_value = mock_session

        # The decorator should catch and handle the error
        with patch("mcp_eval.core.traceback.print_exc"):
            result = await test_error_task()
            assert result.error == "Test error"
            assert result.passed is False


@pytest.mark.asyncio
async def test_task_decorator_with_params():
    """Test task decorator with parameters."""

    @parametrize("value", [1, 2])
    @task(description="Param task")
    async def test_param_task(agent: TestAgent, session: TestSession, value: int):
        assert value in [1, 2]
        return f"value_{value}"

    assert hasattr(test_param_task, "_mcpeval_param_combinations")
    assert len(test_param_task._mcpeval_param_combinations) == 2


@pytest.mark.asyncio
async def test_task_decorator_sync_setup_teardown():
    """Test task decorator with sync setup/teardown functions."""
    _setup_functions.clear()
    _teardown_functions.clear()

    setup_called = False
    teardown_called = False

    @setup
    def sync_setup():
        nonlocal setup_called
        setup_called = True

    @teardown
    def sync_teardown():
        nonlocal teardown_called
        teardown_called = True

    @task(description="Sync setup/teardown task")
    async def test_task_func(agent: TestAgent, session: TestSession):
        return "success"

    with patch("mcp_eval.core.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get_results = Mock(return_value=[])
        mock_session.get_span_tree = Mock(return_value=None)
        mock_session.metrics = None
        MockSession.return_value = mock_session

        await test_task_func()

        assert setup_called
        assert teardown_called


@pytest.mark.asyncio
async def test_task_with_evaluations():
    """Test task decorator with evaluation results."""
    _setup_functions.clear()
    _teardown_functions.clear()

    @task(description="Evaluation task")
    async def test_eval_task(agent: TestAgent, session: TestSession):
        return "success"

    with patch("mcp_eval.core.TestSession") as MockSession:
        mock_session = MagicMock()
        mock_agent = MagicMock(spec=TestAgent)
        mock_session.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock evaluation results
        mock_eval_record = EvaluationRecord(
            name="test_eval",
            evaluator_type="TestEvaluator",
            passed=True,
            score=1.0,
            reasoning="Good",
            confidence=0.9,
        )
        mock_session.get_results = Mock(return_value=[mock_eval_record])
        mock_session.get_span_tree = Mock(return_value=None)
        mock_session.metrics = TestMetrics(
            total_iterations=1,
            total_llm_calls=1,
            total_tool_calls=0,
            tool_calls=[],
            tool_coverage={},
            llm_metrics=None,
        )

        MockSession.return_value = mock_session

        result = await test_eval_task()

        assert result.passed is True
        assert len(result.evaluation_results) == 1
        assert result.metrics is not None


def test_test_result_with_error():
    """Test TestResult with error field."""
    result = TestResult(
        id="error-test",
        test_name="error_test",
        description="Test with error",
        server_name="server",
        servers=["server"],
        agent_name="agent",
        parameters={},
        passed=False,
        evaluation_results=[],
        metrics=None,
        duration_ms=50.0,
        file="test.py",
        error="Something went wrong",
    )

    assert result.passed is False
    assert result.error == "Something went wrong"
