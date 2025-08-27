from datetime import datetime, timedelta
from mcp_eval.otel.span_tree import (
    SpanNode,
    SpanTree,
    LLMRephrasingLoop,
    ToolPathAnalysis,
    ErrorRecoverySequence,
)


def test_span_node_creation():
    """Test SpanNode creation."""
    node = SpanNode(
        span_id="1",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
    )

    assert node.span_id == "1"
    assert node.parent_id is None
    assert node.name == "test_span"
    assert node.children == []
    assert node.attributes == {}


def test_span_node_with_attributes():
    """Test SpanNode with attributes."""
    attributes = {"key": "value", "number": 42}
    node = SpanNode(
        span_id="1",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes=attributes,
        events=[],
        parent_id=None,
    )

    assert node.attributes == attributes
    assert node.attributes["key"] == "value"
    assert node.attributes["number"] == 42


def test_span_node_duration():
    """Test SpanNode duration property."""
    node = SpanNode(
        span_id="1",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 5, 500000),  # 5.5 seconds later
        attributes={},
        events=[],
        parent_id=None,
    )

    assert node.duration == timedelta(seconds=5, microseconds=500000)


def test_span_node_has_error():
    """Test SpanNode error detection."""
    # Test with status.code ERROR
    node_error = SpanNode(
        span_id="1",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={"status.code": "ERROR"},
        events=[],
        parent_id=None,
    )
    assert node_error.has_error() is True

    # Test with result.isError
    node_error2 = SpanNode(
        span_id="2",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={"result.isError": True},
        events=[],
        parent_id=None,
    )
    assert node_error2.has_error() is True

    # Test with exception event
    node_error3 = SpanNode(
        span_id="3",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[{"name": "exception", "message": "error"}],
        parent_id=None,
    )
    assert node_error3.has_error() is True

    # Test normal node
    node_ok = SpanNode(
        span_id="4",
        name="test_span",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
    )
    assert node_ok.has_error() is False


def test_span_node_add_child():
    """Test adding children to SpanNode."""
    parent = SpanNode(
        span_id="1",
        name="parent",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
    )

    child = SpanNode(
        span_id="2",
        name="child",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id="1",
    )

    parent.children.append(child)
    assert len(parent.children) == 1
    assert parent.children[0] == child


def test_span_tree_creation():
    """Test SpanTree creation."""
    root = SpanNode(
        span_id="1",
        name="root",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
    )

    tree = SpanTree(root)
    assert tree.root == root


def test_llm_rephrasing_loop():
    """Test LLMRephrasingLoop dataclass."""
    node1 = SpanNode(
        span_id="1",
        name="llm_call1",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
    )

    node2 = SpanNode(
        span_id="2",
        name="llm_call2",
        start_time=datetime(2024, 1, 1, 0, 0, 1),
        end_time=datetime(2024, 1, 1, 0, 0, 2),
        attributes={},
        events=[],
        parent_id=None,
    )

    loop = LLMRephrasingLoop(
        spans=[node1, node2],
        loop_count=2,
        total_duration=timedelta(seconds=2),
        is_stuck=False,
    )

    assert len(loop.spans) == 2
    assert loop.loop_count == 2
    assert loop.total_duration == timedelta(seconds=2)
    assert loop.is_stuck is False


def test_tool_path_analysis():
    """Test ToolPathAnalysis dataclass."""
    analysis = ToolPathAnalysis(
        actual_path=["tool1", "tool3", "tool2"],
        golden_path=["tool1", "tool2"],
        efficiency_score=0.66,
        unnecessary_calls=["tool3"],
        missing_calls=[],
    )

    assert analysis.actual_path == ["tool1", "tool3", "tool2"]
    assert analysis.golden_path == ["tool1", "tool2"]
    assert analysis.efficiency_score == 0.66
    assert analysis.unnecessary_calls == ["tool3"]
    assert analysis.missing_calls == []


def test_error_recovery_sequence():
    """Test ErrorRecoverySequence dataclass."""
    error_span = SpanNode(
        span_id="1",
        name="error",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={"status.code": "ERROR"},
        events=[],
        parent_id=None,
    )

    recovery_span = SpanNode(
        span_id="2",
        name="recovery",
        start_time=datetime(2024, 1, 1, 0, 0, 1),
        end_time=datetime(2024, 1, 1, 0, 0, 2),
        attributes={},
        events=[],
        parent_id=None,
    )

    sequence = ErrorRecoverySequence(
        error_span=error_span,
        recovery_spans=[recovery_span],
        recovery_successful=True,
        recovery_duration=timedelta(seconds=1),
    )

    assert sequence.error_span == error_span
    assert len(sequence.recovery_spans) == 1
    assert sequence.recovery_successful is True
    assert sequence.recovery_duration == timedelta(seconds=1)


def test_span_tree_find():
    """Test finding nodes in SpanTree."""
    child1 = SpanNode(
        span_id="2",
        name="child",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id="1",
    )

    child2 = SpanNode(
        span_id="3",
        name="child",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id="1",
    )

    root = SpanNode(
        span_id="1",
        name="root",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id=None,
        children=[child1, child2],
    )

    tree = SpanTree(root)

    # Find by name
    found = tree.find(lambda n: n.name == "child")
    assert len(found) == 2
    assert child1 in found
    assert child2 in found


def test_span_tree_methods():
    """Test various SpanTree methods."""
    tool_span = SpanNode(
        span_id="2",
        name="call_tool",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={},
        events=[],
        parent_id="1",
    )

    llm_span = SpanNode(
        span_id="3",
        name="llm_call",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 1),
        attributes={"gen_ai.system": "openai"},
        events=[],
        parent_id="1",
    )

    root = SpanNode(
        span_id="1",
        name="root",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 0, 0, 2),
        attributes={},
        events=[],
        parent_id=None,
        children=[tool_span, llm_span],
    )

    tree = SpanTree(root)

    # Test get_tool_calls
    tool_calls = tree.get_tool_calls()
    assert len(tool_calls) == 1
    assert tool_span in tool_calls

    # Test get_llm_calls
    llm_calls = tree.get_llm_calls()
    assert len(llm_calls) == 1
    assert llm_span in llm_calls

    # Test all_spans
    all_spans = tree.all_spans()
    assert len(all_spans) == 3

    # Test any
    assert tree.any(lambda n: n.name == "root") is True
    assert tree.any(lambda n: n.name == "nonexistent") is False

    # Test find_first
    first = tree.find_first(lambda n: "call" in n.name)
    assert first == tool_span
