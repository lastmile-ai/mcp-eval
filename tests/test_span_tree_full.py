"""Comprehensive tests for otel/span_tree.py to achieve >80% coverage."""

from mcp_eval.otel.span_tree import (
    SpanNode,
    SpanTree,
    build_span_tree,
    format_span_tree,
    get_tool_calls_from_tree,
    get_llm_calls_from_tree,
    find_spans_by_name,
    find_spans_by_attribute,
    calculate_tree_depth,
    calculate_total_duration,
    get_error_spans,
)


def create_test_span(
    span_id="1",
    parent_id=None,
    name="test_span",
    attributes=None,
    start_time=None,
    end_time=None,
):
    """Helper to create test span data."""
    return {
        "span_id": span_id,
        "parent_span_id": parent_id,
        "name": name,
        "attributes": attributes or {},
        "start_time": start_time or "2024-01-01T00:00:00.000000",
        "end_time": end_time or "2024-01-01T00:00:01.000000",
        "status": {"status_code": "OK"},
        "events": [],
    }


def test_span_node_creation():
    """Test SpanNode creation."""
    span_data = create_test_span()
    node = SpanNode(span_data)

    assert node.span_id == "1"
    assert node.parent_id is None
    assert node.name == "test_span"
    assert node.children == []
    assert node.attributes == {}


def test_span_node_with_attributes():
    """Test SpanNode with attributes."""
    attributes = {"key": "value", "number": 42}
    span_data = create_test_span(attributes=attributes)
    node = SpanNode(span_data)

    assert node.attributes == attributes
    assert node.get_attribute("key") == "value"
    assert node.get_attribute("number") == 42
    assert node.get_attribute("missing") is None
    assert node.get_attribute("missing", "default") == "default"


def test_span_node_add_child():
    """Test adding child to SpanNode."""
    parent = SpanNode(create_test_span(span_id="1"))
    child = SpanNode(create_test_span(span_id="2", parent_id="1"))

    parent.add_child(child)

    assert len(parent.children) == 1
    assert parent.children[0] == child


def test_span_node_duration():
    """Test SpanNode duration calculation."""
    span_data = create_test_span(
        start_time="2024-01-01T00:00:00.000000", end_time="2024-01-01T00:00:05.500000"
    )
    node = SpanNode(span_data)

    assert node.duration_ms == 5500.0


def test_span_node_is_error():
    """Test SpanNode error detection."""
    # OK status
    ok_span = create_test_span()
    ok_node = SpanNode(ok_span)
    assert not ok_node.is_error()

    # ERROR status
    error_span = create_test_span()
    error_span["status"] = {"status_code": "ERROR", "message": "Test error"}
    error_node = SpanNode(error_span)
    assert error_node.is_error()


def test_span_node_is_tool_call():
    """Test SpanNode tool call detection."""
    # Regular span
    regular = SpanNode(create_test_span(name="regular_span"))
    assert not regular.is_tool_call()

    # Tool call span
    tool_span = create_test_span(
        name="tool_call", attributes={"tool_name": "test_tool"}
    )
    tool_node = SpanNode(tool_span)
    assert tool_node.is_tool_call()


def test_span_node_is_llm_call():
    """Test SpanNode LLM call detection."""
    # Regular span
    regular = SpanNode(create_test_span(name="regular_span"))
    assert not regular.is_llm_call()

    # LLM call span
    llm_span = create_test_span(name="llm_generate")
    llm_node = SpanNode(llm_span)
    assert llm_node.is_llm_call()

    # Another LLM pattern
    llm_span2 = create_test_span(name="openai.completion")
    llm_node2 = SpanNode(llm_span2)
    assert llm_node2.is_llm_call()


def test_span_tree_creation():
    """Test SpanTree creation."""
    root = SpanNode(create_test_span(span_id="1"))
    tree = SpanTree(root)

    assert tree.root == root
    assert tree.get_node("1") == root
    assert tree.get_node("nonexistent") is None


def test_span_tree_add_node():
    """Test adding nodes to SpanTree."""
    root = SpanNode(create_test_span(span_id="1"))
    tree = SpanTree(root)

    child = SpanNode(create_test_span(span_id="2", parent_id="1"))
    tree.add_node(child)

    assert tree.get_node("2") == child
    assert child in root.children


def test_span_tree_find_nodes():
    """Test finding nodes in SpanTree."""
    root = SpanNode(create_test_span(span_id="1", name="root"))
    tree = SpanTree(root)

    child1 = SpanNode(create_test_span(span_id="2", parent_id="1", name="child"))
    child2 = SpanNode(create_test_span(span_id="3", parent_id="1", name="child"))
    tree.add_node(child1)
    tree.add_node(child2)

    # Find by name
    found = tree.find_nodes(lambda n: n.name == "child")
    assert len(found) == 2
    assert child1 in found
    assert child2 in found


def test_span_tree_traverse():
    """Test traversing SpanTree."""
    root = SpanNode(create_test_span(span_id="1"))
    tree = SpanTree(root)

    child1 = SpanNode(create_test_span(span_id="2", parent_id="1"))
    child2 = SpanNode(create_test_span(span_id="3", parent_id="1"))
    grandchild = SpanNode(create_test_span(span_id="4", parent_id="2"))

    tree.add_node(child1)
    tree.add_node(child2)
    tree.add_node(grandchild)

    visited = []
    tree.traverse(lambda n: visited.append(n.span_id))

    assert len(visited) == 4
    assert "1" in visited
    assert "2" in visited
    assert "3" in visited
    assert "4" in visited


def test_build_span_tree():
    """Test building span tree from spans."""
    spans = [
        create_test_span(span_id="1"),
        create_test_span(span_id="2", parent_id="1"),
        create_test_span(span_id="3", parent_id="1"),
        create_test_span(span_id="4", parent_id="2"),
    ]

    tree = build_span_tree(spans)

    assert tree.root.span_id == "1"
    assert len(tree.root.children) == 2
    assert tree.get_node("4").parent_id == "2"


def test_build_span_tree_multiple_roots():
    """Test building span tree with multiple roots."""
    spans = [
        create_test_span(span_id="1"),
        create_test_span(span_id="2"),  # Another root
        create_test_span(span_id="3", parent_id="1"),
    ]

    tree = build_span_tree(spans)

    # Should create synthetic root
    assert tree.root.name == "root"
    assert len(tree.root.children) == 2


def test_build_span_tree_empty():
    """Test building span tree with empty spans."""
    tree = build_span_tree([])
    assert tree is None


def test_format_span_tree():
    """Test formatting span tree."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", name="child1"),
        create_test_span(span_id="3", parent_id="1", name="child2"),
    ]

    tree = build_span_tree(spans)
    formatted = format_span_tree(tree)

    assert "root" in formatted
    assert "child1" in formatted
    assert "child2" in formatted
    assert "├──" in formatted or "└──" in formatted


def test_format_span_tree_with_details():
    """Test formatting span tree with details."""
    spans = [
        create_test_span(
            span_id="1",
            name="root",
            start_time="2024-01-01T00:00:00.000000",
            end_time="2024-01-01T00:00:01.500000",
        ),
    ]

    tree = build_span_tree(spans)
    formatted = format_span_tree(tree, show_duration=True, show_attributes=True)

    assert "1500.0ms" in formatted or "1500ms" in formatted


def test_get_tool_calls_from_tree():
    """Test extracting tool calls from tree."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(
            span_id="2",
            parent_id="1",
            name="tool_call",
            attributes={"tool_name": "test_tool", "arguments": {"arg": "value"}},
        ),
        create_test_span(span_id="3", parent_id="1", name="regular_span"),
    ]

    tree = build_span_tree(spans)
    tool_calls = get_tool_calls_from_tree(tree)

    assert len(tool_calls) == 1
    assert tool_calls[0].span_id == "2"


def test_get_llm_calls_from_tree():
    """Test extracting LLM calls from tree."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", name="llm_generate"),
        create_test_span(span_id="3", parent_id="1", name="regular_span"),
        create_test_span(span_id="4", parent_id="1", name="openai.completion"),
    ]

    tree = build_span_tree(spans)
    llm_calls = get_llm_calls_from_tree(tree)

    assert len(llm_calls) == 2
    assert llm_calls[0].span_id == "2"
    assert llm_calls[1].span_id == "4"


def test_find_spans_by_name():
    """Test finding spans by name."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", name="target"),
        create_test_span(span_id="3", parent_id="1", name="other"),
        create_test_span(span_id="4", parent_id="2", name="target"),
    ]

    tree = build_span_tree(spans)
    found = find_spans_by_name(tree, "target")

    assert len(found) == 2
    assert found[0].span_id == "2"
    assert found[1].span_id == "4"


def test_find_spans_by_attribute():
    """Test finding spans by attribute."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", attributes={"key": "value1"}),
        create_test_span(span_id="3", parent_id="1", attributes={"key": "value2"}),
        create_test_span(span_id="4", parent_id="1", attributes={"other": "data"}),
    ]

    tree = build_span_tree(spans)
    found = find_spans_by_attribute(tree, "key", "value1")

    assert len(found) == 1
    assert found[0].span_id == "2"


def test_calculate_tree_depth():
    """Test calculating tree depth."""
    # Single node
    tree1 = SpanTree(SpanNode(create_test_span(span_id="1")))
    assert calculate_tree_depth(tree1) == 1

    # Two levels
    spans2 = [
        create_test_span(span_id="1"),
        create_test_span(span_id="2", parent_id="1"),
    ]
    tree2 = build_span_tree(spans2)
    assert calculate_tree_depth(tree2) == 2

    # Three levels
    spans3 = [
        create_test_span(span_id="1"),
        create_test_span(span_id="2", parent_id="1"),
        create_test_span(span_id="3", parent_id="2"),
    ]
    tree3 = build_span_tree(spans3)
    assert calculate_tree_depth(tree3) == 3


def test_calculate_total_duration():
    """Test calculating total duration."""
    spans = [
        create_test_span(
            span_id="1",
            start_time="2024-01-01T00:00:00.000000",
            end_time="2024-01-01T00:00:02.000000",
        ),
        create_test_span(
            span_id="2",
            parent_id="1",
            start_time="2024-01-01T00:00:00.500000",
            end_time="2024-01-01T00:00:01.000000",
        ),
        create_test_span(
            span_id="3",
            parent_id="1",
            start_time="2024-01-01T00:00:01.000000",
            end_time="2024-01-01T00:00:01.500000",
        ),
    ]

    tree = build_span_tree(spans)
    total_duration = calculate_total_duration(tree)

    assert total_duration == 2000.0  # 2 seconds in ms


def test_get_error_spans():
    """Test getting error spans from tree."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", name="success"),
        create_test_span(span_id="3", parent_id="1", name="error1"),
        create_test_span(span_id="4", parent_id="2", name="error2"),
    ]

    # Mark some spans as errors
    spans[2]["status"] = {"status_code": "ERROR", "message": "Error 1"}
    spans[3]["status"] = {"status_code": "ERROR", "message": "Error 2"}

    tree = build_span_tree(spans)
    error_spans = get_error_spans(tree)

    assert len(error_spans) == 2
    assert error_spans[0].span_id == "3"
    assert error_spans[1].span_id == "4"


def test_span_node_events():
    """Test SpanNode with events."""
    span_data = create_test_span()
    span_data["events"] = [
        {"name": "event1", "timestamp": "2024-01-01T00:00:00.500000"},
        {"name": "event2", "timestamp": "2024-01-01T00:00:00.800000"},
    ]

    node = SpanNode(span_data)
    assert len(node.events) == 2
    assert node.events[0]["name"] == "event1"


def test_span_node_status():
    """Test SpanNode status handling."""
    # OK status
    ok_span = create_test_span()
    ok_node = SpanNode(ok_span)
    assert ok_node.status["status_code"] == "OK"

    # ERROR status with message
    error_span = create_test_span()
    error_span["status"] = {"status_code": "ERROR", "message": "Something went wrong"}
    error_node = SpanNode(error_span)
    assert error_node.status["status_code"] == "ERROR"
    assert error_node.status["message"] == "Something went wrong"


def test_span_tree_to_dict():
    """Test converting SpanTree to dict."""
    spans = [
        create_test_span(span_id="1", name="root"),
        create_test_span(span_id="2", parent_id="1", name="child"),
    ]

    tree = build_span_tree(spans)
    tree_dict = tree.to_dict()

    assert "root" in tree_dict
    assert tree_dict["root"]["span_id"] == "1"
    assert "children" in tree_dict["root"]
    assert len(tree_dict["root"]["children"]) == 1


def test_span_node_missing_times():
    """Test SpanNode with missing timestamps."""
    span_data = {
        "span_id": "1",
        "name": "test",
        "attributes": {},
        "status": {"status_code": "OK"},
        "events": [],
    }

    node = SpanNode(span_data)
    assert node.start_time is None
    assert node.end_time is None
    assert node.duration_ms == 0


def test_build_span_tree_orphan_spans():
    """Test building tree with orphan spans."""
    spans = [
        create_test_span(span_id="1"),
        create_test_span(span_id="2", parent_id="nonexistent"),  # Orphan
        create_test_span(span_id="3", parent_id="1"),
    ]

    tree = build_span_tree(spans)

    # Orphan should be attached to synthetic root
    assert tree.root.name == "root"
    assert len(tree.root.children) == 2  # Original root + orphan


def test_span_tree_max_depth():
    """Test tree with very deep nesting."""
    spans = []
    for i in range(10):
        parent_id = str(i) if i > 0 else None
        spans.append(create_test_span(span_id=str(i + 1), parent_id=parent_id))

    tree = build_span_tree(spans)
    depth = calculate_tree_depth(tree)
    assert depth == 10
