from mcp_eval.catalog import Expect


def test_catalog_factories_create_evaluators():
    e1 = Expect.content.contains("abc")
    assert e1.text == "abc"

    e2 = Expect.tools.was_called("fetch")
    assert e2.tool_name == "fetch"

    e3 = Expect.performance.max_iterations(3)
    # has attribute from evaluator type
    assert hasattr(e3, "max_iterations")

    e4 = Expect.judge.llm("rubric text", min_score=0.2)
    assert e4.rubric and e4.min_score == 0.2

    e5 = Expect.path.efficiency(optimal_steps=2)
    assert e5.optimal_steps == 2


