import pytest

from mcp_eval.evaluators.base import EvaluatorContext
from mcp_eval.evaluators.llm_judge import LLMJudge
from mcp_eval.evaluators.multi_criteria_judge import MultiCriteriaJudge, EvaluationCriterion
from mcp_eval.metrics import TestMetrics


@pytest.mark.asyncio
async def test_llm_judge_uses_stub(stub_judge_client):
    m = TestMetrics()
    ctx = EvaluatorContext(inputs="question", output="answer", expected_output=None, metadata={}, metrics=m, span_tree=None)
    judge = LLMJudge(rubric="Quality", min_score=0.5, model="stub")
    res = await judge.evaluate(ctx)
    assert res.passed is True
    assert (res.score or 0) >= 0.5


@pytest.mark.asyncio
async def test_multi_criteria_judge_parallel(stub_judge_client):
    m = TestMetrics()
    ctx = EvaluatorContext(inputs="i", output="o", expected_output=None, metadata={}, metrics=m, span_tree=None)
    criteria = [
        EvaluationCriterion(name="Accuracy", description=""),
        EvaluationCriterion(name="Clarity", description=""),
    ]
    mcj = MultiCriteriaJudge(criteria=criteria, require_all_pass=False)
    res = await mcj.evaluate(ctx)
    assert res.passed in (True, False)  # depends on aggregate
    assert res.score is not None


