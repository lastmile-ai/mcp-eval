import pytest

from mcp_eval.datasets import Case, Dataset
from mcp_eval.evaluators.equals_expected import EqualsExpected


@pytest.mark.asyncio
async def test_dataset_evaluate_simple_flow():
    # Define a simple dataset with one case and EqualsExpected evaluator
    case = Case(
        name="c1", inputs="ping", expected_output="pong", evaluators=[EqualsExpected()]
    )
    ds = Dataset(name="ds1", cases=[case])

    async def task_func(inp, agent, session):
        # Return deterministic output
        return "pong" if inp == "ping" else "unknown"

    report = await ds.evaluate(task_func)
    assert report.dataset_name == "ds1"
    assert report.task_name == task_func.__name__
    assert len(report.results) == 1
    assert report.results[0].passed is True
