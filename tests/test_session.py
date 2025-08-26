import asyncio
from pathlib import Path

import pytest

from mcp_eval.session import TestSession
from mcp_eval.evaluators.response_contains import ResponseContains
from mcp_eval.evaluators.equals_expected import EqualsExpected
from mcp_eval.evaluators.shared import EvaluatorResult


@pytest.mark.asyncio
async def test_session_lifecycle_and_artifacts(tmp_path):
    session = TestSession(test_name="lifecycle_test")

    async with session as agent:
        # No LLM is attached by default because provider/model are None in tests
        assert session.agent is not None
        assert session.test_agent is not None

        # Defer an evaluator that doesn't need output
        session.add_deferred_evaluator(ResponseContains("anything"), name="contains")

    # After exit: artifacts should be saved to configured reports dir
    reports_dir = Path(session.get_span_tree() and "unused")  # touch metrics to ensure processed
    config = session.get_results()  # ensure results available
    # Verify that results file exists in ./test-reports
    reports_root = Path("./test-reports")
    assert reports_root.exists()
    # One results file should exist
    files = list(reports_root.glob("*.json"))
    assert files, "Expected a results json to be written"


@pytest.mark.asyncio
async def test_session_assert_now_and_end(tmp_path):
    # Immediate evaluation with provided response using sync evaluator
    session = TestSession(test_name="assert_now")
    async with session as agent:
        await session.assert_that(EqualsExpected(exact_match=True), response="ok")

        # Defer to end (requires_final_metrics via when="end")
        await session.assert_that(ResponseContains("ok"), response="ok", when="end")

    results = session.get_results()
    assert any(r.name for r in results)


@pytest.mark.asyncio
async def test_session_context_manager_helper_alias():
    # Ensure test_session asynccontextmanager works and cleans up
    from mcp_eval import test_session as ts_ctx

    async with ts_ctx("ctx_helper") as agent:
        assert agent is not None


