"""Test MultiCriteriaJudge with enhanced evaluation capabilities."""

import mcp_eval
from mcp_eval import task, setup
from mcp_eval.evaluators.builtin import (
    STANDARD_CRITERIA,
    MultiCriteriaJudge,
    EvaluationCriterion,
)
from mcp_eval.session import TestAgent, TestSession


@setup
def configure_multi_criteria_tests():
    """Configure for multi-criteria judge testing."""
    mcp_eval.use_server("fetch")


@task("Test MultiCriteriaJudge with standard criteria")
async def test_standard_criteria_evaluation(agent: TestAgent, session: TestSession):
    """Test MultiCriteriaJudge using standard evaluation criteria."""
    input = "Say 'hello my name is Sam. I am 15 years old. I love to play futsal and eat mee pok' in chinese"
    response = await agent.generate_str(input)

    # Create custom criteria for web content evaluation
    web_content_criteria = [
        EvaluationCriterion(
            name="Content Accuracy",
            description="Response accurately describes the fetched web content",
            weight=2.0,
            min_score=1.0,
        ),
        EvaluationCriterion(
            name="Task Completion",
            description="Successfully fetched content and provided summary as requested",
            weight=2.0,
            min_score=0.8,
        ),
        EvaluationCriterion(
            name="Response Clarity",
            description="Summary is clear, well-organized, and easy to understand",
            weight=1.0,
            min_score=0.6,
        ),
    ]

    # # Use MultiCriteriaJudge with weighted aggregation
    judge = MultiCriteriaJudge(
        criteria=web_content_criteria,
        aggregate_method="weighted",
        use_cot=True,
        model="claude-3-5-haiku-20241022",
    )

    await session.evaluate_now_async(
        judge,
        input=input,
        response=response,
        name="extraction_quality_assessment",
    )


@task("Test MultiCriteriaJudge with require_all_pass mode")
async def test_require_all_pass_mode(agent: TestAgent, session: TestSession):
    """Test MultiCriteriaJudge with strict all-criteria-pass requirement."""
    input = "Fetch https://httpbin.org/json and explain what type of data it returns"
    response = await agent.generate_str(input)

    # Create strict criteria that all must pass
    strict_criteria = [
        EvaluationCriterion(
            name="Tool Usage",
            description="Used the fetch tool to retrieve the URL",
            weight=1.0,
            min_score=0.9,
        ),
        EvaluationCriterion(
            name="Data Type Recognition",
            description="Correctly identified that it returns JSON data",
            weight=1.0,
            min_score=0.8,
        ),
        EvaluationCriterion(
            name="Complete Response",
            description="Provided a complete explanation of the data format",
            weight=1.0,
            min_score=0.7,
        ),
    ]

    # Use require_all_pass mode with minimum aggregation
    judge = MultiCriteriaJudge(
        criteria=strict_criteria,
        require_all_pass=True,
        aggregate_method="min",
        use_cot=False,
        model="claude-3-5-haiku-20241022",
    )

    await session.evaluate_now_async(
        judge, input=input, response=response, name="strict_all_pass_evaluation"
    )


@task("Test MultiCriteriaJudge with predefined criteria sets")
async def test_predefined_criteria_sets(agent: TestAgent, session: TestSession):
    """Test MultiCriteriaJudge using predefined criteria sets."""
    input = "Fetch content from https://example.com twice using different approaches and compare the results"
    response = await agent.generate_str(input)

    # Test with standard criteria using harmonic mean (penalizes low scores)
    judge = MultiCriteriaJudge(
        criteria=STANDARD_CRITERIA,
        aggregate_method="harmonic_mean",
        include_confidence=True,
        model="claude-3-5-haiku-20241022",
    )

    await session.evaluate_now_async(
        judge, input=input, response=response, name="standard_criteria_harmonic"
    )


@task("Test MultiCriteriaJudge error handling and edge cases")
async def test_error_handling(agent: TestAgent, session: TestSession):
    """Test MultiCriteriaJudge with edge cases and potential errors."""
    input = "Try to fetch from an invalid URL like https://this-definitely-does-not-exist-12345.com"
    response = await agent.generate_str(input)

    # Create criteria for error handling evaluation
    error_handling_criteria = [
        EvaluationCriterion(
            name="Error Recognition",
            description="Correctly identified that the URL is invalid or unreachable",
            weight=2.0,
            min_score=0.8,
            examples={
                "1.0": "Clearly stated the URL is invalid and explained why",
                "0.5": "Mentioned some error but unclear explanation",
                "0.0": "Did not recognize or address the error",
            },
        ),
        EvaluationCriterion(
            name="Graceful Handling",
            description="Handled the error gracefully without crashing or giving unhelpful response",
            weight=1.5,
            min_score=0.7,
        ),
        EvaluationCriterion(
            name="User Communication",
            description="Provided helpful information to the user about what went wrong",
            weight=1.0,
            min_score=0.6,
        ),
    ]

    # Test with different aggregation methods for comparison
    judge_weighted = MultiCriteriaJudge(
        criteria=error_handling_criteria,
        aggregate_method="weighted",
        use_cot=True,
        model="claude-3-5-haiku-20241022",
    )

    await session.evaluate_now_async(
        judge_weighted, input=input, response=response, name="error_handling_weighted"
    )

    judge_min = MultiCriteriaJudge(
        criteria=error_handling_criteria,
        aggregate_method="min",
        use_cot=True,
        model="claude-3-5-haiku-20241022",
    )

    await session.evaluate_now_async(
        judge_min, input=input, response=response, name="error_handling_min"
    )
