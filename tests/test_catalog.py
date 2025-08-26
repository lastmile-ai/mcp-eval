"""Tests for the catalog module."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Pattern
import re

from mcp_eval.catalog import (
    Expect,
    Content,
    Tools,
    Performance,
    Judge,
    Path,
)
from mcp_eval.evaluators import (
    ResponseContains,
    NotContains,
    ToolWasCalled,
    ToolCalledWith,
    ExactToolCount,
    ToolSuccessRate,
    ToolFailed,
    ToolOutputMatches,
    ToolSequence,
    MaxIterations,
    ResponseTimeCheck,
    LLMJudge,
    PathEfficiency,
    MultiCriteriaJudge,
    EvaluationCriterion,
)


class TestContent:
    """Test the Content class methods."""

    def test_contains(self):
        """Test Content.contains method."""
        evaluator = Content.contains("test text")
        assert isinstance(evaluator, ResponseContains)
        assert evaluator.text == "test text"
        assert evaluator.case_sensitive is False

    def test_contains_case_sensitive(self):
        """Test Content.contains with case sensitivity."""
        evaluator = Content.contains("Test Text", case_sensitive=True)
        assert isinstance(evaluator, ResponseContains)
        assert evaluator.text == "Test Text"
        assert evaluator.case_sensitive is True

    def test_not_contains(self):
        """Test Content.not_contains method."""
        evaluator = Content.not_contains("forbidden text")
        assert isinstance(evaluator, NotContains)
        assert evaluator.text == "forbidden text"
        assert evaluator.case_sensitive is False

    def test_not_contains_case_sensitive(self):
        """Test Content.not_contains with case sensitivity."""
        evaluator = Content.not_contains("Forbidden", case_sensitive=True)
        assert isinstance(evaluator, NotContains)
        assert evaluator.text == "Forbidden"
        assert evaluator.case_sensitive is True

    def test_regex(self):
        """Test Content.regex method."""
        evaluator = Content.regex(r"\d{3}-\d{4}")
        assert isinstance(evaluator, ResponseContains)
        assert evaluator.text == r"\d{3}-\d{4}"
        assert evaluator.regex is True
        assert evaluator.case_sensitive is False

    def test_regex_case_sensitive(self):
        """Test Content.regex with case sensitivity."""
        evaluator = Content.regex(r"[A-Z]+", case_sensitive=True)
        assert isinstance(evaluator, ResponseContains)
        assert evaluator.text == r"[A-Z]+"
        assert evaluator.regex is True
        assert evaluator.case_sensitive is True


class TestTools:
    """Test the Tools class methods."""

    def test_was_called(self):
        """Test Tools.was_called method."""
        evaluator = Tools.was_called("fetch")
        assert isinstance(evaluator, ToolWasCalled)
        assert evaluator.tool_name == "fetch"
        assert evaluator.min_times == 1

    def test_was_called_min_times(self):
        """Test Tools.was_called with min_times."""
        evaluator = Tools.was_called("process", min_times=3)
        assert isinstance(evaluator, ToolWasCalled)
        assert evaluator.tool_name == "process"
        assert evaluator.min_times == 3

    def test_called_with(self):
        """Test Tools.called_with method."""
        args = {"url": "https://example.com", "method": "GET"}
        evaluator = Tools.called_with("http_request", args)
        assert isinstance(evaluator, ToolCalledWith)
        assert evaluator.tool_name == "http_request"
        assert evaluator.expected_args == args

    def test_count(self):
        """Test Tools.count method."""
        evaluator = Tools.count("calculate", 5)
        assert isinstance(evaluator, ExactToolCount)
        assert evaluator.tool_name == "calculate"
        assert evaluator.expected_count == 5

    def test_success_rate(self):
        """Test Tools.success_rate method."""
        evaluator = Tools.success_rate(0.95)
        assert isinstance(evaluator, ToolSuccessRate)
        assert evaluator.min_rate == 0.95
        assert evaluator.tool_name is None

    def test_success_rate_with_tool(self):
        """Test Tools.success_rate with specific tool."""
        evaluator = Tools.success_rate(0.8, tool_name="api_call")
        assert isinstance(evaluator, ToolSuccessRate)
        assert evaluator.min_rate == 0.8
        assert evaluator.tool_name == "api_call"

    def test_failed(self):
        """Test Tools.failed method."""
        evaluator = Tools.failed("broken_tool")
        assert isinstance(evaluator, ToolFailed)
        assert evaluator.tool_name == "broken_tool"
        assert evaluator.min_rate == 0.0

    def test_output_matches_dict(self):
        """Test Tools.output_matches with dict expected output."""
        expected = {"status": "success", "data": {"id": 123}}
        evaluator = Tools.output_matches("api_call", expected)
        assert isinstance(evaluator, ToolOutputMatches)
        assert evaluator.tool_name == "api_call"
        assert evaluator.expected_output == expected
        assert evaluator.match_type == "exact"
        assert evaluator.case_sensitive is True
        assert evaluator.call_index == -1

    def test_output_matches_with_options(self):
        """Test Tools.output_matches with all options."""
        evaluator = Tools.output_matches(
            "fetch",
            "expected text",
            field_path="response.body",
            match_type="contains",
            case_sensitive=False,
            call_index=0,
        )
        assert isinstance(evaluator, ToolOutputMatches)
        assert evaluator.tool_name == "fetch"
        assert evaluator.expected_output == "expected text"
        assert evaluator.field_path == "response.body"
        assert evaluator.match_type == "contains"
        assert evaluator.case_sensitive is False
        assert evaluator.call_index == 0

    def test_output_matches_pattern(self):
        """Test Tools.output_matches with Pattern."""
        pattern = re.compile(r"\d+")
        evaluator = Tools.output_matches("extract", pattern)
        assert isinstance(evaluator, ToolOutputMatches)
        assert evaluator.expected_output == pattern

    def test_sequence(self):
        """Test Tools.sequence method."""
        seq = ["auth", "fetch", "process", "save"]
        evaluator = Tools.sequence(seq)
        assert isinstance(evaluator, ToolSequence)
        assert evaluator.expected_sequence == seq
        assert evaluator.allow_other_calls is False

    def test_sequence_allow_other_calls(self):
        """Test Tools.sequence with allow_other_calls."""
        seq = ["start", "end"]
        evaluator = Tools.sequence(seq, allow_other_calls=True)
        assert isinstance(evaluator, ToolSequence)
        assert evaluator.expected_sequence == seq
        assert evaluator.allow_other_calls is True


class TestPerformance:
    """Test the Performance class methods."""

    def test_max_iterations(self):
        """Test Performance.max_iterations method."""
        evaluator = Performance.max_iterations(10)
        assert isinstance(evaluator, MaxIterations)
        assert evaluator.max_iterations == 10

    def test_response_time_under(self):
        """Test Performance.response_time_under method."""
        evaluator = Performance.response_time_under(500)
        assert isinstance(evaluator, ResponseTimeCheck)
        assert evaluator.max_ms == 500


class TestJudge:
    """Test the Judge class methods."""

    def test_llm_default(self):
        """Test Judge.llm with default parameters."""
        rubric = "The response should be helpful and accurate."
        evaluator = Judge.llm(rubric)
        assert isinstance(evaluator, LLMJudge)
        assert evaluator.rubric == rubric
        assert evaluator.min_score == 0.8
        assert evaluator.include_input is False
        assert evaluator.require_reasoning is True

    def test_llm_custom_params(self):
        """Test Judge.llm with custom parameters."""
        rubric = "Custom rubric"
        evaluator = Judge.llm(
            rubric,
            min_score=0.6,
            include_input=True,
            require_reasoning=False,
        )
        assert isinstance(evaluator, LLMJudge)
        assert evaluator.rubric == rubric
        assert evaluator.min_score == 0.6
        assert evaluator.include_input is True
        assert evaluator.require_reasoning is False

    def test_multi_criteria_with_dict(self):
        """Test Judge.multi_criteria with dict input."""
        criteria = {
            "accuracy": "The response should be factually correct",
            "completeness": "The response should cover all aspects",
        }
        evaluator = Judge.multi_criteria(criteria)
        assert isinstance(evaluator, MultiCriteriaJudge)
        assert len(evaluator.criteria) == 2
        assert evaluator.aggregate_method == "weighted"
        assert evaluator.require_all_pass is False
        assert evaluator.include_confidence is True
        assert evaluator.use_cot is True

    def test_multi_criteria_with_list(self):
        """Test Judge.multi_criteria with EvaluationCriterion list."""
        criteria = [
            EvaluationCriterion(
                name="relevance",
                description="Response should be relevant",
                weight=1.0,
                min_score=0.7,
            ),
            EvaluationCriterion(
                name="clarity",
                description="Response should be clear",
                weight=0.5,
                min_score=0.8,
            ),
        ]
        evaluator = Judge.multi_criteria(criteria)
        assert isinstance(evaluator, MultiCriteriaJudge)
        assert len(evaluator.criteria) == 2
        assert evaluator.criteria[0].name == "relevance"
        assert evaluator.criteria[1].name == "clarity"

    def test_multi_criteria_custom_params(self):
        """Test Judge.multi_criteria with custom parameters."""
        criteria = {"test": "Test criterion"}
        evaluator = Judge.multi_criteria(
            criteria,
            aggregate_method="average",
            require_all_pass=True,
            include_confidence=False,
            use_cot=False,
            model="gpt-4",
        )
        assert isinstance(evaluator, MultiCriteriaJudge)
        assert evaluator.aggregate_method == "average"
        assert evaluator.require_all_pass is True
        assert evaluator.include_confidence is False
        assert evaluator.use_cot is False
        assert evaluator.model == "gpt-4"


class TestPath:
    """Test the Path class methods."""

    def test_efficiency_default(self):
        """Test Path.efficiency with default parameters."""
        evaluator = Path.efficiency()
        assert isinstance(evaluator, PathEfficiency)
        assert evaluator.optimal_steps is None
        assert evaluator.expected_tool_sequence is None
        assert evaluator.allow_extra_steps == 0
        assert evaluator.penalize_backtracking is True
        assert evaluator.penalize_repeated_tools is True
        assert evaluator.tool_usage_limits is None
        assert evaluator.default_tool_limit == 1

    def test_efficiency_with_optimal_steps(self):
        """Test Path.efficiency with optimal_steps."""
        evaluator = Path.efficiency(optimal_steps=5)
        assert isinstance(evaluator, PathEfficiency)
        assert evaluator.optimal_steps == 5

    def test_efficiency_with_sequence(self):
        """Test Path.efficiency with expected_tool_sequence."""
        sequence = ["init", "process", "save"]
        evaluator = Path.efficiency(expected_tool_sequence=sequence)
        assert isinstance(evaluator, PathEfficiency)
        assert evaluator.expected_tool_sequence == sequence

    def test_efficiency_full_params(self):
        """Test Path.efficiency with all parameters."""
        sequence = ["start", "middle", "end"]
        limits = {"expensive_tool": 1, "normal_tool": 3}
        evaluator = Path.efficiency(
            optimal_steps=10,
            expected_tool_sequence=sequence,
            allow_extra_steps=2,
            penalize_backtracking=False,
            penalize_repeated_tools=False,
            tool_usage_limits=limits,
            default_tool_limit=2,
        )
        assert isinstance(evaluator, PathEfficiency)
        assert evaluator.optimal_steps == 10
        assert evaluator.expected_tool_sequence == sequence
        assert evaluator.allow_extra_steps == 2
        assert evaluator.penalize_backtracking is False
        assert evaluator.penalize_repeated_tools is False
        assert evaluator.tool_usage_limits == limits
        assert evaluator.default_tool_limit == 2


class TestExpect:
    """Test the Expect namespace class."""

    def test_expect_has_all_namespaces(self):
        """Test that Expect has all sub-namespaces."""
        assert hasattr(Expect, "content")
        assert hasattr(Expect, "tools")
        assert hasattr(Expect, "performance")
        assert hasattr(Expect, "judge")
        assert hasattr(Expect, "path")

    def test_expect_namespaces_are_classes(self):
        """Test that Expect namespaces are the correct classes."""
        assert Expect.content is Content
        assert Expect.tools is Tools
        assert Expect.performance is Performance
        assert Expect.judge is Judge
        assert Expect.path is Path

    def test_expect_usage_examples(self):
        """Test common usage patterns with Expect."""
        # Content assertions
        evaluator = Expect.content.contains("test")
        assert isinstance(evaluator, ResponseContains)

        # Tools assertions
        evaluator = Expect.tools.was_called("fetch")
        assert isinstance(evaluator, ToolWasCalled)

        # Performance assertions
        evaluator = Expect.performance.max_iterations(5)
        assert isinstance(evaluator, MaxIterations)

        # Judge assertions
        evaluator = Expect.judge.llm("Test rubric")
        assert isinstance(evaluator, LLMJudge)

        # Path assertions
        evaluator = Expect.path.efficiency(optimal_steps=3)
        assert isinstance(evaluator, PathEfficiency)

    def test_chaining_evaluators(self):
        """Test that evaluators can be created and used independently."""
        evaluators = [
            Expect.content.contains("success"),
            Expect.tools.was_called("api_call", min_times=2),
            Expect.performance.response_time_under(1000),
        ]

        assert len(evaluators) == 3
        assert isinstance(evaluators[0], ResponseContains)
        assert isinstance(evaluators[1], ToolWasCalled)
        assert isinstance(evaluators[2], ResponseTimeCheck)