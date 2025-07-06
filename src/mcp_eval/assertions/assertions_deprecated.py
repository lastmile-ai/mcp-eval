import json
from mcp_eval.core import TestSession
from mcp_eval.metrics import TestMetrics
from pydantic import BaseModel, Field
from typing import Optional

# --- Pydantic Models for Judge Assertions ---


class JudgeResult(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score from 0.0 to 1.0 on how well the response meets the criteria.",
    )
    reasoning: str = Field(description="A brief explanation for the score.")


class ObjectiveSuccessResult(BaseModel):
    succeeded: bool = Field(
        description="A boolean flag, true if the response successfully accomplishes the objective, false otherwise."
    )
    reasoning: str = Field(description="A brief explanation for the decision.")


class EfficiencyResult(BaseModel):
    is_efficient: bool = Field(
        description="A boolean flag, true if the plan was efficient, false if there were significant redundancies or better paths."
    )
    reasoning: str = Field(
        description="An explanation of any inefficiencies found or a confirmation of efficiency."
    )
    suggested_improvement: Optional[str] = Field(
        None, description="A more efficient plan of action, if applicable."
    )


# --- Assertion Implementations ---


def contains(response: str, substring: str, session: TestSession):
    """Asserts that the response contains a specific substring."""
    name = f"contains('{substring}')"
    try:
        assert substring in response
        session.add_assertion(name, passed=True)
    except AssertionError:
        error_msg = f"Response does not contain '{substring}'"
        session.add_assertion(name, passed=False, error=error_msg)
        raise AssertionError(error_msg)


def tool_was_called(
    tool_name: str, min_times: int = 1, session: TestSession | None = None
):
    """Asserts that a specific tool was called at least `min_times`."""

    def check(metrics: TestMetrics):
        tool_metric = metrics.tool_metrics.get(tool_name)
        assert tool_metric is not None, f"Tool '{tool_name}' was not called."
        assert tool_metric["call_count"] >= min_times, (
            f"Tool '{tool_name}' was called {tool_metric['call_count']} times, expected at least {min_times}."
        )

    check.__name__ = f"tool_was_called('{tool_name}', min_times={min_times})"
    session.add_deferred_assertion(check)


def tool_success_rate_is_above(tool_name: str, min_rate: float, session: TestSession):
    """Asserts that the success rate of a specific tool is above a minimum threshold."""

    def check(metrics: TestMetrics):
        tool_metric = metrics.tool_metrics.get(tool_name)
        assert tool_metric is not None, (
            f"Tool '{tool_name}' was not called, so success rate cannot be calculated."
        )

        call_count = tool_metric.get("call_count", 0)
        assert call_count > 0, (
            f"Tool '{tool_name}' was called 0 times, so success rate cannot be calculated."
        )

        success_count = tool_metric.get("success_count", 0)
        success_rate = success_count / call_count
        assert success_rate >= min_rate, (
            f"Tool '{tool_name}' success rate {success_rate:.2f} is below the minimum of {min_rate:.2f}."
        )

    check.__name__ = f"tool_success_rate_is_above('{tool_name}', {min_rate:.2f})"
    session.add_deferred_assertion(check)


def tool_arguments_match(tool_name: str, expected_args: dict, session: TestSession):
    """Asserts that a specific tool was called with arguments that match the expected values."""

    def check(metrics: TestMetrics):
        tool_call = next(
            (tc for tc in metrics.tool_calls if tc.name == tool_name), None
        )
        assert tool_call is not None, f"Tool '{tool_name}' was not called."

        for key, expected_value in expected_args.items():
            assert key in tool_call.arguments, (
                f"Tool '{tool_name}' was called without argument '{key}'."
            )
            actual_value = tool_call.arguments[key]
            assert actual_value == expected_value, (
                f"Argument '{key}' for tool '{tool_name}' did not match. Expected '{expected_value}', got '{actual_value}'."
            )

    check.__name__ = f"tool_arguments_match('{tool_name}', {json.dumps(expected_args)})"
    session.add_deferred_assertion(check)


def number_of_steps_under(max_steps: int, session: TestSession):
    """Asserts that the number of agent turns (LLM calls) is under a specified limit."""

    def check(metrics: TestMetrics):
        assert metrics.turns <= max_steps, (
            f"Agent took {metrics.turns} steps, which exceeds the limit of {max_steps}."
        )

    check.__name__ = f"number_of_steps_under({max_steps})"
    session.add_deferred_assertion(check)


def cost_under(max_cost: float, session: TestSession):
    """Asserts that the total cost of the test is under a certain amount."""

    def check(metrics: TestMetrics):
        total_cost = sum(llm.cost for llm in metrics.llm_metrics.values())
        assert total_cost < max_cost, (
            f"Total cost ${total_cost:.6f} exceeded the limit of ${max_cost:.6f}"
        )

    check.__name__ = f"cost_under(${max_cost:.6f})"
    session.add_deferred_assertion(check)


def llm_judge(
    response_to_judge: str, rubric: str, min_score: float, session: TestSession
):
    """Uses an LLM to judge a response against a rubric."""

    async def check(_: TestMetrics):
        prompt = f"""
        Please act as an impartial judge. Evaluate the following response based on the provided rubric.
        Provide a score from 0.0 to 1.0, where 1.0 is the best possible score.
        
        Rubric: {rubric}
        
        Response to evaluate:
        ---
        {response_to_judge}
        ---
        
        Return your evaluation as a JSON object with a 'score' and 'reasoning'.
        """

        judge_result = await session.agent.llm.generate_structured(
            prompt, response_model=JudgeResult
        )
        assert judge_result.score >= min_score, (
            f"Judge score {judge_result.score} is below the minimum of {min_score}. Reasoning: {judge_result.reasoning}"
        )

    check.__name__ = f"llm_judge('{rubric[:30]}...', min_score={min_score})"
    session.add_async_deferred_assertion(check)


def objective_succeeded(objective: str, final_response: str, session: TestSession):
    """Uses an LLM to judge if the agent's final response successfully achieved the given objective."""

    async def check(_: TestMetrics):
        prompt = f"""
        Please act as an impartial judge. Your task is to determine if the agent's final response successfully accomplished the initial objective.

        Initial Objective:
        ---
        {objective}
        ---

        Agent's Final Response:
        ---
        {final_response}
        ---

        Did the agent's response fully and correctly achieve the objective?
        Return your evaluation as a JSON object with a 'succeeded' boolean flag and a 'reasoning' string.
        """
        judge_result = await session.agent.llm.generate_structured(
            prompt, response_model=ObjectiveSuccessResult
        )
        assert judge_result.succeeded, (
            f"Objective was not met. Reasoning: {judge_result.reasoning}"
        )

    check.__name__ = f"objective_succeeded('{objective[:30]}...')"
    session.add_async_deferred_assertion(check)


def plan_is_efficient(objective: str, session: TestSession):
    """Uses an LLM to judge if the agent's execution plan was efficient."""

    async def check(metrics: TestMetrics):
        execution_trace = f"Objective: {objective}\n\n"
        execution_trace += f"Total Steps (LLM calls): {metrics.turns}\n"
        execution_trace += "Execution Path:\n"
        for i, tool_call in enumerate(metrics.tool_calls):
            execution_trace += f"{i + 1}. Tool Call: {tool_call.name}\n"
            execution_trace += f"   Arguments: {tool_call.arguments}\n"
            result_str = str(tool_call.result)
            execution_trace += f"   Result: {'(Error)' if tool_call.is_error else result_str[:100] + '...'}\n"

        prompt = f"""
        Please act as an AI agent efficiency expert. Analyze the following execution trace and determine if the agent achieved its objective in an efficient manner.
        
        Consider the following:
        - Were there any redundant or unnecessary tool calls?
        - Could a different tool or a different sequence of actions have achieved the goal in fewer steps?
        - Were the arguments passed to the tools optimal?

        Execution Trace:
        ---
        {execution_trace}
        ---

        Based on your analysis, was the plan efficient?
        Return your evaluation as a JSON object with 'is_efficient' (boolean), 'reasoning' (string), and an optional 'suggested_improvement' (string).
        """

        judge_result = await session.agent.llm.generate_structured(
            prompt, response_model=EfficiencyResult
        )
        assert judge_result.is_efficient, (
            f"Plan was deemed inefficient. Reasoning: {judge_result.reasoning}"
        )

    check.__name__ = f"plan_is_efficient('{objective[:30]}...')"
    session.add_async_deferred_assertion(check)
