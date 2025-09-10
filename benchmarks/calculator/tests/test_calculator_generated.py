import pytest
import mcp_eval
from mcp_eval import Expect
from mcp_eval.session import TestAgent
from mcp_agent.agents.agent_spec import AgentSpec

# Pin tests to the intended server by configuring a suite-level AgentSpec.
# This avoids relying on whatever the current default agent is in mcpeval.yaml.
@mcp_eval.setup
def _configure_suite_agent():
    mcp_eval.use_agent(
        AgentSpec(
            name="generated-pytest",
            instruction="You are a helpful assistant that can use MCP servers effectively.",
            server_names=["calculator"],
        )
    )

@pytest.mark.asyncio
async def test_basic_addition(agent: TestAgent):
    response = await agent.generate_str("Add 5 and 3 using the special addition tool")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 5, 'b': 3}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=16, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("16", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_add tool with inputs 5 and 3, understands that it doubles the normal addition result (5+3=8, doubled=16), and communicates the final result of 16 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_basic_subtraction(agent: TestAgent):
    response = await agent.generate_str("Subtract 2 from 10 using the special subtraction tool")
    await agent.session.assert_that(Expect.tools.was_called("special_subtract", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_subtract", {'a': 10, 'b': 2}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_subtract", expected_output=4, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("4", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_subtract tool with inputs 10 and 2, understands that it halves the normal subtraction result (10-2=8, halved=4), and communicates the final result of 4 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_basic_multiplication(agent: TestAgent):
    response = await agent.generate_str("Multiply 4 and 6 using the special multiplication tool")
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 4, 'b': 6}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=48, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("48", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_multiply tool with inputs 4 and 6, understands that it doubles the normal multiplication result (4\u00d76=24, doubled=48), and communicates the final result of 48 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_basic_division(agent: TestAgent):
    response = await agent.generate_str("Divide 20 by 4 using the special division tool")
    await agent.session.assert_that(Expect.tools.was_called("special_divide", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_divide", {'a': 20, 'b': 4}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=2.5, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("2.5", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_divide tool with inputs 20 and 4, understands that it halves the normal division result (20\u00f74=5, halved=2.5), and communicates the final result of 2.5 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_zero_addition(agent: TestAgent):
    response = await agent.generate_str("Add 0 and 15 using the special add function")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 0, 'b': 15}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=30, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("30", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_add tool with inputs 0 and 15, properly handles the zero value in addition, understands that it doubles the normal addition result (0+15=15, doubled=30), and communicates the final result of 30 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_negative_numbers_addition(agent: TestAgent):
    response = await agent.generate_str("Add -5 and 8 using special addition")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': -5, 'b': 8}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=6, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("6", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_add tool with inputs -5 and 8, properly handles negative numbers in addition, understands that it doubles the normal addition result (-5+8=3, doubled=6), and communicates the final result of 6 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_negative_result_subtraction(agent: TestAgent):
    response = await agent.generate_str("Subtract 15 from 10 using the special subtraction tool")
    await agent.session.assert_that(Expect.tools.was_called("special_subtract", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_subtract", {'a': 10, 'b': 15}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_subtract", expected_output=-2.5, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("-2.5", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_subtract tool with inputs 10 and 15, properly handles the case where subtraction results in a negative number, understands that it halves the normal subtraction result (10-15=-5, halved=-2.5), and communicates the final result of -2.5 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_large_numbers_multiplication(agent: TestAgent):
    response = await agent.generate_str("Multiply 100 and 50 using special multiplication")
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 100, 'b': 50}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=10000, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("10000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_multiply tool with inputs 100 and 50, properly handles large numbers in multiplication, understands that it doubles the normal multiplication result (100\u00d750=5000, doubled=10000), and communicates the final result of 10000 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_division_with_remainder(agent: TestAgent):
    response = await agent.generate_str("Divide 7 by 3 using the special division tool")
    await agent.session.assert_that(Expect.tools.was_called("special_divide", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_divide", {'a': 7, 'b': 3}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=1.1666666666666667, field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=1.1666666666666667, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("1.16", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_divide tool with inputs 7 and 3, properly handles division that results in a decimal/fraction, understands that it halves the normal division result (7\u00f73=2.333..., halved=1.166...), and communicates the final decimal result clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_chain_operations_add_then_subtract(agent: TestAgent):
    response = await agent.generate_str("First add 10 and 5, then subtract 3 from the result using the special tools")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_subtract", min_times=1))
    await agent.session.assert_that(Expect.tools.sequence(["special_add", "special_subtract"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 10, 'b': 5}))
    await agent.session.assert_that(Expect.tools.called_with("special_subtract", {'a': 30, 'b': 3}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=30, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_subtract", expected_output=13.5, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.content.contains("13.5", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly performs chained operations: first adds 10 and 5 using special_add (result: 30), then subtracts 3 from that result using special_subtract (30-3=27, halved=13.5). The response demonstrates understanding of tool chaining and communicates the final result of 13.5 clearly.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_multiply_by_zero(agent: TestAgent):
    response = await agent.generate_str("Multiply 25 by 0 using the special multiplication tool")
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 25, 'b': 0}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=0, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("0", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_multiply tool with inputs 25 and 0, properly handles multiplication by zero (understanding that any number times zero equals zero), recognizes that even after doubling the result it remains zero (25\u00d70=0, doubled=0), and communicates the final result of 0 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_divide_by_one(agent: TestAgent):
    response = await agent.generate_str("Divide 42 by 1 using special division")
    await agent.session.assert_that(Expect.tools.was_called("special_divide", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_divide", {'a': 42, 'b': 1}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=21, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("21", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_divide tool with inputs 42 and 1, properly handles division by one (understanding that any number divided by one equals itself), recognizes that the special tool halves the result (42\u00f71=42, halved=21), and communicates the final result of 21 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_explanation_request(agent: TestAgent):
    response = await agent.generate_str("Can you explain how the special addition tool works by demonstrating with 6 + 4?")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 6, 'b': 4}))
    await agent.session.assert_that(Expect.content.contains("double", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response explains that the tool adds numbers and doubles the result", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=20, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("20", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("doubles", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("10", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("6", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("4", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(6000.0))
    await agent.session.assert_that(Expect.judge.llm("The response demonstrates understanding by: 1) Using the special_add tool with 6 and 4, 2) Explaining that normal addition would give 10, 3) Clarifying that the special tool doubles this result to get 20, 4) Providing a clear step-by-step explanation of how the special addition tool works differently from regular addition.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_comparison_calculation(agent: TestAgent):
    response = await agent.generate_str("Calculate 8 + 2 and 9 + 1 using special addition and tell me which result is larger")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=2))
    await agent.session.assert_that(Expect.content.contains("same", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 8, 'b': 2}))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 9, 'b': 1}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=20, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=20, field_path=None, match_type="exact", case_sensitive=True, call_index=1))
    await agent.session.assert_that(Expect.content.contains("20", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("equal", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses special_add twice (8+2 and 9+1), calculates both results as 20 (since both normal additions equal 10, doubled to 20), and accurately concludes that both results are the same/equal rather than one being larger than the other.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_complex_word_problem(agent: TestAgent):
    response = await agent.generate_str("I have 12 apples and give away 7. Then I buy twice as many as I have left. How many apples do I end up with? Use the special math tools.")
    await agent.session.assert_that(Expect.tools.was_called("special_subtract", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.judge.llm("Response correctly solves the word problem using the special tools", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("special_subtract", {'a': 12, 'b': 7}))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 2.5, 'b': 2}))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 2.5, 'b': 10}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_subtract", expected_output=2.5, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=10, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=25, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.sequence(["special_subtract", "special_multiply", "special_add"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("25", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("2.5", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("10", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(6))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly solves the multi-step word problem: 1) Uses special_subtract to find remaining apples after giving away 7 (12-7=5, halved=2.5), 2) Uses special_multiply to calculate buying twice as many (2.5\u00d72=5, doubled=10), 3) Uses special_add to find total apples (2.5+10=12.5, doubled=25). The response demonstrates understanding of the special tool modifications and provides clear step-by-step reasoning.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_negative_multiplication(agent: TestAgent):
    response = await agent.generate_str("Multiply -3 by 7 using the special multiplication tool")
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': -3, 'b': 7}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=-42, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("-42", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_multiply tool with inputs -3 and 7, properly handles negative number multiplication, understands that it doubles the normal multiplication result (-3\u00d77=-21, doubled=-42), and communicates the final result of -42 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_double_negative_addition(agent: TestAgent):
    response = await agent.generate_str("Add -8 and -12 using special addition")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': -8, 'b': -12}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=-40, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("-40", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_add tool with inputs -8 and -12, properly handles double negative addition, understands that it doubles the normal addition result (-8+(-12)=-20, doubled=-40), and communicates the final result of -40 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fraction_division(agent: TestAgent):
    response = await agent.generate_str("Divide 15 by 6 using the special division tool")
    await agent.session.assert_that(Expect.tools.was_called("special_divide", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("special_divide", {'a': 15, 'b': 6}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=1.25, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("1.25", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly uses the special_divide tool with inputs 15 and 6, properly handles division that results in a decimal, understands that it halves the normal division result (15\u00f76=2.5, halved=1.25), and communicates the final result of 1.25 clearly to the user.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_all_four_operations(agent: TestAgent):
    response = await agent.generate_str("Demonstrate all four special math operations using the numbers 12 and 3")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_subtract", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_divide", min_times=1))
    await agent.session.assert_that(Expect.judge.llm("Response demonstrates all four operations with 12 and 3", min_score=0.9), response=response)
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 12, 'b': 3}))
    await agent.session.assert_that(Expect.tools.called_with("special_subtract", {'a': 12, 'b': 3}))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 12, 'b': 3}))
    await agent.session.assert_that(Expect.tools.called_with("special_divide", {'a': 12, 'b': 3}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=30, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_subtract", expected_output=4.5, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=72, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_divide", expected_output=2, field_path=None, match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("30", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("4.5", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("72", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("2", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(7))
    await agent.session.assert_that(Expect.performance.response_time_under(12000.0))
    await agent.session.assert_that(Expect.judge.llm("The response demonstrates all four special operations using 12 and 3: special_add (12+3=15, doubled=30), special_subtract (12-3=9, halved=4.5), special_multiply (12\u00d73=36, doubled=72), and special_divide (12\u00f73=4, halved=2). The response clearly explains each operation and its special transformation.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_order_of_operations(agent: TestAgent):
    response = await agent.generate_str("Calculate (5 + 3) * 2 using the special tools in the correct order")
    await agent.session.assert_that(Expect.tools.was_called("special_add", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("special_multiply", min_times=1))
    await agent.session.assert_that(Expect.tools.sequence(["special_add", "special_multiply"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.called_with("special_add", {'a': 5, 'b': 3}))
    await agent.session.assert_that(Expect.tools.called_with("special_multiply", {'a': 16, 'b': 2}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_add", expected_output=16, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="special_multiply", expected_output=64, field_path=None, match_type="exact", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.content.contains("16", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("64", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("parentheses", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response correctly follows order of operations by first calculating the parentheses using special_add (5+3=8, doubled=16), then multiplying by 2 using special_multiply (16\u00d72=32, doubled=64). The response demonstrates understanding of mathematical precedence and the special tool transformations.", min_score=0.85), response=response)


