import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_Summarize_Article_Concisely(agent: TestAgent):
    response = await agent.generate_str("Read the following article and provide a concise summary in 2-3 sentences. Do not include any internal chain-of-thought or references to being an AI.\n\nArticle:\n\"Researchers discovered a new technique that improves battery life by 30%...\"")
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(2000.0))
    await agent.session.assert_that(Expect.judge.llm("Score from 0.0 to 1.0. The summary should be (1) accurate: reflect main points of the article, (2) concise: 2-3 sentences, and (3) no model self-reference or chain-of-thought. Give 0.0 for completely incorrect or revealing chain-of-thought, 1.0 for accurate, concise, and no self-reference.", min_score=0.9), response=response)
    await agent.session.assert_that(Expect.tools.was_called("search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("calculator", {'expression': '2+2', 'precision': 0}))
    await agent.session.assert_that(Expect.content.contains("Thank you", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="weather_api", expected_output={'temp': 20, 'unit': 'C'}, field_path='data.current', match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.performance.response_time_under(500.0))
    await agent.session.assert_that(Expect.judge.llm("Score the assistant response 0-1 based on correctness, helpfulness, and clarity. Responses scoring \u003e= 0.9 pass.", min_score=0.9), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["search", "browser.open", "summarizer"], allow_other_calls=True))

@pytest.mark.asyncio
async def test_Fetch_Current_Weather_for_Paris(agent: TestAgent):
    response = await agent.generate_str("What is the current weather in Paris? Use the get_weather tool and then present a brief readable answer: temperature and condition.")
    await agent.session.assert_that(Expect.tools.called_with("get_weather", {'location': 'Paris', 'units': 'metric'}))
    await agent.session.assert_that(Expect.tools.was_called("get_weather", min_times=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_weather", expected_output={'location': 'Paris'}, field_path='location', match_type="exact", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.content.contains("\u00b0C", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url_pattern': '(?i)(paris|weather|meteo|openweathermap|weather.com)', 'note': 'URL or query should reference Paris or a weather provider (case-insensitive regex).'}))
    await agent.session.assert_that(Expect.tools.sequence(["fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output=['Paris', 'temperature'], field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.content.contains("temperature", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("condition", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("Score (0..1) averaged over criteria: 1) Tool use: fetched a Paris weather source (0.2). 2) Temperature present and numeric (0.25). 3) Weather condition present (e.g., \u0027clear\u0027, \u0027rain\u0027) (0.25). 4) Conciseness: answer is 1-2 short sentences (0.15). 5) No chain-of-thought or self-reference (0.15). Provide full justification for any score \u003c 0.8.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_Query_User_Email_and_Format(agent: TestAgent):
    response = await agent.generate_str("Fetch the email address for user id 123 from the user database and then pass the email through format_email tool to produce a polite message. Return only the final formatted message.")
    await agent.session.assert_that(Expect.tools.sequence(["query_database", "format_email"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.called_with("query_database", {'query': 'SELECT email FROM users WHERE id=123', 'timeout_ms': 5000}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="query_database", expected_output={'email': 'user@example.com'}, field_path='email', match_type="exact", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.content.contains("Dear", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url_pattern': '(?i)(/users/123|\\bid=123\\b|/user/123|/api/users/123)', 'note': 'Fetched URL should reference user id 123 (case-insensitive regex).'}))
    await agent.session.assert_that(Expect.tools.was_called("format_email", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("format_email", {'email_pattern': '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', 'style': 'polite', 'note': 'format_email should be invoked with a valid-looking email and a request for a polite message.'}))
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "format_email"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', field_path=None, match_type="contains_regex", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="format_email", expected_output=['(?i)^(dear|hello|hi)\\b', '(?i)\\b(regards|sincerely|best regards|best)\\b'], field_path=None, match_type="all_contain_regex", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.content.contains("@", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("Score (0..1) averaged over criteria: 1) Correct tool use and order: fetch then format_email (0.20). 2) Retrieved a valid email for user id 123 (0.25). 3) Final output is the formatted polite message only (no extra tool traces or metadata) (0.30). 4) No self-reference or chain-of-thought present (0.15). 5) Concise and readable message (0.10). If overall score \u003c min_score, provide specific failures and examples.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_Solve_Math_Problem_with_Limited_Iterations(agent: TestAgent):
    response = await agent.generate_str("Calculate the sum of the first 50 positive integers and show the final numeric answer. Provide the numeric result only.")
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.content.contains("1275", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.judge.llm("Score from 0.0 to 1.0. Check that the final numeric answer is correct (1275) and that only the numeric result is returned (no explanations). 1.0 = exactly \u00271275\u0027 and nothing else; 0.0 = incorrect or additional explanation.", min_score=0.85), response=response)
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=0))
    await agent.session.assert_that(Expect.tools.sequence([], allow_other_calls=False))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(2000.0))
    await agent.session.assert_that(Expect.judge.llm("Score (0..1) averaged over criteria: 1) Correct numeric value: output equals \u00271275\u0027 (0.5). 2) Format: response contains only the numeric result and nothing else except optional surrounding whitespace/newline (0.3). 3) No chain-of-thought, no self-reference, and no tool traces (0.2). If score \u003c min_score, return specific failure reasons and examples of expected vs actual.", min_score=0.95), response=response)


