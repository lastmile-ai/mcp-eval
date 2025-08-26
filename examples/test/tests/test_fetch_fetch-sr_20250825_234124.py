import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_basic_webpage_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch the content from https://example.com and tell me what it contains")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://example.com'}))
    await agent.session.assert_that(Expect.content.contains("Example Domain", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response accurately describes the content of example.com webpage", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("example.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("content", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!doctype html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<title>Example Domain</title>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response accurately describe the contents of example.com, mentioning it\u0027s a simple example page used for documentation purposes?", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(agent: TestAgent):
    response = await agent.generate_str("Fetch https://httpbin.org/html but only get the first 1000 characters")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 1000}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Herman Melville', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.judge.llm("Response acknowledges the length limitation and shows partial content", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("1000 characters", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response indicate that the fetch was limited to 1000 characters and provide a summary of the HTML content retrieved from httpbin.org/html?", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response mention or acknowledge the character limit constraint (1000 characters) that was requested?", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_raw_html_fetch(agent: TestAgent):
    response = await agent.generate_str("I need to see the raw HTML source code of https://httpbin.org/html")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'raw': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='</html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<head>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<body>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response provide the actual raw HTML source code from httpbin.org/html, including proper HTML tags and structure?", min_score=0.85), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response acknowledge that raw HTML was requested and delivered (as opposed to processed/simplified content)?", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_paginated_fetch_with_start_index(agent: TestAgent):
    response = await agent.generate_str("First fetch https://httpbin.org/html with a limit of 500 characters, then get the next 500 characters starting from where we left off")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 500}))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'start_index': 500, 'max_length': 500}))
    await agent.session.assert_that(Expect.judge.llm("Response shows content from both fetch operations and explains the pagination approach", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 500, 'start_index': 500}))
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("first", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("500 characters", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("next", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(12000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response demonstrate fetching content in two parts: first 500 characters, then the next 500 characters starting from index 500?", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response show understanding of pagination by using start_index parameter correctly for the second fetch?", min_score=0.85), response=response)
    await agent.session.assert_that(Expect.content.contains("starting from", case_sensitive=False), response=response)

@pytest.mark.asyncio
async def test_json_api_fetch(agent: TestAgent):
    response = await agent.generate_str("Fetch data from https://httpbin.org/json and explain what information it contains")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/json'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slideshow', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response correctly identifies and explains the JSON data structure", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("httpbin.org/json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("information", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='{', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='}', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slides', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("contains", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response fetch JSON data from httpbin.org/json and provide a clear explanation of what information the JSON contains (such as slideshow data with title, author, and slides)?", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response demonstrate understanding of JSON structure by mentioning specific fields or data types found in the response?", min_score=0.75), response=response)
    await agent.session.assert_that(Expect.content.contains("data", case_sensitive=False), response=response)

@pytest.mark.asyncio
async def test_multiple_url_comparison(agent: TestAgent):
    response = await agent.generate_str("Compare the content of https://httpbin.org/html and https://example.com - what are the main differences?")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html'}))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://example.com'}))
    await agent.session.assert_that(Expect.content.contains("Herman Melville", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("Example Domain", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides a meaningful comparison between the two websites highlighting key differences", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("compare", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("differences", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("httpbin.org", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("example.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!doctype html>', field_path=None, match_type="contains", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Example Domain', field_path=None, match_type="contains", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("Does the response successfully fetch content from both URLs and provide a meaningful comparison highlighting the main differences between httpbin.org/html and example.com?", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response identify specific differences such as content complexity, purpose, styling, or structure between the two websites?", min_score=0.75), response=response)
    await agent.session.assert_that(Expect.content.contains("main differences", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Does the response demonstrate that both websites were successfully accessed and analyzed for comparison?", min_score=0.7), response=response)


