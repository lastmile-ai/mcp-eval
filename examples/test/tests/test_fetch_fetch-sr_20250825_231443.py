import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_basic_webpage_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch the content from https://example.com and tell me what it contains")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://example.com'}))
    await agent.session.assert_that(Expect.content.contains("example", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response describes the content of the webpage accurately", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("example.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("content", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Example Domain', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should acknowledge having internet access via the fetch tool and provide a clear summary of what the example.com website contains, including mentioning that it\u0027s a domain used for examples and documentation.", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(agent: TestAgent):
    response = await agent.generate_str("Fetch https://httpbin.org/html but only get the first 1000 characters")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 1000}))
    await agent.session.assert_that(Expect.judge.llm("Response acknowledges the length limitation and provides a summary of the truncated content", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("first", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Herman Melville - Moby Dick', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with max_length parameter set to 1000, acknowledge the length limitation, and provide a summary of the fetched content showing it was truncated to the specified character limit. The response should mention that only the first 1000 characters were retrieved as requested.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_raw_html_fetch(agent: TestAgent):
    response = await agent.generate_str("I need the raw HTML content from https://httpbin.org/html without any markdown conversion")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'raw': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.judge.llm("Response contains or references raw HTML tags and structure", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="starts_with", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='</html>', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<title>', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with raw=true parameter to get unprocessed HTML content. The response should acknowledge that raw HTML was requested and fetched, and should display or mention the actual HTML tags and structure without markdown formatting or simplification.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_pagination_with_start_index(agent: TestAgent):
    response = await agent.generate_str("Fetch https://httpbin.org/html starting from character position 500 to see more content")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'start_index': 500}))
    await agent.session.assert_that(Expect.judge.llm("Response acknowledges fetching content from a specific starting position", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("500", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("starting", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("character", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Call me Ishmael', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with start_index=500 parameter to retrieve content starting from character position 500. The response should acknowledge that content is being fetched from a specific starting position and should not contain the very beginning of the HTML document (like DOCTYPE or opening html tag) since it starts from position 500.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_json_api_fetch(agent: TestAgent):
    response = await agent.generate_str("Fetch the JSON data from https://httpbin.org/json and explain what information it contains")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/json'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slideshow', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response accurately describes the structure and content of the JSON data", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("information", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='{', field_path=None, match_type="starts_with", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='author', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='title', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("slideshow", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("contains", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly fetch JSON data from the specified URL and provide a clear explanation of what information the JSON contains. The response should identify key elements like slideshow data, author information, title, and other relevant JSON structure components. The explanation should be informative and demonstrate understanding of the JSON content.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_multiple_fetch_comparison(agent: TestAgent):
    response = await agent.generate_str("Compare the content from https://httpbin.org/html and https://httpbin.org/json by fetching both")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("comparison", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response demonstrates comparison between HTML and JSON content from both URLs", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html'}))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/json'}))
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("https://httpbin.org/json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("compare", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("both", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Herman Melville', field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slideshow', field_path=None, match_type="contains", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("difference", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should fetch both URLs and provide a meaningful comparison between the HTML content (which contains Moby Dick text) and the JSON content (which contains slideshow data). The comparison should highlight the different formats, content types, and purposes of each endpoint. The response should demonstrate understanding of both data structures.", min_score=0.8), response=response)


