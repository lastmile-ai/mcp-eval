import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_basic_url_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch the content from https://httpbin.org/html and show me what you find.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html'}))
    await agent.session.assert_that(Expect.content.contains("fetched", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("httpbin.org", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("html", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<title>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch the URL and present the HTML content or a summary of what was found. The response should acknowledge having internet access and show the actual content from the httpbin.org/html endpoint.", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(agent: TestAgent):
    response = await agent.generate_str("Fetch https://example.com but limit the content to 1000 characters please.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://example.com', 'max_length': 1000}))
    await agent.session.assert_that(Expect.content.contains("example.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Example Domain', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!doctype html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch https://example.com with a 1000 character limit. The response should acknowledge the length limit was applied and show the truncated content from example.com. The assistant should mention that the content was limited to 1000 characters as requested.", min_score=8.0), response=response)

@pytest.mark.asyncio
async def test_fetch_raw_html(agent: TestAgent):
    response = await agent.generate_str("I need the raw HTML from https://httpbin.org/html without any markdown conversion.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'raw': True}))
    await agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="starts_with", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='</html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<head>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<body>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch the raw HTML from https://httpbin.org/html using the raw=true parameter. The response should show actual HTML tags and structure without any markdown conversion. The assistant should acknowledge that it\u0027s providing raw HTML as requested and display the unprocessed HTML content.", min_score=8.0), response=response)

@pytest.mark.asyncio
async def test_fetch_with_start_index(agent: TestAgent):
    response = await agent.generate_str("Fetch https://httpbin.org/html starting from character position 500.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'start_index': 500}))
    await agent.session.assert_that(Expect.content.contains("500", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("starting", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("character", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch https://httpbin.org/html starting from character position 500. The response should acknowledge that it\u0027s fetching from a specific starting position and show content that begins partway through the HTML document (not from the beginning with DOCTYPE or html tags). The assistant should explain that it\u0027s showing content starting from character 500.", min_score=8.0), response=response)

@pytest.mark.asyncio
async def test_multiple_url_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch both https://httpbin.org/json and https://httpbin.org/xml and compare their content types.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("compare", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/json'}))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/xml'}))
    await agent.session.assert_that(Expect.content.contains("json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("xml", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("content type", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='{', field_path=None, match_type="contains", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<?xml', field_path=None, match_type="contains", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slideshow', field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch both URLs and provide a meaningful comparison of their content types. It should identify that one returns JSON data and the other returns XML data, and explain the differences between these formats. The response should show actual content from both endpoints and discuss their structural differences.", min_score=8.0), response=response)

@pytest.mark.asyncio
async def test_internet_access_acknowledgment(agent: TestAgent):
    response = await agent.generate_str("Do you have internet access? Can you fetch web pages?")
    await agent.session.assert_that(Expect.content.contains("internet access", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("The response should clearly indicate that the agent now has internet access and can fetch web pages, contrary to previous limitations.", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("yes", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("fetch", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tool", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(1))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The assistant should clearly acknowledge that it now has internet access through the fetch tool. It should explain that while it originally did not have internet access, the fetch tool now enables it to retrieve web pages and content from the internet. The response should be confident and affirmative about its current capabilities.", min_score=8.0), response=response)




# ---- mcp-eval: additional generated tests ----

import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_basic_webpage_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch the Wikipedia homepage and tell me what you find there.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://en.wikipedia.org'}))
    await agent.session.assert_that(Expect.content.contains("Wikipedia", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response demonstrates that webpage content was successfully fetched and described", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://en.wikipedia.org/wiki/Main_Page'}))
    await agent.session.assert_that(Expect.content.contains("homepage", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Wikipedia', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Main Page", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should acknowledge that internet access is now available, successfully fetch the Wikipedia homepage, and provide a meaningful summary of what was found on the page including key sections or content areas.", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_raw_html_extraction(agent: TestAgent):
    response = await agent.generate_str("Get the raw HTML from https://httpbin.org/html - I need to see the actual HTML tags and structure.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'raw': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='</html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("\u003c!DOCTYPE", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("\u003chtml", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("\u003chead\u003e", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("\u003cbody\u003e", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully fetch and display the raw HTML content from httpbin.org/html, showing actual HTML tags like DOCTYPE, html, head, body elements rather than markdown-formatted content. The assistant should acknowledge using the raw parameter to get unprocessed HTML.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_length_limited_fetch(agent: TestAgent):
    response = await agent.generate_str("Fetch https://jsonplaceholder.typicode.com/posts but limit it to just the first 500 characters - I don\u0027t need the full response.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://jsonplaceholder.typicode.com/posts', 'max_length': 500}))
    await agent.session.assert_that(Expect.judge.llm("Response indicates content was truncated or limited, and mentions the character limit", min_score=0.6), response=response)
    await agent.session.assert_that(Expect.content.contains("500", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='[', field_path=None, match_type="starts_with", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='"userId"', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='"id"', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='"title"', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("500 characters", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("limited", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("posts", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully fetch the JSONPlaceholder posts endpoint with a 500-character limit, acknowledge the length restriction was applied, and show the truncated JSON data containing post objects with fields like userId, id, title. The assistant should explain that only the first 500 characters are shown as requested.", min_score=0.75), response=response)

@pytest.mark.asyncio
async def test_continuation_with_start_index(agent: TestAgent):
    response = await agent.generate_str("First fetch https://httpbin.org/html with max_length 100, then continue reading from where it left off to get more content.")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 100}))
    await agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await agent.session.assert_that(Expect.judge.llm("Response demonstrates understanding of pagination/continuation by making a second fetch call with start_index", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'start_index': 100}))
    await agent.session.assert_that(Expect.content.contains("first fetch", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("continue", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("100", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("more content", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should demonstrate a two-step process: first fetching the URL with a 100-character limit, then making a second fetch call using start_index=100 to continue from where the first fetch ended. The assistant should explain the continuation process and show content from both fetches, demonstrating understanding of the start_index parameter.", min_score=0.8), response=response)




# ---- mcp-eval: additional generated tests ----

import pytest
from mcp_eval import Expect
from mcp_eval.session import TestAgent

@pytest.mark.asyncio
async def test_basic_webpage_fetch(agent: TestAgent):
    response = await agent.generate_str("Please fetch the content from https://httpbin.org/html and tell me what it\u0027s about")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html'}))
    await agent.session.assert_that(Expect.content.contains("Herman Melville", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response demonstrates that the webpage was successfully fetched and the content was understood and described", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='Herman Melville - Moby Dick', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Moby Dick", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("webpage", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should accurately describe what the fetched webpage is about, mentioning that it contains content related to Herman Melville\u0027s Moby Dick novel. The response should be informative and demonstrate that the AI successfully used the fetch tool to access and understand the webpage content.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(agent: TestAgent):
    response = await agent.generate_str("Fetch https://httpbin.org/html but limit the response to 1000 characters")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'max_length': 1000}))
    await agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response acknowledges the length limitation and shows truncated content", min_score=0.6), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output=1000, field_path=None, match_type="max_length", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.content.contains("1000 characters", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("limited", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("Herman Melville", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should demonstrate that the AI correctly used the fetch tool with a max_length parameter of 1000 characters. The AI should acknowledge the length limitation and provide a summary of what was retrieved within that constraint. The response should mention that the content was truncated or limited to 1000 characters as requested.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fetch_raw_html_content(agent: TestAgent):
    response = await agent.generate_str("Get the raw HTML source code from https://httpbin.org/html without any markdown conversion")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/html', 'raw': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='<!DOCTYPE html>', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='</html>', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("\u003c!DOCTYPE html\u003e", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("\u003chtml", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("\u003c/html\u003e", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should demonstrate that the AI correctly used the fetch tool with raw=true parameter to retrieve the actual HTML source code. The response should contain HTML tags and structure, not markdown formatting. The AI should present the raw HTML content as requested and explain that it retrieved the unprocessed HTML source.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fetch_json_api_endpoint(agent: TestAgent):
    response = await agent.generate_str("Fetch data from https://httpbin.org/json and explain what information it contains")
    await agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fetch", {'url': 'https://httpbin.org/json'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='slideshow', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response correctly identifies and explains the JSON data structure and its contents", min_score=0.75), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='{', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output='author', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("slideshow", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("author", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("title", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("slides", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should demonstrate that the AI successfully fetched JSON data from the API endpoint and properly explained its structure and contents. The AI should identify that it\u0027s a slideshow object with properties like author, title, and slides array. The explanation should be clear and demonstrate understanding of the JSON structure.", min_score=0.8), response=response)


