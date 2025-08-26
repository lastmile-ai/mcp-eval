import pytest
from mcp_eval import Expect

@pytest.mark.asyncio
async def test_basic_url_fetch(mcp_agent):
    response = await mcp_agent.generate_str("Please fetch the content from https://httpbin.org/html and tell me what you find.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"url": "https://httpbin.org/html"}))
    await mcp_agent.session.assert_that(Expect.content.contains("html", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003ctitle\u003eHerman Melville - Moby-Dick\u003c/title\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("Moby", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should acknowledge having internet access via the tool, successfully fetch the URL content, and describe what was found on the page (HTML content about Moby-Dick). The response should be informative and demonstrate successful URL fetching.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(mcp_agent):
    response = await mcp_agent.generate_str("Fetch https://example.com but limit the response to only 500 characters.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 500, "url": "https://example.com"}))
    await mcp_agent.session.assert_that(Expect.content.contains("500", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="Example Domain", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!doctype html\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("characters", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("Example", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with max_length parameter set to 500, successfully retrieve content from example.com, acknowledge the character limit was applied, and describe what was found within that limit. The assistant should demonstrate understanding of the length constraint.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_fetch_raw_html(mcp_agent):
    response = await mcp_agent.generate_str("I need the raw HTML source code from https://httpbin.org/html - don\u0027t convert it to markdown.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"raw": true, "url": "https://httpbin.org/html"}))
    await mcp_agent.session.assert_that(Expect.content.contains("\u003chtml", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chtml\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chead\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003cbody\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c/html\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("\u003c!DOCTYPE html\u003e", case_sensitive=True), response=response)
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with raw=true parameter, retrieve the actual HTML source code from httpbin.org/html without markdown conversion, and present the raw HTML tags and structure. The response should acknowledge that raw HTML was requested and delivered.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_fetch_with_start_index(mcp_agent):
    response = await mcp_agent.generate_str("Fetch https://httpbin.org/html starting from character position 1000.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"start_index": 1000, "url": "https://httpbin.org/html"}))
    await mcp_agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="not_contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chtml\u003e", field_path=null, match_type="not_contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("starting from", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("character", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with start_index=1000, successfully retrieve content from httpbin.org/html starting at character position 1000, acknowledge the start position parameter was used, and describe what content was found from that offset. The content should not include the beginning of the HTML document.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_invalid_url_handling(mcp_agent):
    response = await mcp_agent.generate_str("Please fetch content from this invalid URL: not-a-valid-url")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.judge.llm("The assistant should attempt to fetch the URL and handle any error gracefully, explaining to the user that the URL is invalid or cannot be fetched. The response should be helpful and informative about what went wrong.", min_score=0.7), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.content.contains("invalid", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("URL", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("not-a-valid-url", case_sensitive=True), response=response)
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should recognize that \u0027not-a-valid-url\u0027 is not a valid URL format and either: 1) refuse to call the fetch tool and explain the URL is invalid, or 2) attempt to call the tool but handle the error gracefully and explain the issue to the user. The response should be helpful and educational about proper URL formats.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_comprehensive_fetch_analysis(mcp_agent):
    response = await mcp_agent.generate_str("Fetch the first 2000 characters of https://www.wikipedia.org in raw HTML format, then summarize what type of content structure you observe.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 2000, "raw": true, "url": "https://www.wikipedia.org"}))
    await mcp_agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The assistant should successfully fetch the raw HTML content with the specified length limit and provide a meaningful analysis of the HTML structure, mentioning elements like DOCTYPE, head, body, or specific HTML tags observed.", min_score=0.8), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chtml", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chead\u003e", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="Wikipedia", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("2000 characters", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("structure", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("Wikipedia", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should correctly use the fetch tool with all specified parameters (URL=https://www.wikipedia.org, max_length=2000, raw=true), successfully retrieve the raw HTML content, and provide a meaningful analysis of the HTML structure observed. The analysis should mention HTML elements like DOCTYPE, head, meta tags, or other structural components typical of Wikipedia\u0027s homepage.", min_score=0.85), response=response)


