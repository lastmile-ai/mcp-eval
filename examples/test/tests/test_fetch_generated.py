import pytest
from mcp_eval import Expect

@pytest.mark.asyncio
async def test_basic_webpage_fetch(mcp_agent):
    response = await mcp_agent.generate_str("Please fetch the content from https://httpbin.org/html and tell me what you find.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"url": "https://httpbin.org/html"}))
    await mcp_agent.session.assert_that(Expect.content.contains("Herman Melville", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.content.contains("httpbin.org", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003ctitle\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should clearly describe what content was found on the webpage, mentioning key elements like title, headings, or notable content structure. The assistant should acknowledge that it now has internet access.", min_score=7.0), response=response)

@pytest.mark.asyncio
async def test_fetch_with_length_limit(mcp_agent):
    response = await mcp_agent.generate_str("Fetch the first 500 characters from https://jsonplaceholder.typicode.com/posts and summarize what type of content it contains.")
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 500, "url": "https://jsonplaceholder.typicode.com/posts"}))
    await mcp_agent.session.assert_that(Expect.content.contains("JSON", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should acknowledge that the content was truncated to 500 characters and provide a reasonable summary of the JSON content structure.", min_score=7.0), response=response)
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="[", field_path=null, match_type="starts_with", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\"userId\"", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\"title\"", field_path=null, match_type="contains", case_sensitive=True, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("posts", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("500", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("characters", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should accurately summarize that the content contains JSON data with blog posts/articles, mentioning key fields like userId, title, body, etc. Should acknowledge the 500 character limit was applied.", min_score=7.0), response=response)

@pytest.mark.asyncio
async def test_raw_html_extraction(mcp_agent):
    response = await mcp_agent.generate_str("I need to see the raw HTML source code from https://httpbin.org/html. Please fetch it without any markdown conversion.")
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"raw": true, "url": "https://httpbin.org/html"}))
    await mcp_agent.session.assert_that(Expect.content.contains("\u003chtml\u003e", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("\u003c/html\u003e", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="starts_with", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chtml\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c/html\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003chead\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003cbody\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=-1))
    await mcp_agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("source code", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("\u003c!DOCTYPE", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(2))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should display the raw HTML source code with proper HTML tags visible (DOCTYPE, html, head, body tags). Should acknowledge that raw HTML was requested and fetched without markdown conversion.", min_score=8.0), response=response)

@pytest.mark.asyncio
async def test_pagination_with_start_index(mcp_agent):
    response = await mcp_agent.generate_str("First fetch 1000 characters from https://jsonplaceholder.typicode.com/posts, then fetch the next 1000 characters starting from where the first fetch ended.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await mcp_agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should demonstrate understanding of pagination by showing different content from the two fetches and explaining how start_index was used.", min_score=8.0), response=response)
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 1000, "start_index": 0, "url": "https://jsonplaceholder.typicode.com/posts"}))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 1000, "start_index": 1000, "url": "https://jsonplaceholder.typicode.com/posts"}))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="[", field_path=null, match_type="contains", case_sensitive=True, call_index=0))
    await mcp_agent.session.assert_that(Expect.content.contains("first", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("second", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("next", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(4))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should clearly indicate that two separate fetch operations were performed - first fetching 1000 characters from the beginning, then the next 1000 characters starting from index 1000. Should show understanding of pagination and mention the sequential nature of the fetches.", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_error_handling_invalid_url(mcp_agent):
    response = await mcp_agent.generate_str("Please try to fetch content from this invalid URL: https://this-domain-definitely-does-not-exist-12345.com/test")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=1))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"url": "https://this-domain-definitely-does-not-exist-12345.com/test"}))
    await mcp_agent.session.assert_that(Expect.content.contains("error", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should gracefully handle the error, explain what went wrong, and not crash or provide misleading information.", min_score=7.0), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("unable", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("fetch", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("this-domain-definitely-does-not-exist-12345.com", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(3))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should gracefully handle the error from the invalid URL. Should explain that the fetch failed due to the domain not existing or being unreachable. Should not claim success or show any webpage content. Should demonstrate proper error handling and communication.", min_score=7.0), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should be helpful and informative about why the fetch failed, mentioning concepts like DNS resolution failure, domain not found, or network error.", min_score=6.0), response=response)

@pytest.mark.asyncio
async def test_comprehensive_webpage_analysis(mcp_agent):
    response = await mcp_agent.generate_str("Analyze the GitHub API documentation at https://docs.github.com/en/rest. First get an overview with the first 2000 characters, then get the raw HTML of the same page to check for any JavaScript or dynamic elements.")
    await mcp_agent.session.assert_that(Expect.tools.was_called("fetch", min_times=2))
    await mcp_agent.session.assert_that(Expect.tools.sequence(["fetch", "fetch"], allow_other_calls=False))
    await mcp_agent.session.assert_that(Expect.content.contains("API", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should compare the markdown and raw HTML versions, identify key differences, and provide insights about the page structure and any dynamic elements.", min_score=8.0), response=response)
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"max_length": 2000, "raw": false, "url": "https://docs.github.com/en/rest"}))
    await mcp_agent.session.assert_that(Expect.tools.called_with("fetch", {"raw": true, "url": "https://docs.github.com/en/rest"}))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003c!DOCTYPE html\u003e", field_path=null, match_type="contains", case_sensitive=False, call_index=1))
    await mcp_agent.session.assert_that(Expect.tools.output_matches(tool_name="fetch", expected_output="\u003cscript", field_path=null, match_type="contains", case_sensitive=False, call_index=1))
    await mcp_agent.session.assert_that(Expect.content.contains("GitHub API", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("REST", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("2000", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("overview", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("raw HTML", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("JavaScript", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("dynamic", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.content.contains("analysis", case_sensitive=False), response=response)
    await mcp_agent.session.assert_that(Expect.performance.max_iterations(5))
    await mcp_agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should demonstrate a comprehensive analysis by first providing an overview from the simplified content, then examining the raw HTML for technical elements. Should mention findings about JavaScript, dynamic content, or other technical aspects discovered in the HTML source.", min_score=0.7), response=response)
    await mcp_agent.session.assert_that(Expect.judge.llm("The response should clearly differentiate between the two fetch operations - the first for content overview and the second for technical HTML analysis. Should provide insights about the GitHub documentation structure and any dynamic elements found.", min_score=0.6), response=response)


