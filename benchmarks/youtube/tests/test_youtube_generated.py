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
            server_names=["youtube"],
        )
    )

@pytest.mark.asyncio
async def test_basic_transcript_extraction(agent: TestAgent):
    response = await agent.generate_str("Get the transcript from this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}))
    await agent.session.assert_that(Expect.judge.llm("Response provides transcript content from the video", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract and present the transcript content from the YouTube video. It should be readable and properly formatted, containing the actual spoken content from the video.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_transcript_with_specific_format(agent: TestAgent):
    response = await agent.generate_str("Get the transcript from https://www.youtube.com/watch?v=abc123 in text-only format without timestamps")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=abc123', 'format_output': 'text_only', 'include_timestamps': False}))
    await agent.session.assert_that(Expect.judge.llm("Response shows transcript was retrieved in text-only format", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=abc123', 'include_timestamps': False, 'format_output': 'text_only'}))
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should contain a clean text-only transcript without any timestamps, time markers, or formatting brackets. The text should flow naturally as plain spoken content without timing information.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_multilingual_transcript_spanish(agent: TestAgent):
    response = await agent.generate_str("Extract the Spanish transcript from this video: https://www.youtube.com/watch?v=xyz789")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=xyz789', 'lang': 'es'}))
    await agent.session.assert_that(Expect.judge.llm("Response indicates Spanish transcript was requested", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=xyz789'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("spanish", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully check for available languages first, then extract the Spanish transcript. The content should be in Spanish and properly formatted. The response should confirm that Spanish transcripts were retrieved.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_check_available_languages(agent: TestAgent):
    response = await agent.generate_str("What languages are available for transcripts on this YouTube video? https://www.youtube.com/watch?v=multilang123")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=multilang123'}))
    await agent.session.assert_that(Expect.judge.llm("Response lists available transcript languages", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("available", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should list the available transcript languages for the video without extracting actual transcript content. It should clearly present language options (like \u0027en\u0027, \u0027es\u0027, \u0027fr\u0027, etc.) and not include the full transcript text.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_search_specific_term(agent: TestAgent):
    response = await agent.generate_str("Search for the word \u0027algorithm\u0027 in the transcript of https://www.youtube.com/watch?v=tech456")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=tech456', 'search_term': 'algorithm'}))
    await agent.session.assert_that(Expect.judge.llm("Response shows search results for \u0027algorithm\u0027 in the transcript", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("algorithm", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for the term \u0027algorithm\u0027 in the video transcript and return relevant matches with surrounding context. It should show where in the transcript the term appears and provide contextual information around each occurrence.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_search_with_context(agent: TestAgent):
    response = await agent.generate_str("Find mentions of \u0027machine learning\u0027 in this video transcript with 3 lines of context: https://www.youtube.com/watch?v=ai789")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=ai789', 'search_term': 'machine learning', 'context_lines': 3}))
    await agent.session.assert_that(Expect.judge.llm("Response shows search results with contextual lines around matches", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("machine learning", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("3", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for \u0027machine learning\u0027 in the transcript and return results with exactly 3 lines of context before and after each match. The context should provide meaningful surrounding text to understand how the term is being used in the video.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_get_video_summary(agent: TestAgent):
    response = await agent.generate_str("Summarize this YouTube video for me: https://www.youtube.com/watch?v=summary123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=summary123'}))
    await agent.session.assert_that(Expect.judge.llm("Response provides a summary of the video content", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("video", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a concise summary of the video content, not the full transcript. The summary should capture key points and main themes from the video in a condensed format that\u0027s easy to understand and significantly shorter than a full transcript.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_short_summary_request(agent: TestAgent):
    response = await agent.generate_str("Give me a brief 200-character summary of this video: https://www.youtube.com/watch?v=brief456")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=brief456', 'max_length': 200}))
    await agent.session.assert_that(Expect.judge.llm("Response provides a concise summary within length constraints", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("200", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a very brief summary that respects the 200-character limit. The summary should be concise, capture the essence of the video content, and be significantly shorter than a standard summary. It should not exceed the requested character count.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_video_id_only_input(agent: TestAgent):
    response = await agent.generate_str("Extract transcript from video ID: dQw4w9WgXcQ")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'dQw4w9WgXcQ'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles video ID input correctly", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("dQw4w9WgXcQ", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript using just the video ID without requiring a full YouTube URL. It should handle the video ID format correctly and return transcript content, demonstrating that the tool can work with both full URLs and standalone video IDs.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_korean_transcript_request(agent: TestAgent):
    response = await agent.generate_str("Get the Korean transcript from https://www.youtube.com/watch?v=korean123 with detailed formatting")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=korean123', 'lang': 'ko', 'format_output': 'detailed'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Korean transcript in detailed format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=korean123'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("korean", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("detailed", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully check for Korean language availability first, then extract the Korean transcript in detailed format with proper formatting and timestamps. The content should demonstrate Korean language characters and detailed formatting structure.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_timed_text_format(agent: TestAgent):
    response = await agent.generate_str("I need the transcript from https://www.youtube.com/watch?v=timing456 in timed text format")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=timing456', 'format_output': 'timed_text'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests transcript in timed text format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=timing456', 'format_output': 'timed_text', 'include_timestamps': True}))
    await agent.session.assert_that(Expect.content.contains("timed text", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("00:", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("--\u003e", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide the transcript in timed text format with proper timing markers and timestamps. The format should include time codes (like 00:00:15 --\u003e 00:00:18) followed by the corresponding text, similar to subtitle or SRT format.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_french_language_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027intelligence artificielle\u0027 in the French transcript of https://www.youtube.com/watch?v=french789")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=french789', 'search_term': 'intelligence artificielle', 'lang': 'fr'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for French term in French transcript", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=french789'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "search_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("intelligence artificielle", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("french", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for French language availability, then successfully search for the French term \u0027intelligence artificielle\u0027 in the French transcript. It should return relevant matches with context from the French-language content.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_multiple_tools_workflow(agent: TestAgent):
    response = await agent.generate_str("First check what languages are available for https://www.youtube.com/watch?v=multi123, then get the English transcript")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.judge.llm("Response first checks available languages then retrieves English transcript", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=multi123'}))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=multi123', 'lang': 'en'}))
    await agent.session.assert_that(Expect.content.contains("languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("available", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("english", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(40000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first show the available languages for the video, then proceed to extract the English transcript. It should clearly demonstrate a two-step workflow where language availability is checked first, followed by transcript extraction in English.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_search_and_summarize(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027climate change\u0027 in https://www.youtube.com/watch?v=climate456 and also give me a summary of the whole video")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=climate456', 'search_term': 'climate change'}))
    await agent.session.assert_that(Expect.judge.llm("Response provides both search results and video summary", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=climate456'}))
    await agent.session.assert_that(Expect.content.contains("climate change", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide both a search for \u0027climate change\u0027 showing specific matches with context, and a separate comprehensive summary of the entire video. Both pieces of information should be clearly presented and distinct from each other.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_extended_context_search(agent: TestAgent):
    response = await agent.generate_str("Find all mentions of \u0027quantum\u0027 in https://www.youtube.com/watch?v=quantum789 with 5 lines of context around each match")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=quantum789', 'search_term': 'quantum', 'context_lines': 5}))
    await agent.session.assert_that(Expect.judge.llm("Response shows search results with extended context lines", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.content.contains("quantum", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("5 lines", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("mentions", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should find all occurrences of the word \u0027quantum\u0027 in the transcript and provide exactly 5 lines of context before and after each match. The extended context should give comprehensive surrounding information for each mention of the term.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_no_timestamps_request(agent: TestAgent):
    response = await agent.generate_str("Get the transcript from https://www.youtube.com/watch?v=notime123 but don\u0027t include any timestamps")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=notime123', 'include_timestamps': False}))
    await agent.session.assert_that(Expect.judge.llm("Response requests transcript without timestamps", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a clean transcript without any timestamps, time markers, or timing information. The text should flow naturally without any time-related formatting or brackets indicating when content was spoken.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_comprehensive_video_analysis(agent: TestAgent):
    response = await agent.generate_str("Analyze this video completely: https://www.youtube.com/watch?v=analyze456 - check available languages, get the full transcript, search for \u0027important\u0027, and provide a summary")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.judge.llm("Response performs comprehensive analysis using all available tools", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=analyze456'}))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=analyze456'}))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=analyze456', 'search_term': 'important'}))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=analyze456'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript", "search_transcript", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("important", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(60000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a comprehensive analysis using all four tools in sequence: (1) list available languages, (2) provide the full transcript, (3) show search results for \u0027important\u0027 with context, and (4) give an overall summary. All four components should be clearly presented and organized.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_case_sensitive_search(agent: TestAgent):
    response = await agent.generate_str("Search for exactly \u0027AI\u0027 (case sensitive) in https://www.youtube.com/watch?v=ai123")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=ai123', 'search_term': 'AI'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for the exact term \u0027AI\u0027", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("AI", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("case sensitive", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search specifically for \u0027AI\u0027 in uppercase letters only, not matching lowercase \u0027ai\u0027 or mixed case variants. The search results should demonstrate case-sensitive matching and only show instances where \u0027AI\u0027 appears exactly as specified.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_long_summary_request(agent: TestAgent):
    response = await agent.generate_str("Give me a detailed 1000-character summary of this educational video: https://www.youtube.com/watch?v=education789")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=education789', 'max_length': 1000}))
    await agent.session.assert_that(Expect.judge.llm("Response requests detailed summary with specified length", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("1000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("detailed", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("educational", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a detailed summary that approaches 1000 characters in length, significantly longer than the default 500-character summary. The summary should be comprehensive and detailed, covering the main educational content of the video thoroughly.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_youtube_shorts_transcript(agent: TestAgent):
    response = await agent.generate_str("Get transcript from this YouTube Short: https://www.youtube.com/shorts/short123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/shorts/short123'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles YouTube Shorts URL correctly", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("short", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("shorts", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from a YouTube Shorts URL, demonstrating that the tool can handle both regular YouTube videos and YouTube Shorts format. The transcript should be brief, reflecting the short-form nature of the content.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_phrase_search(agent: TestAgent):
    response = await agent.generate_str("Search for the exact phrase \u0027machine learning algorithms\u0027 in https://www.youtube.com/watch?v=mlalgo456")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=mlalgo456', 'search_term': 'machine learning algorithms'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for the complete phrase", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("machine learning algorithms", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("exact phrase", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for the complete phrase \u0027machine learning algorithms\u0027 as a whole unit, not just individual words. The results should show only instances where this exact phrase appears together in the transcript, with appropriate context around each match.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_german_transcript_summary(agent: TestAgent):
    response = await agent.generate_str("Get a summary of the German transcript from https://www.youtube.com/watch?v=deutsch123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=deutsch123', 'lang': 'de'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests German transcript summary", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=deutsch123'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("german", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for German language availability, then successfully generate a summary from the German transcript. The summary should be based on German content and demonstrate that the tool can work with non-English language transcripts.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_minimal_context_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027python\u0027 in https://www.youtube.com/watch?v=python456 with just 1 line of context")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=python456', 'search_term': 'python', 'context_lines': 1}))
    await agent.session.assert_that(Expect.judge.llm("Response searches with minimal context lines", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("python", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("1 line", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for \u0027python\u0027 and provide minimal context with exactly 1 line before and after each match, rather than the default 2 lines. The results should be more concise with limited surrounding text.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_playlist_video_transcript(agent: TestAgent):
    response = await agent.generate_str("Extract transcript from this video in a playlist: https://www.youtube.com/watch?v=playlist123\u0026list=PLexample")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=playlist123&list=PLexample'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles playlist URL correctly", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("playlist", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from the specific video in the playlist URL, handling the playlist parameter correctly and focusing on the individual video (playlist123) rather than trying to process the entire playlist. The transcript should be for the single specified video.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_japanese_language_check(agent: TestAgent):
    response = await agent.generate_str("Check if Japanese transcripts are available for https://www.youtube.com/watch?v=japan789")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=japan789'}))
    await agent.session.assert_that(Expect.judge.llm("Response checks available languages for Japanese content", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("japanese", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("available", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should specifically check for and report on the availability of Japanese language transcripts without extracting the actual transcript content. It should clearly indicate whether Japanese (ja) is among the available language options for this video.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_technical_term_search(agent: TestAgent):
    response = await agent.generate_str("Find all occurrences of \u0027blockchain\u0027 in this tech video: https://www.youtube.com/watch?v=blockchain123")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=blockchain123', 'search_term': 'blockchain'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for technical term \u0027blockchain\u0027", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("blockchain", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("occurrences", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tech", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("all", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should comprehensively find and present all instances where \u0027blockchain\u0027 appears in the tech video transcript. It should show multiple occurrences (if they exist) with context, demonstrating a thorough search of the technical content.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_transcript_with_timestamps(agent: TestAgent):
    response = await agent.generate_str("Get the full transcript from https://www.youtube.com/watch?v=timestamps456 with all timing information included")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=timestamps456', 'include_timestamps': True}))
    await agent.session.assert_that(Expect.judge.llm("Response requests transcript with timestamps included", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("timestamps", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("timing", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("00:", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide the complete transcript with full timing information included. Timestamps should be clearly visible throughout the transcript, showing when each segment of speech occurs in the video.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_italian_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027intelligenza artificiale\u0027 in the Italian transcript of https://www.youtube.com/watch?v=italiano123")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=italiano123', 'search_term': 'intelligenza artificiale', 'lang': 'it'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches Italian transcript for Italian term", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=italiano123'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "search_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("intelligenza artificiale", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("italian", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Italian language availability, then successfully search for the Italian phrase \u0027intelligenza artificiale\u0027 in the Italian transcript. It should return relevant matches with context from the Italian-language content.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_very_brief_summary(agent: TestAgent):
    response = await agent.generate_str("Give me just a 100-character summary of https://www.youtube.com/watch?v=brief789")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=brief789', 'max_length': 100}))
    await agent.session.assert_that(Expect.judge.llm("Response requests very brief summary within character limit", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("100", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("brief", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(18000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide an extremely brief summary that stays within the 100-character limit. The summary should be concise and capture only the most essential information from the video in a very condensed format.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_portuguese_transcript(agent: TestAgent):
    response = await agent.generate_str("Extract the Portuguese transcript from https://www.youtube.com/watch?v=portugal456 in detailed format")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=portugal456', 'lang': 'pt', 'format_output': 'detailed'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Portuguese transcript in detailed format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=portugal456'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("portuguese", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("detailed", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Portuguese language availability, then successfully extract the Portuguese transcript in detailed format with proper formatting and structure. The content should demonstrate Portuguese language characters and detailed formatting.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_number_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u00272024\u0027 in the transcript of https://www.youtube.com/watch?v=year2024")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=year2024', 'search_term': '2024'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for numeric term \u00272024\u0027", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("2024", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for the exact number \u00272024\u0027 in the transcript and return relevant matches with context. It should show where this specific year is mentioned in the video content without matching other similar years.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_educational_video_workflow(agent: TestAgent):
    response = await agent.generate_str("For this educational video https://www.youtube.com/watch?v=edu456: check languages, get English transcript without timestamps, search for \u0027learning\u0027, and create a 300-char summary")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=edu456', 'lang': 'en', 'include_timestamps': False}))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=edu456', 'max_length': 300}))
    await agent.session.assert_that(Expect.judge.llm("Response executes complete educational video analysis workflow", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=edu456'}))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=edu456', 'search_term': 'learning'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript", "search_transcript", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("english", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("learning", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("300", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(70000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should execute a complete 4-step workflow for the educational video: (1) check available languages, (2) extract English transcript without timestamps, (3) search for \u0027learning\u0027 with context, and (4) create a 300-character summary. All components should be clearly presented in order.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_embedded_video_transcript(agent: TestAgent):
    response = await agent.generate_str("Get transcript from this embedded video URL: https://www.youtube.com/embed/embed123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/embed/embed123'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles embedded video URL format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("embedded", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from the embedded YouTube URL format, demonstrating that the tool can handle embed URLs in addition to standard watch URLs. The transcript should be extracted properly regardless of the URL format.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_chinese_traditional_transcript(agent: TestAgent):
    response = await agent.generate_str("Extract Traditional Chinese transcript from https://www.youtube.com/watch?v=taiwan123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=taiwan123', 'lang': 'zh-TW'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Traditional Chinese transcript", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=taiwan123'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("traditional chinese", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Traditional Chinese language availability, then successfully extract the Traditional Chinese transcript. The content should demonstrate Traditional Chinese characters and distinguish from Simplified Chinese if both are available.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_zero_context_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027innovation\u0027 in https://www.youtube.com/watch?v=innovation456 with no context lines")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=innovation456', 'search_term': 'innovation', 'context_lines': 0}))
    await agent.session.assert_that(Expect.judge.llm("Response searches with zero context lines", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("innovation", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("no context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for \u0027innovation\u0027 and return only the exact matches without any surrounding context lines. The results should show just the specific instances where the term appears, without any additional text before or after each match.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_long_video_id(agent: TestAgent):
    response = await agent.generate_str("Get transcript from video ID: ABCDEFGHIJKLMNOPQRSTuvwxyz123456")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'ABCDEFGHIJKLMNOPQRSTuvwxyz123456'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles long video ID format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("ABCDEFGHIJKLMNOPQRSTuvwxyz123456", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully handle the longer video ID format and extract the transcript. It should demonstrate that the tool can work with video IDs of varying lengths and formats, not just the standard 11-character YouTube video IDs.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_russian_language_support(agent: TestAgent):
    response = await agent.generate_str("Get the Russian transcript summary from https://www.youtube.com/watch?v=russia789")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=russia789', 'lang': 'ru'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Russian transcript summary", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=russia789'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("russian", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Russian language availability, then successfully generate a summary from the Russian transcript. The summary should be based on Russian content and demonstrate that the tool can work with Cyrillic characters and Russian language transcripts.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_special_characters_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027C++\u0027 in this programming video: https://www.youtube.com/watch?v=cpp123")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=cpp123', 'search_term': 'C++'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for term with special characters", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("C++", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("programming", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for the exact term \u0027C++\u0027 including the special characters (plus signs) in the programming video transcript. It should handle special characters correctly and return relevant matches with context around programming content.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_maximum_summary_length(agent: TestAgent):
    response = await agent.generate_str("Give me the longest possible summary (2000 characters) of https://www.youtube.com/watch?v=longsum456")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=longsum456', 'max_length': 2000}))
    await agent.session.assert_that(Expect.judge.llm("Response requests maximum length summary", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("2000", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("longest", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide a comprehensive, extended summary that approaches the maximum 2000-character limit. The summary should be significantly more detailed than standard summaries, covering multiple aspects and themes from the video content thoroughly.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_hindi_transcript_check(agent: TestAgent):
    response = await agent.generate_str("Check if Hindi transcripts are available for https://www.youtube.com/watch?v=hindi123 and get one if available")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=hindi123'}))
    await agent.session.assert_that(Expect.judge.llm("Response checks for Hindi language availability and may request transcript", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=0))
    await agent.session.assert_that(Expect.content.contains("hindi", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("available", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("check", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Hindi language availability. If Hindi is available, it should then extract the Hindi transcript. If Hindi is not available, it should clearly indicate that Hindi transcripts are not available for this video. The response should be conditional based on availability.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_acronym_search(agent: TestAgent):
    response = await agent.generate_str("Find \u0027NASA\u0027 mentions in https://www.youtube.com/watch?v=space456")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=space456', 'search_term': 'NASA'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for acronym \u0027NASA\u0027", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("NASA", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("mentions", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("space", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully find mentions of the acronym \u0027NASA\u0027 in all caps in the space-related video transcript. It should show where NASA is mentioned with appropriate context, demonstrating accurate acronym matching.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_mobile_youtube_url(agent: TestAgent):
    response = await agent.generate_str("Extract transcript from this mobile YouTube link: https://m.youtube.com/watch?v=mobile123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://m.youtube.com/watch?v=mobile123'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles mobile YouTube URL format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("mobile", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from the mobile YouTube URL format (m.youtube.com), demonstrating that the tool can handle different YouTube URL variations including mobile links. The transcript should be extracted properly regardless of the URL format.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_dutch_language_transcript(agent: TestAgent):
    response = await agent.generate_str("Get Dutch transcript in text-only format from https://www.youtube.com/watch?v=netherlands456")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=netherlands456', 'lang': 'nl', 'format_output': 'text_only'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Dutch transcript in text-only format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=netherlands456'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("dutch", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("text-only", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Dutch language availability, then successfully extract the Dutch transcript in text-only format without timestamps or detailed formatting. The content should be clean Dutch text without timing information.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_contextual_search_phrase(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027data science\u0027 in https://www.youtube.com/watch?v=datascience789 with 4 lines of surrounding context")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=datascience789', 'search_term': 'data science', 'context_lines': 4}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for phrase with specified context lines", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("data science", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("4 lines", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("surrounding", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for the phrase \u0027data science\u0027 and provide exactly 4 lines of context before and after each match. The results should show comprehensive surrounding information with more context than the default 2 lines, helping understand how the term is used in the video.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_youtu_be_short_url(agent: TestAgent):
    response = await agent.generate_str("Get transcript from short URL: https://youtu.be/shorturl123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://youtu.be/shorturl123'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles youtu.be short URL format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("short URL", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from the youtu.be short URL format, demonstrating that the tool can handle different YouTube URL variations including the shortened domain. The transcript should be extracted properly regardless of the URL format.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_arabic_transcript_request(agent: TestAgent):
    response = await agent.generate_str("Extract Arabic transcript from https://www.youtube.com/watch?v=arabic456 with timestamps")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=arabic456', 'lang': 'ar', 'include_timestamps': True}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Arabic transcript with timestamps", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=arabic456'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("arabic", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("timestamps", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("00:", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Arabic language availability, then successfully extract the Arabic transcript with timestamps included. The content should demonstrate Arabic characters and include timing information throughout the transcript.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_quoted_search_term(agent: TestAgent):
    response = await agent.generate_str("Search for the exact phrase \u0027artificial intelligence\u0027 in quotes in https://www.youtube.com/watch?v=aiquotes123")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=aiquotes123', 'search_term': 'artificial intelligence'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for quoted phrase", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("artificial intelligence", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("exact phrase", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("quotes", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for the complete phrase \u0027artificial intelligence\u0027 as a single unit, not individual words. The results should show only instances where this exact two-word phrase appears together in the transcript, with appropriate context around each occurrence.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_swedish_language_summary(agent: TestAgent):
    response = await agent.generate_str("Get a Swedish summary of https://www.youtube.com/watch?v=sweden789 limited to 400 characters")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=sweden789', 'lang': 'sv', 'max_length': 400}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Swedish summary with character limit", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=sweden789'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("swedish", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("400", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Swedish language availability, then successfully generate a Swedish summary limited to 400 characters. The summary should be based on Swedish content and demonstrate that the tool can work with Swedish language transcripts within the specified character limit.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_hyphenated_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u0027state-of-the-art\u0027 in https://www.youtube.com/watch?v=sota456")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=sota456', 'search_term': 'state-of-the-art'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for hyphenated term", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("state-of-the-art", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("hyphenated", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for the exact hyphenated term \u0027state-of-the-art\u0027 including all hyphens, demonstrating that the tool can handle hyphenated phrases correctly. It should not match variations without hyphens and should show relevant matches with context.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_timestamp_parameter_url(agent: TestAgent):
    response = await agent.generate_str("Get transcript from video with timestamp: https://www.youtube.com/watch?v=timestamp123\u0026t=120s")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=timestamp123&t=120s'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles URL with timestamp parameter", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("timestamp", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the full transcript from the YouTube URL with timestamp parameter, demonstrating that the tool can handle URLs with additional parameters like timestamp markers. The transcript should be complete regardless of the timestamp parameter.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_polish_transcript_timed(agent: TestAgent):
    response = await agent.generate_str("Get Polish transcript in timed text format from https://www.youtube.com/watch?v=poland123")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=poland123', 'lang': 'pl', 'format_output': 'timed_text'}))
    await agent.session.assert_that(Expect.judge.llm("Response requests Polish transcript in timed text format", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=poland123'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("polish", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("timed text", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("00:", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("--\u003e", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Polish language availability, then successfully extract the Polish transcript in timed text format with proper time markers and timestamps. The format should include Polish content with timing codes similar to subtitle format.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_multi_word_technical_search(agent: TestAgent):
    response = await agent.generate_str("Find \u0027deep learning neural networks\u0027 in https://www.youtube.com/watch?v=deeplearn456 with extensive context")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=deeplearn456', 'search_term': 'deep learning neural networks', 'context_lines': 6}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for multi-word technical term with extended context", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=deeplearn456', 'search_term': 'deep learning neural networks', 'context_lines': 5}))
    await agent.session.assert_that(Expect.content.contains("deep learning neural networks", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("extensive context", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("technical", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("5 lines", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should search for the complete four-word technical phrase \u0027deep learning neural networks\u0027 as a single unit and provide extensive context (5 lines) around each match. The results should show comprehensive surrounding information to understand the technical discussion context.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_video_with_playlist_and_index(agent: TestAgent):
    response = await agent.generate_str("Extract transcript from: https://www.youtube.com/watch?v=playidx123\u0026list=PLtest\u0026index=5")
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=playidx123&list=PLtest&index=5'}))
    await agent.session.assert_that(Expect.judge.llm("Response handles complex URL with playlist and index parameters", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("transcript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("playlist", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(25000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully extract the transcript from the YouTube URL with both playlist and index parameters, demonstrating that the tool can handle complex URLs with multiple query parameters. The transcript should be for the specific video regardless of playlist context.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_turkish_language_check_and_extract(agent: TestAgent):
    response = await agent.generate_str("Check if Turkish is available for https://www.youtube.com/watch?v=turkey789, then extract Turkish transcript if possible")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=turkey789'}))
    await agent.session.assert_that(Expect.judge.llm("Response checks Turkish language availability and may extract transcript", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=0))
    await agent.session.assert_that(Expect.content.contains("turkish", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("available", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("check", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(35000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first check for Turkish language availability. If Turkish is available, it should then conditionally extract the Turkish transcript. If Turkish is not available, it should clearly indicate that Turkish transcripts are not available for this video. The response should be conditional based on availability.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_percentage_search(agent: TestAgent):
    response = await agent.generate_str("Search for \u002795%\u0027 in https://www.youtube.com/watch?v=percentage456")
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=percentage456', 'search_term': '95%'}))
    await agent.session.assert_that(Expect.judge.llm("Response searches for percentage symbol term", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("95%", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("percentage", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("search", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="type", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(22000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully search for the exact percentage \u002795%\u0027 including the percent symbol, demonstrating that the tool can handle special characters like % in search terms. It should return relevant matches with context around statistical or percentage-related content.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_comprehensive_multilingual_analysis(agent: TestAgent):
    response = await agent.generate_str("For https://www.youtube.com/watch?v=multilang789: check all available languages, get English detailed transcript, search for \u0027global\u0027 in English, and provide a 600-char English summary")
    await agent.session.assert_that(Expect.tools.was_called("get_available_languages", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("search_transcript", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("get_transcript_summary", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript", {'url': 'https://www.youtube.com/watch?v=multilang789', 'lang': 'en', 'format_output': 'detailed'}))
    await agent.session.assert_that(Expect.tools.called_with("search_transcript", {'url': 'https://www.youtube.com/watch?v=multilang789', 'search_term': 'global', 'lang': 'en'}))
    await agent.session.assert_that(Expect.tools.called_with("get_transcript_summary", {'url': 'https://www.youtube.com/watch?v=multilang789', 'lang': 'en', 'max_length': 600}))
    await agent.session.assert_that(Expect.judge.llm("Response performs comprehensive multilingual video analysis with all tools", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("get_available_languages", {'url': 'https://www.youtube.com/watch?v=multilang789'}))
    await agent.session.assert_that(Expect.tools.sequence(["get_available_languages", "get_transcript", "search_transcript", "get_transcript_summary"], allow_other_calls=False))
    await agent.session.assert_that(Expect.content.contains("available languages", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("english", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("detailed", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("global", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("summary", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("600", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_available_languages", expected_output='array', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript", expected_output='string', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_transcript", expected_output='object', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_transcript_summary", expected_output='string', field_path=None, match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(80000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should execute a comprehensive 4-step multilingual analysis: (1) list all available languages, (2) extract detailed English transcript, (3) search for \u0027global\u0027 in English with context, and (4) create a 600-character English summary. All components should be clearly presented in the specified order with English language focus.", min_score=0.85), response=response)


