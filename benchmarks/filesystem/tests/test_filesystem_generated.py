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
            server_names=["filesystem"],
        )
    )

@pytest.mark.asyncio
async def test_basic_file_read(agent: TestAgent):
    response = await agent.generate_str("Read the contents of config.txt and tell me what it contains")
    await agent.session.assert_that(Expect.tools.was_called("read_text_file", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("read_text_file", {'path': 'config.txt'}))
    await agent.session.assert_that(Expect.judge.llm("Response accurately describes the file contents", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("config.txt", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("contains", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="read_text_file", expected_output='.*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should clearly communicate what the config.txt file contains, presenting the file contents in a readable format and acknowledging that it has successfully read the requested file.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_file_head_tail_operations(agent: TestAgent):
    response = await agent.generate_str("Show me the first 5 lines of logs.txt and the last 3 lines of errors.log")
    await agent.session.assert_that(Expect.tools.was_called("read_text_file", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("read_text_file", {'path': 'logs.txt', 'head': 5}))
    await agent.session.assert_that(Expect.tools.called_with("read_text_file", {'path': 'errors.log', 'tail': 3}))
    await agent.session.assert_that(Expect.content.contains("first 5 lines", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("last 3 lines", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("logs.txt", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("errors.log", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["read_text_file", "read_text_file"], allow_other_calls=False))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should clearly present the first 5 lines from logs.txt and the last 3 lines from errors.log, properly labeling each section and demonstrating correct use of head/tail parameters.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_create_and_write_project_structure(agent: TestAgent):
    response = await agent.generate_str("Create a new project directory called \u0027my_app\u0027 and create a main.py file inside it with a simple Hello World program")
    await agent.session.assert_that(Expect.tools.was_called("create_directory", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("write_file", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("create_directory", {'path': 'my_app'}))
    await agent.session.assert_that(Expect.tools.called_with("write_file", {'path': 'my_app/main.py', 'content': "print('Hello, World!')"}))
    await agent.session.assert_that(Expect.content.contains("created", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.called_with("write_file", {'path': 'my_app/main.py'}))
    await agent.session.assert_that(Expect.content.contains("my_app", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("main.py", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("Hello World", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["create_directory", "write_file"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="write_file", expected_output='print\\(.*[Hh]ello.*[Ww]orld.*\\)', field_path='content', match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should successfully create a directory called \u0027my_app\u0027 and create a main.py file inside it with valid Python code that prints \u0027Hello World\u0027 or similar greeting. The code should be syntactically correct Python.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_search_and_analyze_files(agent: TestAgent):
    response = await agent.generate_str("Search for all Python files in the project directory and then read the contents of any files that contain \u0027import numpy\u0027")
    await agent.session.assert_that(Expect.tools.was_called("search_files", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("search_files", {'path': '.', 'pattern': '*.py'}))
    await agent.session.assert_that(Expect.judge.llm("Response shows systematic search and analysis of Python files", min_score=0.7), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(10))
    await agent.session.assert_that(Expect.tools.called_with("search_files", {'pattern': '.py'}))
    await agent.session.assert_that(Expect.content.contains("Python files", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains(".py", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("import numpy", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="search_files", expected_output='.*\\.py.*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(8))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first search for Python files (.py extension), then examine the found files to identify which ones contain \u0027import numpy\u0027, and finally display the contents of any files that match this criteria. The workflow should be logical and complete.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_batch_file_comparison(agent: TestAgent):
    response = await agent.generate_str("Compare the contents of config_dev.json, config_staging.json, and config_prod.json to identify differences")
    await agent.session.assert_that(Expect.tools.was_called("read_multiple_files", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("read_multiple_files", {'paths': ['config_dev.json', 'config_staging.json', 'config_prod.json']}))
    await agent.session.assert_that(Expect.content.contains("differences", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides meaningful comparison of configuration files", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("config_dev.json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("config_staging.json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("config_prod.json", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("difference", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("compar", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="read_multiple_files", expected_output='.*config_dev\\.json.*config_staging\\.json.*config_prod\\.json.*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should efficiently read all three configuration files (dev, staging, prod) and provide a clear comparison identifying key differences between them. The analysis should highlight variations in configuration values, missing keys, or structural differences between the environments.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_file_editing_with_preview(agent: TestAgent):
    response = await agent.generate_str("In the file app.py, replace the line \u0027debug = False\u0027 with \u0027debug = True\u0027 but show me a preview first before making the change")
    await agent.session.assert_that(Expect.tools.was_called("edit_file", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("edit_file", {'path': 'app.py', 'edits': [{'oldText': 'debug = False', 'newText': 'debug = True'}], 'dryRun': True}))
    await agent.session.assert_that(Expect.tools.called_with("edit_file", {'path': 'app.py', 'edits': [{'oldText': 'debug = False', 'newText': 'debug = True'}], 'dryRun': False}))
    await agent.session.assert_that(Expect.content.contains("preview", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.called_with("edit_file", {'path': 'app.py', 'dryRun': True}))
    await agent.session.assert_that(Expect.tools.called_with("edit_file", {'path': 'app.py', 'edits': [{'oldText': 'debug = False', 'newText': 'debug = True'}]}))
    await agent.session.assert_that(Expect.content.contains("app.py", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("debug = False", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("debug = True", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["edit_file", "edit_file"], allow_other_calls=False))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="edit_file", expected_output='.*debug = False.*debug = True.*', field_path=None, match_type="regex", case_sensitive=True, call_index=0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.performance.response_time_under(12000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should first show a preview of the changes using dryRun mode, displaying a diff that shows \u0027debug = False\u0027 being replaced with \u0027debug = True\u0027, then proceed to make the actual change. The workflow should be clear about showing preview first, then applying the change.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_directory_analysis_with_sizes(agent: TestAgent):
    response = await agent.generate_str("Show me the directory structure of the current folder and identify the largest files, sorted by size")
    await agent.session.assert_that(Expect.tools.was_called("list_directory_with_sizes", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("list_directory_with_sizes", {'path': '.', 'sortBy': 'size'}))
    await agent.session.assert_that(Expect.content.contains("largest", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response shows directory contents sorted by size and identifies largest files", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.was_called("directory_tree", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("directory_tree", {'path': '.'}))
    await agent.session.assert_that(Expect.content.contains("directory structure", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("largest files", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("size", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="directory_tree", expected_output='.*"type".*"name".*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="list_directory_with_sizes", expected_output='.*\\[FILE\\].*bytes.*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.sequence(["directory_tree", "list_directory_with_sizes"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should show both the hierarchical directory structure and a size-sorted list of files. It should clearly present the directory tree structure and identify the largest files with their sizes, properly sorted from largest to smallest.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_media_file_handling(agent: TestAgent):
    response = await agent.generate_str("Read the image file logo.png and tell me its format and approximate file size")
    await agent.session.assert_that(Expect.tools.was_called("read_media_file", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("read_media_file", {'path': 'logo.png'}))
    await agent.session.assert_that(Expect.tools.was_called("get_file_info", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("get_file_info", {'path': 'logo.png'}))
    await agent.session.assert_that(Expect.content.contains("PNG", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("size", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("logo.png", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("format", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="read_media_file", expected_output='.*image/png.*', field_path='mime_type', match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="get_file_info", expected_output='.*\\d+.*bytes.*', field_path=None, match_type="regex", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.sequence(["read_media_file", "get_file_info"], allow_other_calls=False))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly identify the file format as PNG from the MIME type, and provide the file size information in a readable format (bytes, KB, etc.). It should use the appropriate media file reading tool for the image file.", min_score=0.8), response=response)


