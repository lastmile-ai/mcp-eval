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
            server_names=["playwright"],
        )
    )

@pytest.mark.asyncio
async def test_navigate_to_google(agent: TestAgent):
    response = await agent.generate_str("Navigate to https://www.google.com")
    await agent.session.assert_that(Expect.tools.was_called("browser_navigate", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_navigate", {'url': 'https://www.google.com'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_navigate", expected_output='200', field_path='status', match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_snapshot", expected_output='Google', field_path='title', match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("google.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully navigated to https://www.google.com and the Google homepage is displayed. The response should indicate successful completion of the navigation task.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_take_screenshot_of_page(agent: TestAgent):
    response = await agent.generate_str("Go to https://example.com and take a screenshot")
    await agent.session.assert_that(Expect.tools.was_called("browser_navigate", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("browser_take_screenshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_navigate", {'url': 'https://example.com'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_take_screenshot", expected_output='.png', field_path='filename', match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.sequence(["browser_navigate", "browser_take_screenshot"], allow_other_calls=True))
    await agent.session.assert_that(Expect.content.contains("screenshot", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("example.com", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_navigate", expected_output='200', field_path='status', match_type="exact", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully navigated to https://example.com and took a screenshot of the page. The response should indicate that both the navigation and screenshot capture were completed successfully.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_resize_browser_window(agent: TestAgent):
    response = await agent.generate_str("Resize the browser window to 1200x800 pixels")
    await agent.session.assert_that(Expect.tools.was_called("browser_resize", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_resize", {'width': 1200, 'height': 800}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_resize", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("1200", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("800", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("resize", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully resized the browser window to exactly 1200x800 pixels using the browser_resize tool. The response should confirm that the resize operation was completed successfully with the correct dimensions.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_press_escape_key(agent: TestAgent):
    response = await agent.generate_str("Press the Escape key")
    await agent.session.assert_that(Expect.tools.was_called("browser_press_key", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_press_key", {'key': 'Escape'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_press_key", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Escape", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("key", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully pressed the Escape key using the browser_press_key tool. The response should confirm that the Escape key was pressed without any errors or complications.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_get_page_snapshot(agent: TestAgent):
    response = await agent.generate_str("Take a snapshot of the current page")
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_snapshot", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_snapshot", expected_output='html', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_snapshot", expected_output='accessibility', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("snapshot", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("page", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully captured an accessibility snapshot of the current page using the browser_snapshot tool. The response should confirm that the snapshot was taken successfully and contains relevant page information in an accessible format.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_navigate_back(agent: TestAgent):
    response = await agent.generate_str("Go back to the previous page")
    await agent.session.assert_that(Expect.tools.was_called("browser_navigate_back", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_navigate_back", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_navigate_back", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("back", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("previous", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully navigated back to the previous page using the browser_navigate_back tool. The response should confirm that the back navigation was completed successfully and the browser returned to the previous page in the history.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_close_browser(agent: TestAgent):
    response = await agent.generate_str("Close the browser")
    await agent.session.assert_that(Expect.tools.was_called("browser_close", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_close", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_close", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("close", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("browser", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully closed the browser using the browser_close tool. The response should confirm that the browser was closed properly without any errors. This is typically a final action that terminates the browser session.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_get_console_messages(agent: TestAgent):
    response = await agent.generate_str("Check for any console messages on the page")
    await agent.session.assert_that(Expect.tools.was_called("browser_console_messages", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_console_messages", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_console_messages", expected_output='messages', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("console", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("message", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully retrieved and reported console messages from the current page using the browser_console_messages tool. The response should indicate what console messages were found (if any) or confirm that no messages were present. The agent should provide useful information about the console state of the page.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_get_network_requests(agent: TestAgent):
    response = await agent.generate_str("Show me all network requests made by this page")
    await agent.session.assert_that(Expect.tools.was_called("browser_network_requests", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_network_requests", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_network_requests", expected_output='requests', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_network_requests", expected_output='url', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("network", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("request", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully retrieved and reported all network requests made by the current page using the browser_network_requests tool. The response should show useful information about the network activity including URLs, request methods, status codes, and other relevant network data. If no requests were made, this should be clearly indicated.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_install_browser(agent: TestAgent):
    response = await agent.generate_str("Install the browser if it\u0027s not already installed")
    await agent.session.assert_that(Expect.tools.was_called("browser_install", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_install", {}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_install", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("install", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("browser", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(60000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully installed the browser using the browser_install tool. The response should confirm that the browser installation was completed successfully, or indicate that the browser was already installed. The agent should handle both scenarios appropriately - installing if needed or confirming existing installation.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_open_new_tab(agent: TestAgent):
    response = await agent.generate_str("Open a new browser tab")
    await agent.session.assert_that(Expect.tools.was_called("browser_tabs", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_tabs", {'action': 'new'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='tab', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='created', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("new tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully opened a new browser tab using the browser_tabs tool with the \u0027new\u0027 action. The response should confirm that a new tab was created and is now available for use. The agent should indicate successful completion of the tab creation operation.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_list_all_tabs(agent: TestAgent):
    response = await agent.generate_str("Show me all open browser tabs")
    await agent.session.assert_that(Expect.tools.was_called("browser_tabs", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_tabs", {'action': 'list'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='tabs', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='index', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("tabs", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("open", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully retrieved and displayed all open browser tabs using the browser_tabs tool with the \u0027list\u0027 action. The response should show useful information about each tab including tab indices, titles, URLs, or other identifying information. At minimum, the current tab should be listed.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_close_current_tab(agent: TestAgent):
    response = await agent.generate_str("Close the current tab")
    await agent.session.assert_that(Expect.tools.was_called("browser_tabs", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_tabs", {'action': 'close'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='closed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("close", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("current tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully closed the current browser tab using the browser_tabs tool with the \u0027close\u0027 action. The response should confirm that the current tab was closed successfully. This is typically a destructive action that closes the active tab.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_wait_for_3_seconds(agent: TestAgent):
    response = await agent.generate_str("Wait for 3 seconds")
    await agent.session.assert_that(Expect.tools.was_called("browser_wait_for", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_wait_for", {'time': 3}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_wait_for", expected_output='waited', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("3 seconds", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("wait", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("3", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully waited for exactly 3 seconds using the browser_wait_for tool with the time parameter set to 3. The response should confirm that the wait period completed successfully and the specified duration was observed.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_accept_alert_dialog(agent: TestAgent):
    response = await agent.generate_str("Accept any alert dialog that appears")
    await agent.session.assert_that(Expect.tools.was_called("browser_handle_dialog", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_handle_dialog", {'accept': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_handle_dialog", expected_output='accepted', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("accept", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("alert", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("dialog", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully accepted an alert dialog using the browser_handle_dialog tool with accept=true. The response should confirm that the alert dialog was found and accepted properly. The agent should handle the case where no dialog is present gracefully.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_dismiss_alert_dialog(agent: TestAgent):
    response = await agent.generate_str("Dismiss any alert dialog that appears")
    await agent.session.assert_that(Expect.tools.was_called("browser_handle_dialog", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_handle_dialog", {'accept': False}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_handle_dialog", expected_output='dismissed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("dismiss", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("alert", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("dialog", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully dismissed an alert dialog using the browser_handle_dialog tool with accept=false. The response should confirm that the alert dialog was found and dismissed (rejected/cancelled) properly. The agent should handle the case where no dialog is present gracefully.", min_score=0.8), response=response)

@pytest.mark.asyncio
async def test_take_fullpage_screenshot(agent: TestAgent):
    response = await agent.generate_str("Take a full page screenshot of the current page")
    await agent.session.assert_that(Expect.tools.was_called("browser_take_screenshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_take_screenshot", {'fullPage': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_take_screenshot", expected_output='.png', field_path='filename', match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_take_screenshot", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("full page", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("screenshot", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("page", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully took a full page screenshot using the browser_take_screenshot tool with fullPage=true. The response should confirm that the entire scrollable page was captured in the screenshot, not just the visible viewport. The screenshot file should be saved successfully.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_take_jpeg_screenshot(agent: TestAgent):
    response = await agent.generate_str("Take a screenshot in JPEG format")
    await agent.session.assert_that(Expect.tools.was_called("browser_take_screenshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_take_screenshot", {'type': 'jpeg'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_take_screenshot", expected_output='.jpeg', field_path='filename', match_type="contains", case_sensitive=True, call_index=-1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_take_screenshot", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("JPEG", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("jpeg", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("screenshot", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully took a screenshot in JPEG format using the browser_take_screenshot tool with type=\u0027jpeg\u0027. The response should confirm that the screenshot was captured in JPEG format and saved with the appropriate file extension (.jpeg). The agent should acknowledge the specific format request.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_press_arrow_keys(agent: TestAgent):
    response = await agent.generate_str("Press the down arrow key twice")
    await agent.session.assert_that(Expect.tools.was_called("browser_press_key", min_times=2))
    await agent.session.assert_that(Expect.tools.called_with("browser_press_key", {'key': 'ArrowDown'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_press_key", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=0))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_press_key", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=1))
    await agent.session.assert_that(Expect.content.contains("down arrow", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("twice", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("ArrowDown", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully pressed the down arrow key exactly twice using the browser_press_key tool with key=\u0027ArrowDown\u0027. The response should confirm that both key presses were executed successfully and acknowledge the specific requirement of pressing the key twice.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_press_enter_key(agent: TestAgent):
    response = await agent.generate_str("Press the Enter key")
    await agent.session.assert_that(Expect.tools.was_called("browser_press_key", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_press_key", {'key': 'Enter'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_press_key", expected_output='success', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Enter", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("key", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("press", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(3000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully pressed the Enter key using the browser_press_key tool with key=\u0027Enter\u0027. The response should confirm that the Enter key was pressed without any errors or complications. This is a simple but important keyboard interaction.", min_score=0.9), response=response)

@pytest.mark.asyncio
async def test_evaluate_page_title(agent: TestAgent):
    response = await agent.generate_str("Get the page title using JavaScript")
    await agent.session.assert_that(Expect.tools.was_called("browser_evaluate", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_evaluate", {'function': '() => { return document.title; }'}))
    await agent.session.assert_that(Expect.tools.called_with("browser_evaluate", {'function': '() => document.title'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_evaluate", expected_output='string', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("title", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("JavaScript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("document.title", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully used the browser_evaluate tool to execute JavaScript code that retrieves the page title using document.title. The response should show the actual title of the current page and confirm that the JavaScript evaluation was successful.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_evaluate_window_location(agent: TestAgent):
    response = await agent.generate_str("Get the current URL using JavaScript")
    await agent.session.assert_that(Expect.tools.was_called("browser_evaluate", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_evaluate", {'function': '() => { return window.location.href; }'}))
    await agent.session.assert_that(Expect.tools.called_with("browser_evaluate", {'function': '() => window.location.href'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_evaluate", expected_output='http', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("URL", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("location", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("JavaScript", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("window.location", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully used the browser_evaluate tool to execute JavaScript code that retrieves the current URL using window.location.href or similar location properties. The response should show the actual URL of the current page and confirm that the JavaScript evaluation was successful.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_wait_for_text_to_appear(agent: TestAgent):
    response = await agent.generate_str("Wait for the text \u0027Loading complete\u0027 to appear on the page")
    await agent.session.assert_that(Expect.tools.was_called("browser_wait_for", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_wait_for", {'text': 'Loading complete'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_wait_for", expected_output='found', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Loading complete", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("wait", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("appear", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully used the browser_wait_for tool to wait for the text \u0027Loading complete\u0027 to appear on the page. The response should confirm that the specified text was found and that the wait operation completed successfully without timing out.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_wait_for_text_to_disappear(agent: TestAgent):
    response = await agent.generate_str("Wait for the text \u0027Loading...\u0027 to disappear from the page")
    await agent.session.assert_that(Expect.tools.was_called("browser_wait_for", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_wait_for", {'textGone': 'Loading...'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_wait_for", expected_output='gone', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Loading...", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("disappear", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("wait", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("gone", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(30000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully used the browser_wait_for tool with the textGone parameter to wait for the text \u0027Loading...\u0027 to disappear from the page. The response should confirm that the specified text was no longer found on the page and that the wait operation completed successfully without timing out.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_select_tab_by_index(agent: TestAgent):
    response = await agent.generate_str("Switch to the second browser tab")
    await agent.session.assert_that(Expect.tools.was_called("browser_tabs", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_tabs", {'action': 'select', 'index': 1}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='selected', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("second tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("switch", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully switched to the second browser tab using the browser_tabs tool with action=\u0027select\u0027 and index=1. The response should confirm that the tab switch was completed successfully and that the browser is now focused on the second tab.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_close_tab_by_index(agent: TestAgent):
    response = await agent.generate_str("Close the third browser tab")
    await agent.session.assert_that(Expect.tools.was_called("browser_tabs", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_tabs", {'action': 'close', 'index': 2}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_tabs", expected_output='closed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("third tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("close", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("tab", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully closed the third browser tab using the browser_tabs tool with action=\u0027close\u0027 and index=2. The response should confirm that the specific tab was closed and is no longer available. This is a destructive operation that removes the tab from the browser.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_handle_prompt_dialog(agent: TestAgent):
    response = await agent.generate_str("Accept a prompt dialog with the text \u0027Hello World\u0027")
    await agent.session.assert_that(Expect.tools.was_called("browser_handle_dialog", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_handle_dialog", {'accept': True, 'promptText': 'Hello World'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_handle_dialog", expected_output='accepted', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Hello World", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("prompt", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("accept", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("dialog", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully accepted a prompt dialog using the browser_handle_dialog tool with accept=true and the specified promptText=\u0027Hello World\u0027. The response should confirm that the prompt dialog was found, the text was entered, and the dialog was accepted properly.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_upload_single_file(agent: TestAgent):
    response = await agent.generate_str("Upload the file at /path/to/document.pdf")
    await agent.session.assert_that(Expect.tools.was_called("browser_file_upload", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_file_upload", {'paths': ['/path/to/document.pdf']}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_file_upload", expected_output='uploaded', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("/path/to/document.pdf", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("upload", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("document.pdf", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("file", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully uploaded a single file using the browser_file_upload tool with the correct file path \u0027/path/to/document.pdf\u0027. The response should confirm that the file was located, selected, and uploaded successfully without any errors.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_upload_multiple_files(agent: TestAgent):
    response = await agent.generate_str("Upload files document.pdf and image.jpg from /uploads folder")
    await agent.session.assert_that(Expect.tools.was_called("browser_file_upload", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_file_upload", {'paths': ['/uploads/document.pdf', '/uploads/image.jpg']}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_file_upload", expected_output='uploaded', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("document.pdf", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("image.jpg", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("/uploads", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("upload", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("files", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully uploaded multiple files using the browser_file_upload tool with the correct paths \u0027/uploads/document.pdf\u0027 and \u0027/uploads/image.jpg\u0027. The response should confirm that both files were located in the uploads folder, selected, and uploaded successfully as a batch operation.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_click_element_with_ref(agent: TestAgent):
    response = await agent.generate_str("Click the login button")
    await agent.session.assert_that(Expect.tools.was_called("browser_click", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_click", {'element': 'login button', 'ref': 'button-123'}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_click", expected_output='clicked', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("login button", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("click", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("button", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_click"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully clicked the login button by first taking a snapshot to identify the element, then using browser_click with the appropriate element description and reference. The response should confirm that the login button was located and clicked successfully.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_double_click_element(agent: TestAgent):
    response = await agent.generate_str("Double-click the file icon")
    await agent.session.assert_that(Expect.tools.was_called("browser_click", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_click", {'element': 'file icon', 'ref': 'file-icon-456', 'doubleClick': True}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_click", {'doubleClick': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_click", expected_output='clicked', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("file icon", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("double", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("icon", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("click", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_click"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully double-clicked the file icon by first taking a snapshot to identify the element, then using browser_click with doubleClick=true and the appropriate element description and reference. The response should confirm that the file icon was located and double-clicked successfully.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_right_click_element(agent: TestAgent):
    response = await agent.generate_str("Right-click the context menu area")
    await agent.session.assert_that(Expect.tools.was_called("browser_click", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_click", {'element': 'context menu area', 'ref': 'menu-area-789', 'button': 'right'}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_click", {'button': 'right'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_click", expected_output='clicked', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("context menu", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("right-click", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("right", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("click", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_click"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully right-clicked the context menu area by first taking a snapshot to identify the element, then using browser_click with button=\u0027right\u0027 and the appropriate element description and reference. The response should confirm that the context menu area was located and right-clicked successfully, potentially triggering a context menu to appear.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_type_in_search_box(agent: TestAgent):
    response = await agent.generate_str("Type \u0027Hello World\u0027 in the search box")
    await agent.session.assert_that(Expect.tools.was_called("browser_type", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'element': 'search box', 'ref': 'search-input-001', 'text': 'Hello World'}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'text': 'Hello World'}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_type", expected_output='typed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("Hello World", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("search box", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("type", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_type"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully typed \u0027Hello World\u0027 in the search box by first taking a snapshot to identify the search box element, then using browser_type with the appropriate element description, reference, and text. The response should confirm that the search box was located and the text was entered successfully.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_type_and_submit(agent: TestAgent):
    response = await agent.generate_str("Type \u0027search query\u0027 in the input field and press Enter")
    await agent.session.assert_that(Expect.tools.was_called("browser_type", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'element': 'input field', 'ref': 'input-002', 'text': 'search query', 'submit': True}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'text': 'search query', 'submit': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_type", expected_output='typed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("search query", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("input field", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("Enter", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("submit", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("type", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_type"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully typed \u0027search query\u0027 in the input field and submitted it by pressing Enter. This should be accomplished using browser_type with submit=true, which types the text and automatically presses Enter. The response should confirm that both the text entry and submission were completed successfully.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_type_slowly(agent: TestAgent):
    response = await agent.generate_str("Type \u0027password123\u0027 slowly in the password field")
    await agent.session.assert_that(Expect.tools.was_called("browser_type", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'element': 'password field', 'ref': 'password-003', 'text': 'password123', 'slowly': True}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_type", {'text': 'password123', 'slowly': True}))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_type", expected_output='typed', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("password123", case_sensitive=True), response=response)
    await agent.session.assert_that(Expect.content.contains("password field", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("slowly", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("type", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_type"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully typed \u0027password123\u0027 slowly in the password field by first taking a snapshot to identify the password field element, then using browser_type with slowly=true and the appropriate element description, reference, and text. The response should confirm that the password field was located and the text was entered character by character (slowly) as requested.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_hover_over_menu(agent: TestAgent):
    response = await agent.generate_str("Hover over the navigation menu")
    await agent.session.assert_that(Expect.tools.was_called("browser_hover", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_hover", {'element': 'navigation menu', 'ref': 'nav-menu-004'}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_hover", expected_output='hovered', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("navigation menu", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("hover", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("menu", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("navigation", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_hover"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(4))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully hovered over the navigation menu by first taking a snapshot to identify the navigation menu element, then using browser_hover with the appropriate element description and reference. The response should confirm that the navigation menu was located and hovered over successfully, potentially revealing dropdown menu items or other hover effects.", min_score=0.85), response=response)

@pytest.mark.asyncio
async def test_drag_and_drop(agent: TestAgent):
    response = await agent.generate_str("Drag the file from the source folder to the destination folder")
    await agent.session.assert_that(Expect.tools.was_called("browser_drag", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("browser_drag", {'startElement': 'source folder file', 'startRef': 'file-source-005', 'endElement': 'destination folder', 'endRef': 'folder-dest-006'}))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.tools.output_matches(tool_name="browser_drag", expected_output='dragged', field_path=None, match_type="contains", case_sensitive=False, call_index=-1))
    await agent.session.assert_that(Expect.content.contains("drag", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("file", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("source folder", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("destination folder", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("drop", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["browser_snapshot", "browser_drag"], allow_other_calls=True))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.judge.llm("The agent successfully dragged a file from the source folder to the destination folder using the browser_drag tool. This should involve taking a snapshot to identify both the source file element and destination folder element, then using browser_drag with appropriate startElement, startRef, endElement, and endRef parameters. The response should confirm that the file was successfully moved from source to destination.", min_score=0.85), response=response)


@pytest.mark.asyncio
async def test_comprehensive_page_audit(agent: TestAgent):
    response = await agent.generate_str("Perform a complete audit of the page including console errors, network requests, and accessibility")
    await agent.session.assert_that(Expect.tools.was_called("browser_console_messages", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("browser_network_requests", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("browser_snapshot", min_times=1))
    await agent.session.assert_that(Expect.performance.max_iterations(10))
    await agent.session.assert_that(Expect.judge.llm("Comprehensive page audit covering multiple aspects", min_score=0.8), response=response)


