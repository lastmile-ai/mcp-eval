import mcp_eval
from mcp_eval import task, setup, parametrize, ToolWasCalled, ResponseContains
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM


@setup
def configure_tests():
    mcp_eval.use_server("fetch")
    mcp_eval.use_agent(
        {
            "name": "fetch_tester",
            "instruction": "You can fetch web content. Complete tasks as requested.",
            "llm_factory": AnthropicAugmentedLLM,
        }
    )


@task("Test basic URL fetching")
async def test_fetch_basic(agent):
    """Test basic fetch functionality with modern evaluators."""
    response = await agent.generate_str("Fetch https://example.com")

    # Use modern evaluator system
    agent.evaluate_now(ToolWasCalled("fetch"), response, "fetch_tool_used")
    agent.evaluate_now(ResponseContains("Example Domain"), response, "contains_title")


@task("Test fetch with multiple URLs")
@parametrize("url", ["https://example.com", "https://github.com", "https://python.org"])
async def test_fetch_urls(agent, url):
    response = await agent.generate_str(f"Fetch {url}")

    # Modern evaluators with explicit session context
    agent.add_deferred_evaluator(ToolWasCalled("fetch"), "fetch_called")
    agent.evaluate_now(ResponseContains("content"), response, "has_content")
