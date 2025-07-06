import asyncio
from mcp_eval import (
    Case,
    Dataset,
    ToolWasCalled,
    ResponseContains,
    LLMJudge,
    test_session,
)
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

# Define test cases
cases = [
    Case(
        name="fetch_example",
        inputs="Fetch https://example.com",
        expected_output="Example Domain",
        evaluators=[
            ToolWasCalled("fetch"),
            ResponseContains("Example Domain"),
        ],
    ),
    Case(
        name="fetch_and_summarize",
        inputs="Fetch https://example.com and summarize in one sentence",
        evaluators=[
            ToolWasCalled("fetch"),
            LLMJudge("Response contains a one-sentence summary"),
        ],
    ),
]

# Create dataset (uses same unified TestSession)
dataset = Dataset(
    name="Fetch Server Tests",
    cases=cases,
    server_name="fetch",
    agent_config={
        "llm_factory": AnthropicAugmentedLLM,
    },
)


async def fetch_task(inputs: str) -> str:
    """System under test using unified session."""
    async with test_session("fetch", "dataset_task") as agent:
        return await agent.generate_str(inputs)


async def main():
    # Run evaluation using unified session architecture
    report = await dataset.evaluate(fetch_task)
    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    asyncio.run(main())
