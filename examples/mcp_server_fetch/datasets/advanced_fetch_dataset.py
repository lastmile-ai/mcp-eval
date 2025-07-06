"""Advanced dataset for comprehensive fetch server testing."""

from mcp_eval import (
    Case,
    Dataset,
    ToolWasCalled,
    ToolSuccessRate,
    ResponseContains,
    LLMJudge,
    ToolSequence,
    MaxIterations,
)

# Advanced test cases
advanced_fetch_cases = [
    Case(
        name="multi_url_sequential_fetch",
        inputs="Fetch https://example.com, then https://httpbin.org/json, and compare their content types",
        metadata={"difficulty": "hard", "category": "multi_fetch"},
        evaluators=[
            ToolSequence(["fetch", "fetch"]),
            LLMJudge(
                rubric="Response should fetch both URLs and provide a comparison of their content types",
                min_score=0.8,
            ),
        ],
    ),
    Case(
        name="large_content_handling",
        inputs="Fetch https://httpbin.org/stream/50 and handle the large response appropriately",
        metadata={"difficulty": "hard", "category": "performance"},
        evaluators=[
            ToolWasCalled("fetch"),
            MaxIterations(max_iterations=5),
            LLMJudge(
                rubric="Response should handle streaming content appropriately",
                min_score=0.7,
            ),
        ],
    ),
    Case(
        name="error_recovery_with_fallback",
        inputs="Try to fetch https://nonexistent-12345.com, and if that fails, fetch https://example.com as fallback",
        metadata={"difficulty": "hard", "category": "error_recovery"},
        evaluators=[
            ToolWasCalled("fetch", min_times=2),
            ResponseContains("Example Domain"),
            LLMJudge(
                rubric="Response should attempt the invalid URL, recognize the error, and successfully fetch the fallback",
                min_score=0.85,
            ),
        ],
    ),
    Case(
        name="content_type_detection",
        inputs="Fetch https://httpbin.org/json, https://httpbin.org/html, and https://httpbin.org/xml, then identify each content type",
        metadata={"difficulty": "hard", "category": "content_analysis"},
        evaluators=[
            ToolWasCalled("fetch", min_times=3),
            ResponseContains("json", case_sensitive=False),
            ResponseContains("html", case_sensitive=False),
            ResponseContains("xml", case_sensitive=False),
            LLMJudge(
                rubric="Response should correctly identify all three content types",
                min_score=0.9,
            ),
        ],
    ),
    Case(
        name="markdown_conversion_quality",
        inputs="Fetch https://httpbin.org/html and convert it to well-formatted markdown",
        metadata={"difficulty": "medium", "category": "content_processing"},
        evaluators=[
            ToolWasCalled("fetch"),
            LLMJudge(
                rubric="Response should demonstrate proper HTML to markdown conversion with appropriate formatting",
                min_score=0.8,
                require_reasoning=True,
            ),
        ],
    ),
    Case(
        name="http_status_handling",
        inputs="Fetch these URLs and report their status: https://httpbin.org/status/200, https://httpbin.org/status/404, https://httpbin.org/status/500",
        metadata={"difficulty": "hard", "category": "status_handling"},
        evaluators=[
            ToolWasCalled("fetch", min_times=3),
            LLMJudge(
                rubric="Response should correctly identify and report different HTTP status codes",
                min_score=0.8,
            ),
        ],
    ),
]

# Create advanced dataset
advanced_fetch_dataset = Dataset(
    name="MCP Fetch Server Advanced Tests",
    cases=advanced_fetch_cases,
    server_name="fetch",
    agent_config={
        "name": "advanced_dataset_tester",
        "instruction": """You are an expert web content agent. 
        You excel at handling multiple URLs, error recovery, and content analysis.
        Always be thorough and provide detailed information about what you fetch.""",
        "llm_factory": "AnthropicAugmentedLLM",
        "model": "claude-3-sonnet-20240229",
        "max_iterations": 10,
    },
    evaluators=[
        # Global evaluators for all advanced cases
        ToolSuccessRate(min_rate=0.7, tool_name="fetch"),
    ],
)
