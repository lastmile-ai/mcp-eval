# MCP-Eval Subagents

These subagents are specialized AI assistants that help with MCP-Eval development and test generation. They can be loaded by both Claude Code and the MCP-Eval CLI generator.

## Available Subagents

### For Writing Tests

- **`mcp-eval-test-writer.md`** - Expert at writing comprehensive MCP-Eval tests in all styles (decorator, pytest, dataset)
- **`mcp-eval-debugger.md`** - Expert at debugging test failures and configuration issues
- **`mcp-eval-config-expert.md`** - Expert at configuring MCP-Eval projects and mcpeval.yaml files

### For Test Generation (Used by CLI)

- **`test-scenario-designer.md`** - Creates diverse, high-quality test scenarios
- **`test-assertion-refiner.md`** - Enhances test assertions for robustness
- **`test-code-emitter.md`** - Converts scenarios into valid Python test code

## Usage in Claude Code

These subagents are automatically loaded when you use Claude Code with MCP-Eval. Claude will proactively use them when appropriate.

To explicitly invoke a subagent:
```
> Use the mcp-eval-test-writer subagent to create tests for my fetch server
```

## Usage in MCP-Eval Generator

The test generation subagents are automatically used by the CLI generator:

```bash
# They work behind the scenes during generation
mcp-eval generate --style pytest --n-examples 10

# The generator uses:
# 1. test-scenario-designer - to create scenarios
# 2. test-assertion-refiner - to enhance assertions  
# 3. test-code-emitter - to produce Python code
```

## Usage with MCP-Agent

These subagents follow the Claude-style markdown format and can be loaded by mcp-agent:

```python
from mcp_agent.workflows.factory import load_agent_specs_from_dir

# Load all subagents
specs = load_agent_specs_from_dir("src/mcp_eval/data/subagents")

# Use in routing, orchestration, etc.
router = await create_router_llm(
    agents=specs,
    provider="anthropic",
    context=context
)
```

## Configuration

### Option 1: Reference from Package (Recommended)

Add this to your `mcpeval.yaml` to use the subagents directly from the package:

```yaml
agents:
  enabled: true
  search_paths:
    # Try to load from installed package first
    # The actual path depends on where Python packages are installed
    # You can find it with: python -c "import mcp_eval, os; print(os.path.join(os.path.dirname(mcp_eval.__file__), 'data', 'subagents'))"
    - "path/to/site-packages/mcp_eval/data/subagents"
    # Standard locations
    - ".claude/agents"
    - "~/.claude/agents"
    - ".mcp-agent/agents"
    - "~/.mcp-agent/agents"
  pattern: "*.md"
```

### Option 2: Copy to Project

Copy the subagents to your project:
```bash
# For Claude Code
cp -r src/mcp_eval/data/subagents/*.md .claude/agents/

# For mcp-agent
cp -r src/mcp_eval/data/subagents/*.md .mcp-agent/agents/
```

### Option 3: Development Mode

If running from source, reference the source directory:
```yaml
agents:
  enabled: true
  search_paths:
    - "./src/mcp_eval/data/subagents"
  pattern: "*.md"
```

## Format

Each subagent follows the Claude markdown format:

```markdown
---
name: subagent-name
description: When to use this subagent
tools: optional, comma-separated list
---

System prompt describing the subagent's expertise and behavior
```

## Contributing

When adding new subagents:
1. Follow the existing naming convention
2. Provide clear descriptions of when to use them
3. Include detailed expertise and examples in the prompt
4. Test with both Claude Code and the CLI generator