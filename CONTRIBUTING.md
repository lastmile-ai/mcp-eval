# Contributing

We welcome **all** kinds of contributions - bug fixes, new features, documentation improvements, test additions, and more. _You don't need to be an AI expert or even a seasoned Python developer to help out._

## Checklist

Contributions are made through
[pull requests](https://help.github.com/articles/using-pull-requests/).

Before sending a pull request, make sure to do the following:

- Fork the repo, and create a feature branch prefixed with `feature/`
- [Lint, typecheck, and format](#code-quality) your code
- [Add or update tests](#testing) for your changes
- (Optional) [Add examples](#examples) if introducing new functionality

_Please reach out to the MCP-Eval maintainers before starting work on a large
contribution._ Get in touch at
[GitHub issues](https://github.com/lastmile-ai/mcp-eval/issues)
or [GitHub Discussions](https://github.com/lastmile-ai/mcp-eval/discussions).

## Prerequisites

To develop MCP-Eval, you'll need the following installed:

- Install [uv](https://docs.astral.sh/uv/), which we use for Python package management
- Install [Python](https://www.python.org/) >= 3.10. (You may already have it installed. To see your version, use `python -V` at the command line.)

  If you don't, install it using `uv python install 3.10`

- Install dev dependencies using:
  ```bash
  make sync
  ```
  This will sync all packages with extras and dev dependencies.

## Development Commands

We provide a [Makefile](./Makefile) with common development commands:

### Code Quality

**Note**: Lint and format are also run as part of any pre-commit hooks if configured.

**Format:**

```bash
make format
```

**Lint:**

This autofixes linter errors as well:

```bash
make lint
```

### Testing

**Run tests:**

```bash
make tests
```

**Run tests with coverage:**

```bash
make coverage
```

**Generate HTML coverage report:**

```bash
make coverage-report
```

### Documentation

**Sync subagents:**

```bash
make sync-subagents
```

**Verify subagents:**

```bash
make verify-subagents
```

## Scripts

There are several useful scripts in the `scripts/` directory that can be invoked via `uv run scripts/<script>.py [ARGS]`

### promptify.py

**Generates prompt.md file for LLMs**. Very helpful in leveraging LLMs to help develop MCP-Eval.

You can use the Makefile command for a quick generation with sensible defaults:

```bash
make prompt
```

Or run it directly with custom arguments:

```bash
uv run scripts/promptify.py -i "**/*.py" -x "**/tests/**"
```

Use `-i REGEX` to include only specific files, and `-x REGEX` to exclude certain files.

### format.py and lint.py

These scripts handle code formatting and linting respectively and are used by the Makefile commands.

## Testing

MCP-Eval is a testing framework itself, so maintaining high test quality is crucial. We use pytest for unit testing the framework components.

### Writing Tests

- Add unit tests in the `tests/` directory
- Follow existing test patterns and naming conventions
- Ensure tests are isolated and don't depend on external services unless necessary
- Use mocks and fixtures for external dependencies

### Test Organization

Tests are organized by module:
- `tests/test_core.py` - Core functionality tests
- `tests/test_assertions.py` - Assertion API tests
- `tests/test_metrics.py` - Metrics collection tests
- `tests/test_cli.py` - CLI command tests

## Examples

When adding new features or assertion types, please include example usage in the [`examples/`](./examples/) directory.

### Adding Examples

Each example should:
1. Have its own directory under `examples/`
2. Include a README explaining the purpose and usage
3. Contain runnable test files demonstrating the feature
4. Include any necessary configuration files

**Example structure:**
```
examples/
  your_feature/
    README.md
    test_feature.py
    mcpeval.yaml
    requirements.txt (if needed)
```

## Code Style

### Python Style Guide

- Follow PEP 8 with a line length of 100 characters
- Use type hints for all public functions and methods
- Add docstrings to all public modules, classes, and functions
- Keep functions focused and under 50 lines when possible

### Imports

- Group imports in the following order: standard library, third-party, local
- Use absolute imports for clarity
- Sort imports alphabetically within each group

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix with underscore `_private_method`

### Error Handling

- Use specific exception types rather than bare `except:` clauses
- Include helpful error messages that guide users toward solutions
- Log errors appropriately using the logging framework

## Documentation

### Docstrings

All public APIs should have comprehensive docstrings:

```python
def assert_that(evaluator: BaseEvaluator, response: str = None) -> AssertionResult:
    """Assert that an evaluator condition is met.
    
    Args:
        evaluator: The evaluator to run
        response: Optional response to evaluate
        
    Returns:
        AssertionResult containing pass/fail status and details
        
    Raises:
        AssertionError: If the assertion fails
    """
```

### README Updates

When adding significant features, update the main README.md with:
- Feature description in the appropriate section
- Usage examples
- Any new configuration options

## Commit Messages

Write clear, descriptive commit messages:
- First line: concise summary (50 chars or less)
- Blank line
- Detailed explanation if needed
- Reference issues with `Fixes #123` or `Relates to #456`

Example:
```
Add LLM judge multi-criteria evaluation

Implements multi-criteria evaluation support for LLM judges,
allowing weighted scoring across multiple dimensions like
accuracy, clarity, and completeness.

Fixes #789
```

## Pull Request Process

1. **Create a feature branch** from `main` with a descriptive name
2. **Make your changes** following the guidelines above
3. **Run tests and linting** locally: `make tests lint`
4. **Update documentation** as needed
5. **Submit the PR** with a clear description of changes
6. **Address review feedback** promptly
7. **Ensure CI passes** before merging

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why these changes are needed

## Changes
- List of specific changes
- Include any breaking changes

## Testing
How the changes were tested

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes
- [ ] Examples added (if applicable)
```

## Editor Settings

If you use VSCode, you might find the following `settings.json` useful:

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.rulers": [100]
  },
  "yaml.schemas": {
    "./schema/mcpeval.schema.json": [
      "mcpeval.yaml",
      "mcpeval.secrets.yaml"
    ]
  }
}
```

## Release Process

Releases are managed by maintainers following semantic versioning:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with release notes
3. Create a git tag: `git tag v1.2.3`
4. Push tag to trigger release workflow
5. Verify PyPI package publication

## Getting Help

If you need help or have questions:

- Check existing [GitHub Issues](https://github.com/lastmile-ai/mcp-eval/issues)
- Start a [GitHub Discussion](https://github.com/lastmile-ai/mcp-eval/discussions)
- Review the [documentation](https://mcp-eval.ai)

## Security

If you discover a security vulnerability:
- **Do not** open a public issue
- Email security concerns to the maintainers directly
- Include steps to reproduce if possible

## Thank You

If you are considering contributing, or have already done so, **thank you**. MCP-Eval aims to make MCP server testing reliable and straightforward, and we appreciate all contributions that help achieve this goal. Happy testing!