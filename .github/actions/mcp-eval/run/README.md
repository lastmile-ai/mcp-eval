# Run MCP-Eval tests with uv (run action)

This run action executes MCP-Eval tests with the uv toolchain, generates JSON/Markdown/HTML reports, uploads artifacts, and appends a Markdown summary to the job.

- Installs dependencies with `uv pip install` (editable project by default)
- Executes `mcp-eval run` against a provided tests path/selector
- Outputs report file paths and uploads artifacts (including `./test-reports` traces)
- Optional: use in a Pages deploy job to publish the HTML report

## Inputs

- python-version (default `3.11`): Python version
- working-directory (default `.`): Directory to install and run from
- tests (default `tests/`): Path(s) or selector for `mcp-eval run`
- run-args (default `-v`): Extra CLI args (e.g., `--max-concurrency 4`)
- reports-dir (default `mcpeval-reports`): Directory for reports
- json-report (default `mcpeval-results.json`)
- markdown-report (default `mcpeval-results.md`)
- html-report (default `mcpeval-results.html`)
- requirements (optional): If provided, action runs `uv pip install -r`
- install-args (default `-e .`): Args to `uv pip install` when `pyproject.toml` exists
- extra-packages (optional): Space-separated additional packages to install
- skip-install (default `false`)
- use-cache (default `true`)
- artifact-name (default `mcpeval-artifacts`)
- upload-artifacts (default `true`)
- set-summary (default `true`)
- pr-comment (default `false`): Post a sticky PR comment with the Markdown report
- sticky-comment-tag (default `mcpeval-report`): Tag used to identify/update the comment
- commit-report (default `false`): Commit Markdown/HTML reports back to the repo
- commit-path (default `docs/mcpeval`): Directory to write committed reports
- commit-branch (optional): Target branch (defaults to current)
- commit-message (default `chore(mcpeval): update CI report [skip ci]`)
- env-file (optional): A KEY=VALUE per line file to export to env
- extra-env (optional): Inline KEY=VALUE lines (newline-separated) to export

## Outputs

- results-json-path, results-md-path, results-html-path: Absolute paths
- reports-dir: Absolute path to reports directory
- test-reports-dir: Absolute path to the `./test-reports` directory produced by MCP-Eval

## Example

```yaml
name: MCP-Eval CI
on: [pull_request]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/mcp-eval/run
        with:
          python-version: '3.11'
          tests: tests/
          run-args: '-v --max-concurrency 4'
          reports-dir: mcpeval-reports
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

To publish the HTML report via Pages, add a separate job that downloads the artifact and sets `site/index.html` from the HTML report.

### Example: PR comment and commit reports

```yaml
jobs:
  tests:
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/mcp-eval/run
        with:
          pr-comment: 'true'
          commit-report: 'true'
          commit-path: 'docs/mcpeval'
          commit-message: 'chore: update mcpeval report [skip ci]'
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Example: Additional environment via env-file/extra-env

```yaml
      - uses: ./.github/actions/mcp-eval/run
        with:
          env-file: '.github/mcpeval.env'
          extra-env: |
            UV_NO_PROGRESS=1
            SOME_FLAG=true
```
