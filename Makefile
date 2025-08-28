.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

# Linter and Formatter
.PHONY: format
format: 
	uv run scripts/format.py

.PHONY: lint
lint: 
	uv run scripts/lint.py --fix

# Tests
.PHONY: tests
tests: 
	uv run pytest 

.PHONY: coverage
coverage:
	uv run coverage run -m pytest
	uv run coverage xml -o coverage.xml
	uv run coverage report -m --fail-under=50

.PHONY: coverage-report
coverage-report:
	uv run coverage run -m pytest
	uv run coverage html

.PHONY: schema
schema:
	uv run scripts/gen_schema.py

.PHONY: prompt
prompt:
	rm -f prompt.md
	uv run scripts/promptify.py

# Documentation
.PHONY: sync-subagents
sync-subagents:
	uv run python scripts/sync_subagents.py --sync

.PHONY: verify-subagents
verify-subagents:
	uv run python scripts/sync_subagents.py