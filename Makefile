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
	uv run coverage report -m --fail-under=80

.PHONY: coverage-report
coverage-report:
	uv run coverage run -m pytest
	uv run coverage html

.PHONY: prompt
prompt:
	rm -f prompt.md
	uv run scripts/promptify.py