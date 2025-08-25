#!/bin/bash

# run `chmod +x ./scripts/setup-pre-commit.sh` first to make it executable

echo "Installing dependencies for development..."
uv sync --dev

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Running pre-commit on all files..."
uv run pre-commit run --all-files

echo "Setup completed!"