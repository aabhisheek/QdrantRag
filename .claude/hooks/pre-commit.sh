#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-commit: ruff check ==="
uv run ruff check . --quiet

echo "=== Pre-commit: ruff format check ==="
uv run ruff format --check . --quiet

echo "=== Pre-commit: unit tests ==="
uv run pytest tests/unit/ -q --tb=short --no-header 2>/dev/null || {
    echo "FAIL: Unit tests failed. Fix before committing."
    exit 1
}

echo "All pre-commit checks passed."
