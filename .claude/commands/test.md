Run tests using pytest. Determine narrowest scope from context.
Commands: `uv run pytest tests/unit/ -q --tb=short` for unit,
`uv run pytest tests/integration/ -q --tb=short` for integration,
`uv run pytest --cov=src --cov-report=term-missing` for coverage.
If a specific file is mentioned, run only that file.
