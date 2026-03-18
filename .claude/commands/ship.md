Pre-ship quality gate. Run in order:
1) `uv run ruff check .`
2) `uv run ruff format --check .`
3) `uv run pytest --tb=short -q`
4) `uv run pip-audit`
5) Check for .env or secrets in staged files
6) Check for debug print/logging statements
7) Verify all tests pass.
Report pass/fail for each gate.
