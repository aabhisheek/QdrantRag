Generate tests using the test-writer agent.
Use pytest + pytest-asyncio + httpx. Create fixtures, parametrize edge cases,
mock LLM responses. Place unit tests in tests/unit/,
integration tests in tests/integration/.
Run `uv run pytest` to verify all pass.
