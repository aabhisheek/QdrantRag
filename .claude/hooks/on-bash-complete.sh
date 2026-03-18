#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${BASH_TOOL_OUTPUT:-}"
EXIT_CODE="${BASH_TOOL_EXIT_CODE:-0}"

if [ "$EXIT_CODE" -eq 0 ]; then
    exit 0
fi

if echo "$OUTPUT" | grep -q "ModuleNotFoundError"; then
    MODULE=$(echo "$OUTPUT" | grep "ModuleNotFoundError" | sed "s/.*No module named '\([^']*\)'.*/\1/" | head -1)
    echo "HINT: Module not found. Try: uv add $MODULE"
fi

if echo "$OUTPUT" | grep -q "No module named 'langchain'"; then
    echo "HINT: LangChain not installed. Try: uv add langchain"
fi

if echo "$OUTPUT" | grep -q "qdrant_client.http.exceptions"; then
    echo "HINT: Qdrant connection error. Check Qdrant is running: docker-compose up -d qdrant"
fi

if echo "$OUTPUT" | grep -q "ruff"; then
    echo "HINT: Ruff errors detected. Try: uv run ruff check . --fix"
fi

if echo "$OUTPUT" | grep -qi "pytest.*collection.*error\|conftest"; then
    echo "HINT: Pytest collection error. Check conftest.py and __init__.py files exist."
fi

if echo "$OUTPUT" | grep -qi "uvicorn.*error\|Error loading ASGI"; then
    echo "HINT: Uvicorn error. Check port availability and import paths (src.main:app)."
fi

if echo "$OUTPUT" | grep -q "Connection refused"; then
    echo "HINT: Connection refused. Check if Qdrant container is running: docker ps | grep qdrant"
fi

if echo "$OUTPUT" | grep -q "OPENAI_API_KEY"; then
    echo "HINT: OPENAI_API_KEY not set. Check .env file exists with a valid key."
fi

if echo "$OUTPUT" | grep -q "dimension mismatch"; then
    echo "HINT: Dimension mismatch. Embedding model changed — recreate the Qdrant collection."
fi
