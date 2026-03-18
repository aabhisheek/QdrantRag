#!/usr/bin/env bash
# Runs after every Write/Edit tool call on src/ or tests/ files.
# Verifies the app can start — catches startup bugs before the user runs the server.

FILE_PATH="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

# Only run for Python source files
if [[ "$FILE_PATH" != *src/*.py ]] && [[ "$FILE_PATH" != *tests/*.py ]]; then
    exit 0
fi

echo "=== Startup smoke test (triggered by edit to $FILE_PATH) ==="

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python scripts/startup_check.py
EXIT=$?

if [ $EXIT -ne 0 ]; then
    echo "FAIL: App startup check failed after editing $FILE_PATH"
    echo "Fix the startup error before proceeding."
    exit 1
fi
