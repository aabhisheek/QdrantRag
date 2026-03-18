#!/usr/bin/env bash
set -euo pipefail

NOTIFICATION="${NOTIFICATION_MESSAGE:-}"

if echo "$NOTIFICATION" | grep -qi "permission"; then
    echo "HINT: Permission issue. Check .claude/settings.json allowlist for permitted tools and paths."
fi

if echo "$NOTIFICATION" | grep -qi "rate limit"; then
    echo "HINT: Rate limited. Wait a moment and retry the request."
fi

if echo "$NOTIFICATION" | grep -qi "error"; then
    echo "HINT: An error occurred. Read the full error message for details."
fi

if echo "$NOTIFICATION" | grep -qi "complete"; then
    echo "Task completed successfully."
fi
