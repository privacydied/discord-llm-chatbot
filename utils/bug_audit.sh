#!/bin/bash
set -euo pipefail

REPO="/volume1/py/discord-llm-chatbot"
BRANCH="main"

cd "$REPO"

# sync latest from remote (non-destructive)
git fetch origin "$BRANCH" --quiet

# run fixes
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
uv run ruff check . --fix -q --exit-zero || true
uv run ruff format . -q || true

# commit + push only when something changed
if ! git diff --quiet --exit-code; then
  git add -A
  git commit -m "chore(bug-audit): automated patch ($(date +%Y-%m-%d %H:%M))"
  git push origin "$BRANCH"
  echo "pushed"
else
  echo "nothing-to-fix"
fi
