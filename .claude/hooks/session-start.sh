#!/bin/bash
# SessionStart hook: install runtime + dev dependencies so tests and the
# verification script work immediately in Claude Code on the web sessions.
# Synchronous (no async block) so deps are guaranteed ready before the agent runs.
set -euo pipefail

cd "$CLAUDE_PROJECT_DIR"

# Only needed in the remote (web) environment; locals manage their own venvs.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

python -m pip install --quiet -r requirements-dev.txt
