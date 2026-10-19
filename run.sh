#!/usr/bin/env bash
# Convenience launcher for the AI News Agent.
# Uses the project venv when present; bootstraps one if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x "./.venv/bin/python" ]; then
    echo "[run.sh] no .venv found - bootstrapping" >&2
    BOOTSTRAP="$(command -v python3.12 || command -v python3)"
    "$BOOTSTRAP" -m venv .venv
    ./.venv/bin/python -m pip install -q --upgrade pip
    ./.venv/bin/python -m pip install -r requirements.txt
fi

exec ./.venv/bin/python news_agent.py "$@"
