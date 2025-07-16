#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e .
