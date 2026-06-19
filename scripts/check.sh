#!/usr/bin/env bash
# Project verification gate: byte-compile every module, then run the test suite.
# Run this BEFORE and AFTER any change to confirm nothing regressed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Byte-compiling sources"
python -m py_compile scraper.py summarizer.py transcript.py helpers.py sendToTelegram.py log.py

echo "==> Running tests"
python -m pytest -q

echo "==> All checks passed"
