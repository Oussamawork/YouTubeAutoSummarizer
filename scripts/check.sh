#!/usr/bin/env bash
# Project verification gate: byte-compile every module, then run the test suite.
# Run this BEFORE and AFTER any change to confirm nothing regressed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Byte-compiling sources"
# Every module, not a hand-kept list: the list drifted to 6 of 15 files.
python -m compileall -q ./*.py tests/*.py

echo "==> Running tests"
python -m pytest -q

echo "==> All checks passed"
