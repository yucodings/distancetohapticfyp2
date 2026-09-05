#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USER_SITE="$("$PYTHON_BIN" -c 'import site; print(site.getusersitepackages())')"

exec sudo -E env \
    PYTHONPATH="${USER_SITE}:${SCRIPT_DIR}" \
    "$PYTHON_BIN" -B \
    "$SCRIPT_DIR/test_depth_realsense_3frame.py" \
    "$@"
