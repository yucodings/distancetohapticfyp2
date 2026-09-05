#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/usr/bin/python3"
USER_SITE="$(${PYTHON_BIN} -c 'import site; print(site.getusersitepackages())')"

# The camera currently requires root access on this Jetson. Preserve the
# desktop display environment and expose the desktop user's Python packages,
# while the Python program itself prioritizes the system RealSense binding.
exec sudo -E env \
    PYTHONPATH="${USER_SITE}:${SCRIPT_DIR}" \
    "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/test_depth_realsense.py" \
    "$@"
