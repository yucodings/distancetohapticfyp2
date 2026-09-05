#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="/usr/bin/python3"
USER_SITE="$(${PYTHON_BIN} -c 'import site; print(site.getusersitepackages())')"

# Keep the system RealSense binding first. Ultralytics and the Jetson PyTorch
# build are installed in the desktop user's local site-packages directory.
PYTHON_PATH="${USER_SITE}:${SCRIPT_DIR}"

exec sudo -E env \
    PYTHONPATH="${PYTHON_PATH}" \
    MPLCONFIGDIR="/tmp/realsense-matplotlib" \
    MPLBACKEND="Agg" \
    "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/distance_to_haptic_system_actuators_disabled.py"
