#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/usr/bin/python3"
USER_SITE="$(${PYTHON_BIN} -c 'import site; print(site.getusersitepackages())')"

exec sudo -E env \
    PYTHONPATH="${USER_SITE}:${SCRIPT_DIR}" \
    MPLCONFIGDIR="/tmp/imx219-stereo-matplotlib" \
    MPLBACKEND="Agg" \
    "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/app.py"
