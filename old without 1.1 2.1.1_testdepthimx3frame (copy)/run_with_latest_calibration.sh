#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION=""
for CANDIDATE in "$SCRIPT_DIR"/../1.0_calibration/images/*/stereo_calibration.npz; do
    [[ -f "$CANDIDATE" ]] || continue
    if [[ -z "$CALIBRATION" || "$CANDIDATE" -nt "$CALIBRATION" ]]; then
        CALIBRATION="$CANDIDATE"
    fi
done

if [[ -z "$CALIBRATION" ]]; then
    printf 'ERROR: no completed 1.0 calibration was found.\n' >&2
    printf 'Run 1.0_calibration/mycalibration.py first.\n' >&2
    exit 1
fi

printf 'Using calibration: %s\n' "$CALIBRATION"
printf 'Using 2.1.1-owned runtime stereo settings (no 1.1 profile).\n'
exec python3 -B "$SCRIPT_DIR/test_depth_imx219_3frame.py" \
    --calibration "$CALIBRATION" \
    --backend vpi-cuda \
    "$@"
