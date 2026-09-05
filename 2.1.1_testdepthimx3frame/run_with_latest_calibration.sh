#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION=""
PROFILE="$SCRIPT_DIR/../1.1_depth_visualization/results/vpi_recommended_profile.json"
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

if [[ ! -f "$PROFILE" ]]; then
    printf 'ERROR: no schema-2 VPI Easy Mode profile was found.\n' >&2
    printf 'Run 1.1_depth_visualization/run_easy_tuner.sh first.\n' >&2
    exit 1
fi

printf 'Using calibration: %s\n' "$CALIBRATION"
printf 'Using VPI profile: %s\n' "$PROFILE"
exec python3 "$SCRIPT_DIR/test_depth_imx219_3frame.py" \
    --calibration "$CALIBRATION" \
    --vpi-profile "$PROFILE" \
    --backend vpi-cuda \
    "$@"
