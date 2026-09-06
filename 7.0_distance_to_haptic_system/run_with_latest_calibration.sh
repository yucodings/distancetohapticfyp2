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
    printf 'ERROR: no completed calibration was found under 1.0_calibration/images.\n' >&2
    exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
    printf 'ERROR: VPI profile not found: %s\n' "$PROFILE" >&2
    exit 1
fi

printf 'Using calibration: %s\n' "$CALIBRATION"
printf 'Using VPI profile: %s\n' "$PROFILE"
exec /usr/bin/python3 -B "$SCRIPT_DIR/app.py" \
    --calibration "$CALIBRATION" \
    --vpi-profile "$PROFILE" \
    "$@"

