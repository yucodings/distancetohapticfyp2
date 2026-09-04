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

printf '%s\n' \
  "EASY MODE" \
  "1. Put a textured box or flat object exactly 1.0 metre from the cameras." \
  "2. Keep it in the centre of both images and keep everything still." \
  "3. When the windows appear, press A once to tune native VPI CUDA settings." \
  "4. The best profile is applied automatically to the 3.0 project." \
  "5. Press Q when finished." \
  "Calibration: $CALIBRATION"

exec python3 "$SCRIPT_DIR/stereo_depth_tuner.py" \
  --calibration "$CALIBRATION" \
  --known-distance-m 1.0 \
  --backend vpi \
  "$@"
