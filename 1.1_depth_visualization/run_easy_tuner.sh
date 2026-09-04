#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION="$SCRIPT_DIR/../1.0_calibration/images/2026-08-31_16-41-19_757318/stereo_calibration.npz"

printf '%s\n' \
  "EASY MODE" \
  "1. Put a textured box or flat object exactly 1.0 metre from the cameras." \
  "2. Keep it in the centre of both images and keep everything still." \
  "3. When the windows appear, press A once to tune native VPI CUDA settings." \
  "4. The best profile is applied automatically to the 3.0 project." \
  "5. Press Q when finished."

exec python3 "$SCRIPT_DIR/stereo_depth_tuner.py" \
  --calibration "$CALIBRATION" \
  --known-distance-m 1.0 \
  --backend vpi \
  "$@"
