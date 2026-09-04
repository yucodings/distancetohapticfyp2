#!/bin/bash

# Jetson Orin Nano + TCA9548A + 3x DA7280 simultaneous pulse test
#
# TCA SC2 -> Left actuator
# TCA SC3 -> Center actuator
# TCA SC4 -> Right actuator
#
# All DA7280s use address 0x4A. They are inspected individually, then the
# TCA9548A enables SC2/SC3/SC4 together so identical write commands are
# broadcast to all three drivers. Never read a DA7280 register while the
# combined mask is selected because same-address read responses can contend.

BUS=1
MUX_ADDR=0x70
HAPTIC_ADDR=0x4A

CHIP_REV=0x00
EXPECTED_CHIP_REV=0xBA
TOP_CTL1=0x22
OVERRIDE_VAL=0x23

SC2_MASK=0x04
SC3_MASK=0x08
SC4_MASK=0x10
ALL_ACTUATORS_MASK=0x1C

# OVERRIDE_VAL range is 0..127. This test deliberately starts low.
AMPLITUDE=25
PULSE_TIME=0.20

CHANNEL_NAMES=("Left" "Center" "Right")
CHANNEL_NUMBERS=(2 3 4)
CHANNEL_MASKS=($SC2_MASK $SC3_MASK $SC4_MASK)

COMMON_INACTIVE_CTL=""
SHUTDOWN_DONE=0


mux_off()
{
    i2cset -y "$BUS" "$MUX_ADDR" 0x00
}


select_mask()
{
    local mask=$1
    mux_off || return 1
    i2cset -y "$BUS" "$MUX_ADDR" "$mask"
}


safe_shutdown()
{
    if [ "$SHUTDOWN_DONE" -eq 1 ]; then
        return
    fi
    SHUTDOWN_DONE=1

    # Use best-effort operations so cleanup continues after an I2C failure.
    set +e
    i2cset -y "$BUS" "$MUX_ADDR" "$ALL_ACTUATORS_MASK"
    i2cset -y "$BUS" "$HAPTIC_ADDR" "$OVERRIDE_VAL" 0x00

    if [ -n "$COMMON_INACTIVE_CTL" ]; then
        # All three upper TOP_CTL1 bits were verified equal before the pulse.
        i2cset -y \
            "$BUS" \
            "$HAPTIC_ADDR" \
            "$TOP_CTL1" \
            "$COMMON_INACTIVE_CTL"
    fi

    mux_off
}


trap 'safe_shutdown; exit 130' INT TERM
trap 'safe_shutdown' EXIT


echo "========================================"
echo " DA7280 Simultaneous Haptic Pulse Test"
echo "========================================"
echo
echo "Mapping: Left=SC2, Center=SC3, Right=SC4"
printf "Combined TCA mask : 0x%02X\n" "$ALL_ACTUATORS_MASK"
printf "Strength          : %d/127 (0x%02X)\n" "$AMPLITUDE" "$AMPLITUDE"
echo "Pulse duration    : ${PULSE_TIME}s"
echo


# Verify each same-address DA7280 separately. Also force zero/inactive and
# confirm that the unrelated TOP_CTL1 bits match before broadcasting TOP_CTL1.
for index in "${!CHANNEL_MASKS[@]}"; do
    name=${CHANNEL_NAMES[$index]}
    channel=${CHANNEL_NUMBERS[$index]}
    mask=${CHANNEL_MASKS[$index]}

    printf "Checking %s actuator on SC%d... " "$name" "$channel"
    if ! select_mask "$mask"; then
        echo "FAILED (could not select mux channel)"
        exit 1
    fi

    revision=$(i2cget -y "$BUS" "$HAPTIC_ADDR" "$CHIP_REV" 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "FAILED (DA7280 not detected)"
        exit 1
    fi
    if (( revision != EXPECTED_CHIP_REV )); then
        printf "FAILED (CHIP_REV=%s, expected 0x%02X)\n" \
            "$revision" "$EXPECTED_CHIP_REV"
        exit 1
    fi

    if ! i2cset -y "$BUS" "$HAPTIC_ADDR" "$OVERRIDE_VAL" 0x00; then
        echo "FAILED (could not zero OVERRIDE_VAL)"
        exit 1
    fi

    current_ctl=$(i2cget -y "$BUS" "$HAPTIC_ADDR" "$TOP_CTL1" 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "FAILED (could not read TOP_CTL1)"
        exit 1
    fi

    inactive_ctl=$((current_ctl & 0xF8))
    printf -v inactive_hex "0x%02X" "$inactive_ctl"
    if ! i2cset -y "$BUS" "$HAPTIC_ADDR" "$TOP_CTL1" "$inactive_hex"; then
        echo "FAILED (could not enter inactive mode)"
        exit 1
    fi

    if [ -z "$COMMON_INACTIVE_CTL" ]; then
        COMMON_INACTIVE_CTL=$inactive_hex
    elif (( inactive_ctl != COMMON_INACTIVE_CTL )); then
        echo "FAILED (TOP_CTL1 upper bits differ between DA7280s)"
        echo "A shared TOP_CTL1 write would not safely preserve each device's bits."
        exit 1
    fi

    echo "OK"
done


# Broadcast only writes after enabling all three mux channels. OVERRIDE_VAL is
# written before OPERATION_MODE=1, matching the DA7280 DRO startup sequence.
if ! select_mask "$ALL_ACTUATORS_MASK"; then
    echo "FAILED: could not enable SC2, SC3, and SC4 together"
    exit 1
fi

mux_status=$(i2cget -y "$BUS" "$MUX_ADDR" 2>/dev/null)
if [ $? -ne 0 ] || (( mux_status != ALL_ACTUATORS_MASK )); then
    echo "FAILED: combined mux mask was not accepted"
    exit 1
fi

printf -v amplitude_hex "0x%02X" "$AMPLITUDE"
dro_ctl=$((COMMON_INACTIVE_CTL | 0x01))
printf -v dro_hex "0x%02X" "$dro_ctl"

if ! i2cset -y "$BUS" "$HAPTIC_ADDR" "$OVERRIDE_VAL" "$amplitude_hex"; then
    echo "FAILED: could not broadcast amplitude"
    exit 1
fi
if ! i2cset -y "$BUS" "$HAPTIC_ADDR" "$TOP_CTL1" "$dro_hex"; then
    echo "FAILED: could not broadcast DRO mode"
    exit 1
fi

echo
echo ">>> LEFT + CENTER + RIGHT ON"
sleep "$PULSE_TIME"
echo ">>> LEFT + CENTER + RIGHT OFF"

safe_shutdown
trap - EXIT

echo
echo "========================================"
echo " Simultaneous pulse test completed"
echo " All three amplitudes are zero"
echo " All three DA7280s are inactive"
echo " All TCA9548A channels are disabled"
echo "========================================"
