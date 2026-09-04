#!/bin/bash

# ============================================================
# Jetson Orin Nano + TCA9548A + 3x DA7280
#
# Jetson Orin Nano:
#   I2C Bus = /dev/i2c-1
#
# TCA9548A:
#   Address = 0x70
#
# Actual wiring:
#   CH2 -> DA7280 #1
#   CH3 -> DA7280 #2
#   CH4 -> DA7280 #3
#
# DA7280 Address = 0x4A
# ============================================================

BUS=1
MUX_ADDR=0x70
HAPTIC_ADDR=0x4A


echo "========================================"
echo " Jetson Orin Nano Haptic I2C Test"
echo " TCA9548A + 3x DA7280"
echo "========================================"
echo


# ============================================================
# Check I2C bus
# ============================================================

if [ ! -e "/dev/i2c-$BUS" ]; then
    echo "[ERROR] /dev/i2c-$BUS does not exist."
    echo
    echo "Available buses:"
    i2cdetect -l
    exit 1
fi

echo "[OK] Using /dev/i2c-$BUS"
echo


# ============================================================
# Check TCA9548A
# ============================================================

echo "Checking TCA9548A at $MUX_ADDR..."

MUX_VALUE=$(i2cget -y $BUS $MUX_ADDR 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "[FAIL] TCA9548A not detected."
    echo
    echo "Try:"
    echo "sudo i2cdetect -y -r $BUS"
    exit 1
fi

echo "[OK] TCA9548A detected."
echo "Current mux register: $MUX_VALUE"
echo


# ============================================================
# Disable all TCA9548A channels
# ============================================================

mux_off()
{
    i2cset -y $BUS $MUX_ADDR 0x00
    sleep 0.05
}


# ============================================================
# Test one DA7280
# ============================================================

test_haptic()
{
    MOTOR=$1
    CHANNEL=$2
    MASK=$3

    echo "========================================"
    echo "Testing Haptic Driver #$MOTOR"
    echo "TCA9548A Channel : CH$CHANNEL"
    echo "Channel Mask     : $MASK"
    echo "========================================"

    # Disable all channels
    mux_off

    # Select requested channel
    if ! i2cset -y $BUS $MUX_ADDR $MASK; then
        echo "[FAIL] Unable to select CH$CHANNEL"
        echo
        return
    fi

    sleep 0.1

    # Read back mux status
    STATUS=$(i2cget -y $BUS $MUX_ADDR 2>/dev/null)

    echo "Mux register = $STATUS"

    if [ "$STATUS" = "$MASK" ]; then
        echo "[OK] CH$CHANNEL selected."
    else
        echo "[FAIL] Expected $MASK, received $STATUS"
        echo
        return
    fi

    echo
    echo "Checking DA7280 #$MOTOR at 0x4A..."

    # Register 0x00 is readable on DA7280
    CHIP_VALUE=$(i2cget -y $BUS $HAPTIC_ADDR 0x00 2>/dev/null)

    if [ $? -eq 0 ]; then
        echo "[OK] DA7280 #$MOTOR detected."
        echo "Register 0x00 = $CHIP_VALUE"
    else
        echo "[FAIL] DA7280 #$MOTOR not detected."
    fi

    echo
}


# ============================================================
# Actual channel mapping
# ============================================================

# Haptic #1 -> CH2
test_haptic 1 2 0x04

# Haptic #2 -> CH3
test_haptic 2 3 0x08

# Haptic #3 -> CH4
test_haptic 3 4 0x10


# ============================================================
# Shutdown
# ============================================================

mux_off

echo "========================================"
echo " Test complete"
echo " All TCA9548A channels OFF"
echo "========================================"
