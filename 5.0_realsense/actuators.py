"""Thread-safe control for three DA7280s behind one TCA9548A.

The DA7280 generates and tracks the LRA carrier. This module changes only
OVERRIDE_VAL to create the much slower software pulse envelope.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Mapping, Optional, Protocol

from smbus2 import SMBus

from config import (
    DA7280_ADDR,
    I2C_BUS,
    REG_CHIP_REV,
    REG_IRQ_EVENT1,
    REG_OVERRIDE_VAL,
    REG_TOP_CTL1,
    TCA9548A_ADDR,
    TCA_CHANNELS,
)
from data_models import MotorPattern, PATTERN_OFF


ErrorCallback = Callable[[str], None]


class SMBusLike(Protocol):
    def write_byte(self, address: int, value: int) -> None: ...

    def read_byte_data(self, address: int, register: int) -> int: ...

    def write_byte_data(self, address: int, register: int, value: int) -> None: ...

    def close(self) -> None: ...


class TCA9548ABus:
    """Own the shared bus and make mux-select plus device access atomic."""

    def __init__(
        self,
        bus_number: int,
        mux_address: int,
        bus_factory: Callable[[int], SMBusLike] = SMBus,
    ):
        self.bus_number = bus_number
        self.mux_address = mux_address
        self._bus_factory = bus_factory
        self._bus: Optional[SMBusLike] = None
        self._lock = threading.RLock()

    def open(self) -> None:
        with self._lock:
            if self._bus is None:
                self._bus = self._bus_factory(self.bus_number)

    def _require_bus(self) -> SMBusLike:
        if self._bus is None:
            raise RuntimeError("TCA9548A I2C bus is not open")
        return self._bus

    def _select_locked(self, channel: int) -> None:
        if not 0 <= channel <= 7:
            raise ValueError(f"Invalid TCA9548A channel: {channel}")
        self._require_bus().write_byte(self.mux_address, 1 << channel)

    def read_register(self, channel: int, device: int, register: int) -> int:
        with self._lock:
            self._select_locked(channel)
            return self._require_bus().read_byte_data(device, register)

    def write_register(
        self, channel: int, device: int, register: int, value: int
    ) -> None:
        with self._lock:
            self._select_locked(channel)
            self._require_bus().write_byte_data(device, register, value & 0xFF)

    def update_register_bits(
        self,
        channel: int,
        device: int,
        register: int,
        mask: int,
        value: int,
    ) -> int:
        """Perform one locked mux-select/read/modify/write transaction."""
        with self._lock:
            self._select_locked(channel)
            bus = self._require_bus()
            current = bus.read_byte_data(device, register)
            updated = (current & (~mask & 0xFF)) | (value & mask)
            if updated != current:
                bus.write_byte_data(device, register, updated)
            return updated

    def disable_all(self) -> None:
        with self._lock:
            self._require_bus().write_byte(self.mux_address, 0x00)

    def close(self) -> None:
        with self._lock:
            if self._bus is None:
                return
            bus = self._bus
            try:
                bus.write_byte(self.mux_address, 0x00)
            finally:
                try:
                    bus.close()
                finally:
                    self._bus = None


class DA7280Driver:
    """One DA7280 operated in direct-register-override (DRO) mode."""

    EXPECTED_CHIP_REV = 0xBA
    OPERATION_MODE_MASK = 0x07
    OPERATION_MODE_INACTIVE = 0x00
    OPERATION_MODE_DRO = 0x01

    def __init__(self, transport: TCA9548ABus, channel: int, name: str):
        self.transport = transport
        self.channel = channel
        self.name = name
        self._dro_active = False
        self._drive_level = 0

    def _read(self, register: int) -> int:
        return self.transport.read_register(self.channel, DA7280_ADDR, register)

    def _write(self, register: int, value: int) -> None:
        self.transport.write_register(self.channel, DA7280_ADDR, register, value)

    def _set_operation_mode(self, mode: int) -> None:
        # TOP_CTL1 bits 2:0 are OPERATION_MODE. The locked read-modify-write
        # preserves STANDBY_EN, SEQ_START, and the upper reserved bits.
        self.transport.update_register_bits(
            self.channel,
            DA7280_ADDR,
            REG_TOP_CTL1,
            self.OPERATION_MODE_MASK,
            mode,
        )

    def initialize(self) -> None:
        revision = self._read(REG_CHIP_REV)
        if revision != self.EXPECTED_CHIP_REV:
            raise RuntimeError(
                f"{self.name} SC{self.channel}: CHIP_REV=0x{revision:02X}; "
                f"expected 0x{self.EXPECTED_CHIP_REV:02X}"
            )

        # Clear stale write-one-to-clear events. Actuator voltage, current,
        # impedance, and resonant-period registers are intentionally untouched:
        # they must be set only from the exact LRA datasheet.
        self._write(REG_IRQ_EVENT1, 0xFF)
        self.stop(force=True)

    @staticmethod
    def _validated_level(level: int) -> int:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError(f"DA7280 drive level must be an integer, got {level!r}")
        if not 0 <= level <= 0x7F:
            raise ValueError(
                f"DA7280 drive level must be 0x00..0x7F, got {level!r}"
            )
        return level

    def enter_dro(self, initial_level: int) -> None:
        """Write OVERRIDE_VAL before entering DRO, as required by the datasheet."""
        level = self._validated_level(initial_level)
        if level == 0:
            raise ValueError("Use stop() instead of entering DRO at zero amplitude")

        if self._drive_level != level:
            self._write(REG_OVERRIDE_VAL, level)
            self._drive_level = level
        if not self._dro_active:
            self._set_operation_mode(self.OPERATION_MODE_DRO)
            self._dro_active = True

    def set_drive_level(self, level: int) -> None:
        """Change strength without leaving DRO (zero is the pulse off phase)."""
        level = self._validated_level(level)
        if not self._dro_active:
            if level == 0:
                return
            self.enter_dro(level)
            return
        if level == self._drive_level:
            return
        self._write(REG_OVERRIDE_VAL, level)
        self._drive_level = level

    def stop(self, force: bool = False) -> None:
        """Zero the amplitude, then select inactive mode; avoid repeated off writes."""
        if force or self._drive_level != 0:
            self._write(REG_OVERRIDE_VAL, 0x00)
        self._drive_level = 0
        if force or self._dro_active:
            self._set_operation_mode(self.OPERATION_MODE_INACTIVE)
        self._dro_active = False


class HapticController:
    """Apply a continuous or software-timed pulse envelope to one driver."""

    def __init__(
        self,
        driver: DA7280Driver,
        error_callback: Optional[ErrorCallback] = None,
    ):
        self.driver = driver
        self.error_callback = error_callback
        self._command = PATTERN_OFF
        self._condition = threading.Condition()
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.driver.initialize()
        with self._condition:
            self._stop_requested = False
            self._command = PATTERN_OFF
        self._thread = threading.Thread(
            target=self._run,
            name=f"haptic-{self.driver.name.lower()}",
            daemon=True,
        )
        self._thread.start()

    def set_pattern(self, pattern: MotorPattern) -> None:
        if not isinstance(pattern, MotorPattern):
            raise TypeError(f"Expected MotorPattern, got {type(pattern).__name__}")
        with self._condition:
            if pattern != self._command:
                self._command = pattern
                self._condition.notify_all()

    def _snapshot(self) -> tuple[bool, MotorPattern]:
        with self._condition:
            return self._stop_requested, self._command

    def _wait_for_change(self, pattern: MotorPattern, timeout: Optional[float]) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._stop_requested or self._command != pattern,
                timeout=timeout,
            )

    def _run_pulsed(self, pattern: MotorPattern) -> None:
        # Python creates only the envelope. The DA7280 continues generating
        # the resonant LRA carrier during each non-zero portion.
        self.driver.enter_dro(pattern.level)
        while True:
            if self._wait_for_change(pattern, pattern.on_time):
                return
            self.driver.set_drive_level(0x00)
            if self._wait_for_change(pattern, pattern.off_time):
                return
            self.driver.set_drive_level(pattern.level)

    def _run(self) -> None:
        try:
            while True:
                stop_requested, pattern = self._snapshot()
                if stop_requested:
                    return

                if pattern == PATTERN_OFF:
                    self.driver.stop()
                    self._wait_for_change(pattern, None)
                elif pattern.on_time == 0.0 and pattern.off_time == 0.0:
                    self.driver.enter_dro(pattern.level)
                    self._wait_for_change(pattern, None)
                    self.driver.stop()
                else:
                    self._run_pulsed(pattern)
                    self.driver.stop()
        except Exception as error:
            if self.error_callback is not None:
                self.error_callback(
                    f"{self.driver.name} actuator stopped after I2C/controller "
                    f"error: {error}"
                )
        finally:
            try:
                self.driver.stop(force=True)
            except Exception:
                pass

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._command = PATTERN_OFF
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None


class ActuatorManager:
    """Route raw-zone MotorPatterns to left/center/right DA7280 drivers."""

    def __init__(
        self,
        error_callback: Optional[ErrorCallback] = None,
        transport: Optional[TCA9548ABus] = None,
    ):
        self.transport = transport or TCA9548ABus(I2C_BUS, TCA9548A_ADDR)
        self.error_callback = error_callback
        self._healthy = True
        self._fault_lock = threading.Lock()
        self.controllers: Dict[str, HapticController] = {
            zone_name: HapticController(
                DA7280Driver(self.transport, channel, zone_name.upper()),
                self._handle_controller_error,
            )
            for zone_name, channel in TCA_CHANNELS.items()
        }
        self.started = False

    def _request_all_off(self) -> None:
        for controller in self.controllers.values():
            try:
                controller.set_pattern(PATTERN_OFF)
            except Exception:
                pass

    def _handle_controller_error(self, message: str) -> None:
        with self._fault_lock:
            first_fault = self._healthy
            self._healthy = False
        self._request_all_off()
        if first_fault and self.error_callback is not None:
            self.error_callback(message + "; all three actuators requested off")

    def start(self) -> None:
        self.transport.open()
        started = []
        try:
            for zone_name in TCA_CHANNELS:
                self.controllers[zone_name].start()
                started.append(zone_name)
            self.started = True
        except Exception:
            self._healthy = False
            self._request_all_off()
            for zone_name in reversed(started):
                self.controllers[zone_name].stop()
            # Best-effort safe state for every downstream device, including a
            # device whose initialization failed part-way through.
            for controller in self.controllers.values():
                try:
                    controller.driver.stop(force=True)
                except Exception:
                    pass
            self.transport.close()
            raise

    def update_patterns(self, patterns: Mapping[str, MotorPattern]) -> None:
        if not self.started or not self._healthy:
            return
        unknown = set(patterns) - set(self.controllers)
        if unknown:
            raise ValueError(f"Unknown actuator zone(s): {sorted(unknown)}")
        if any(not isinstance(value, MotorPattern) for value in patterns.values()):
            raise TypeError("ActuatorManager accepts MotorPattern values only")
        for zone_name, controller in self.controllers.items():
            controller.set_pattern(patterns.get(zone_name, PATTERN_OFF))

    def stop(self) -> None:
        self._request_all_off()
        for controller in self.controllers.values():
            controller.stop()
        self.started = False
        self.transport.close()  # also writes 0x00 to disable every mux channel
