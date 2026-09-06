import time
import unittest

from actuators import ActuatorManager, DA7280Driver, TCA9548ABus
from config import (
    DA7280_ADDR,
    REG_CHIP_REV,
    REG_IRQ_EVENT1,
    REG_OVERRIDE_VAL,
    REG_TOP_CTL1,
    TCA9548A_ADDR,
    TCA_CHANNELS,
)
from data_models import Detection, PATTERN_OFF
from haptic_policy import PATTERN_SLOW


class RecordingTransport:
    def __init__(self, chip_revision=0xBA, top_ctl1=0xD8):
        self.chip_revision = chip_revision
        self.top_ctl1 = top_ctl1
        self.events = []

    def read_register(self, channel, device, register):
        self.events.append(("read", channel, device, register))
        if register == REG_CHIP_REV:
            return self.chip_revision
        if register == REG_TOP_CTL1:
            return self.top_ctl1
        return 0

    def write_register(self, channel, device, register, value):
        self.events.append(("write", channel, device, register, value))

    def update_register_bits(self, channel, device, register, mask, value):
        previous = self.top_ctl1
        self.top_ctl1 = (previous & (~mask & 0xFF)) | (value & mask)
        self.events.append(
            ("update", channel, device, register, mask, value, previous, self.top_ctl1)
        )
        return self.top_ctl1


class MockSMBus:
    def __init__(self, bus_number):
        self.bus_number = bus_number
        self.events = []
        self.registers = {(DA7280_ADDR, REG_TOP_CTL1): 0xD8}
        self.closed = False

    def write_byte(self, address, value):
        self.events.append(("write_byte", address, value))

    def read_byte_data(self, address, register):
        self.events.append(("read_byte_data", address, register))
        return self.registers.get((address, register), 0)

    def write_byte_data(self, address, register, value):
        self.events.append(("write_byte_data", address, register, value))
        self.registers[(address, register)] = value

    def close(self):
        self.events.append(("close",))
        self.closed = True


class ThreeChannelMockSMBus:
    def __init__(self, bus_number):
        self.bus_number = bus_number
        self.selected_mask = 0
        self.events = []
        self.registers = {}
        self.closed = False

    def write_byte(self, address, value):
        self.selected_mask = value
        self.events.append(("mux", address, value))

    def read_byte_data(self, address, register):
        self.events.append(("read", self.selected_mask, address, register))
        if register == REG_CHIP_REV:
            return 0xBA
        return self.registers.get((self.selected_mask, address, register), 0)

    def write_byte_data(self, address, register, value):
        self.events.append(
            ("write", self.selected_mask, address, register, value)
        )
        self.registers[(self.selected_mask, address, register)] = value

    def close(self):
        self.events.append(("close",))
        self.closed = True


class DA7280RegisterSequenceTests(unittest.TestCase):
    def test_initialize_verifies_revision_and_forces_safe_state(self):
        transport = RecordingTransport()
        driver = DA7280Driver(transport, 2, "LEFT")

        driver.initialize()

        self.assertEqual(
            transport.events,
            [
                ("read", 2, DA7280_ADDR, REG_CHIP_REV),
                ("write", 2, DA7280_ADDR, REG_IRQ_EVENT1, 0xFF),
                ("write", 2, DA7280_ADDR, REG_OVERRIDE_VAL, 0x00),
                ("update", 2, DA7280_ADDR, REG_TOP_CTL1, 0x07, 0x00, 0xD8, 0xD8),
            ],
        )

    def test_wrong_chip_revision_is_rejected_before_drive_writes(self):
        transport = RecordingTransport(chip_revision=0xB9)
        driver = DA7280Driver(transport, 2, "LEFT")

        with self.assertRaisesRegex(RuntimeError, "expected 0xBA"):
            driver.initialize()

        self.assertEqual(
            transport.events,
            [("read", 2, DA7280_ADDR, REG_CHIP_REV)],
        )

    def test_dro_order_pulse_off_and_complete_stop(self):
        transport = RecordingTransport(top_ctl1=0xD8)
        driver = DA7280Driver(transport, 3, "CENTER")

        driver.enter_dro(0x60)
        self.assertEqual(
            transport.events,
            [
                ("write", 3, DA7280_ADDR, REG_OVERRIDE_VAL, 0x60),
                ("update", 3, DA7280_ADDR, REG_TOP_CTL1, 0x07, 0x01, 0xD8, 0xD9),
            ],
        )
        self.assertEqual(transport.top_ctl1, 0xD9)

        transport.events.clear()
        driver.set_drive_level(0x00)
        driver.set_drive_level(0x00)
        self.assertEqual(
            transport.events,
            [("write", 3, DA7280_ADDR, REG_OVERRIDE_VAL, 0x00)],
        )
        self.assertEqual(transport.top_ctl1, 0xD9, "pulse-off must remain in DRO")

        transport.events.clear()
        driver.stop()
        driver.stop()
        self.assertEqual(
            transport.events,
            [("update", 3, DA7280_ADDR, REG_TOP_CTL1, 0x07, 0x00, 0xD9, 0xD8)],
        )

    def test_mux_transaction_selects_channel_and_preserves_top_ctl1_bits(self):
        created = []

        def factory(bus_number):
            bus = MockSMBus(bus_number)
            created.append(bus)
            return bus

        transport = TCA9548ABus(1, TCA9548A_ADDR, bus_factory=factory)
        transport.open()
        updated = transport.update_register_bits(
            4, DA7280_ADDR, REG_TOP_CTL1, 0x07, 0x01
        )
        transport.close()

        self.assertEqual(updated, 0xD9)
        self.assertEqual(
            created[0].events,
            [
                ("write_byte", TCA9548A_ADDR, 1 << 4),
                ("read_byte_data", DA7280_ADDR, REG_TOP_CTL1),
                ("write_byte_data", DA7280_ADDR, REG_TOP_CTL1, 0xD9),
                ("write_byte", TCA9548A_ADDR, 0x00),
                ("close",),
            ],
        )


class ManagerSafetyTests(unittest.TestCase):
    def test_confirmed_hardware_mapping(self):
        self.assertEqual(
            TCA_CHANNELS,
            {"left": 2, "center": 3, "right": 4},
        )

    def test_non_pattern_values_cannot_be_actuator_commands(self):
        manager = ActuatorManager(transport=RecordingTransport())
        manager.started = True

        with self.assertRaisesRegex(TypeError, "MotorPattern values only"):
            manager.update_patterns({"left": object()})

    def test_yolo_detection_cannot_be_an_actuator_command(self):
        manager = ActuatorManager(transport=RecordingTransport())
        manager.started = True
        detection = Detection(3, "person", 0.9, (0, 0, 20, 30))

        with self.assertRaisesRegex(TypeError, "MotorPattern values only"):
            manager.update_patterns({"left": detection})

    def test_controller_fault_requests_every_zone_off_only_once(self):
        messages = []
        manager = ActuatorManager(messages.append, transport=RecordingTransport())

        class FakeController:
            def __init__(self):
                self.patterns = []

            def set_pattern(self, pattern):
                self.patterns.append(pattern)

        manager.controllers = {
            "left": FakeController(),
            "center": FakeController(),
            "right": FakeController(),
        }

        manager._handle_controller_error("I2C failed")
        manager._handle_controller_error("I2C failed again")

        for controller in manager.controllers.values():
            self.assertEqual(controller.patterns, [PATTERN_OFF, PATTERN_OFF])
        self.assertEqual(
            messages,
            ["I2C failed; all three actuators requested off"],
        )

    def test_manager_accepts_motor_patterns(self):
        manager = ActuatorManager(transport=RecordingTransport())
        manager.started = True

        class FakeController:
            def __init__(self):
                self.pattern = None

            def set_pattern(self, pattern):
                self.pattern = pattern

        manager.controllers = {name: FakeController() for name in TCA_CHANNELS}
        manager.update_patterns({"left": PATTERN_SLOW})

        self.assertEqual(manager.controllers["left"].pattern, PATTERN_SLOW)
        self.assertEqual(manager.controllers["center"].pattern, PATTERN_OFF)
        self.assertEqual(manager.controllers["right"].pattern, PATTERN_OFF)

    def test_shutdown_zeros_every_driver_enters_inactive_and_disables_mux(self):
        created = []

        def factory(bus_number):
            bus = ThreeChannelMockSMBus(bus_number)
            created.append(bus)
            return bus

        transport = TCA9548ABus(1, TCA9548A_ADDR, bus_factory=factory)
        manager = ActuatorManager(transport=transport)
        manager.start()
        manager.update_patterns(
            {zone_name: PATTERN_SLOW for zone_name in TCA_CHANNELS}
        )

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            started = [
                event
                for event in created[0].events
                if event[:1] == ("write",)
                and event[3:] == (REG_OVERRIDE_VAL, PATTERN_SLOW.level)
            ]
            if len(started) == 3:
                break
            time.sleep(0.005)
        self.assertEqual(len(started), 3)

        manager.stop()

        bus = created[0]
        for channel in TCA_CHANNELS.values():
            mask = 1 << channel
            self.assertEqual(
                bus.registers[(mask, DA7280_ADDR, REG_OVERRIDE_VAL)], 0x00
            )
            self.assertEqual(
                bus.registers.get((mask, DA7280_ADDR, REG_TOP_CTL1), 0) & 0x07,
                0x00,
            )
        self.assertEqual(
            bus.events[-2:], [("mux", TCA9548A_ADDR, 0x00), ("close",)]
        )
        self.assertTrue(bus.closed)


if __name__ == "__main__":
    unittest.main()
