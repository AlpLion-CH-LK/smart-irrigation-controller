"""Raspberry Pi GPIO valve/pump actuator."""

from __future__ import annotations

import logging
import time

from irrigation.actuators.base import ActuatorInterface, IrrigationCommand

logger = logging.getLogger(__name__)


class ValveActuator(ActuatorInterface):
    """Controls an irrigation valve (and optional pump) via Raspberry Pi GPIO.

    The valve is opened by pulling the GPIO pin HIGH. Duration is calculated
    from the requested water volume and the valve's flow rate.

    Args:
        valve_pin: GPIO BCM pin number for the solenoid valve relay.
        pump_pin: Optional GPIO BCM pin number for the water pump relay.
        flow_rate_lpm: Valve flow rate in litres per minute (default 2.0).
    """

    def __init__(
        self,
        valve_pin: int = 22,
        pump_pin: int | None = None,
        flow_rate_lpm: float = 2.0,
    ) -> None:
        self.valve_pin = valve_pin
        self.pump_pin = pump_pin
        self.flow_rate_lpm = flow_rate_lpm
        self._active: bool = False
        self._gpio: object | None = None
        self._setup_hardware()

    def _setup_hardware(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.valve_pin, GPIO.OUT, initial=GPIO.LOW)
            if self.pump_pin is not None:
                GPIO.setup(self.pump_pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            logger.info("Valve actuator initialised on GPIO pin %d", self.valve_pin)
        except (ImportError, RuntimeError):
            logger.warning("RPi.GPIO not available – valve actuator running in stub mode.")

    def _set_valve(self, open: bool) -> None:
        if self._gpio is not None:
            try:
                value = self._gpio.HIGH if open else self._gpio.LOW  # type: ignore[attr-defined]
                self._gpio.output(self.valve_pin, value)  # type: ignore[attr-defined]
                if self.pump_pin is not None:
                    self._gpio.output(self.pump_pin, value)  # type: ignore[attr-defined]
            except RuntimeError as exc:
                logger.error("Failed to set valve state: %s", exc)

    def execute(self, command: IrrigationCommand) -> None:
        if command.water_litres <= 0.0:
            return

        duration = (command.water_litres / self.flow_rate_lpm) * 60.0
        logger.info(
            "Starting irrigation: %.1f L over %.0f s (%.1f L/min)",
            command.water_litres,
            duration,
            self.flow_rate_lpm,
        )
        self._active = True
        self._set_valve(open=True)
        try:
            time.sleep(duration)
        finally:
            self._set_valve(open=False)
            self._active = False
            logger.info("Irrigation complete.")

    def stop(self) -> None:
        self._set_valve(open=False)
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def close(self) -> None:
        self.stop()
        if self._gpio is not None:
            try:
                self._gpio.cleanup()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._gpio = None
