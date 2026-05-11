"""Weather sensor implementation for Raspberry Pi.

Wraps a DHT11/DHT22 temperature-and-humidity sensor and an optional digital
rain-detection module.
"""

from __future__ import annotations

import logging
from datetime import datetime

from irrigation.sensors.base import SensorInterface, SensorReading

logger = logging.getLogger(__name__)


class WeatherSensor(SensorInterface):
    """DHT temperature/humidity + rain detector sensor.

    Args:
        dht_pin: GPIO BCM pin connected to the DHT data line.
        rain_pin: GPIO BCM pin connected to the rain sensor's digital output.
        dht_model: ``"DHT11"`` or ``"DHT22"`` (default: ``"DHT22"``).
    """

    def __init__(
        self,
        dht_pin: int = 17,
        rain_pin: int = 27,
        dht_model: str = "DHT22",
    ) -> None:
        self.dht_pin = dht_pin
        self.rain_pin = rain_pin
        self.dht_model = dht_model
        self._gpio: object | None = None
        self._dht: object | None = None
        self._setup_hardware()

    def _setup_hardware(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.rain_pin, GPIO.IN)
            self._gpio = GPIO
        except (ImportError, RuntimeError):
            logger.warning("RPi.GPIO not available – weather sensor running in stub mode.")
            return

        try:
            import adafruit_dht  # type: ignore[import]
            import board  # type: ignore[import]

            pin = getattr(board, f"D{self.dht_pin}")
            if self.dht_model == "DHT11":
                self._dht = adafruit_dht.DHT11(pin)
            else:
                self._dht = adafruit_dht.DHT22(pin)
            logger.info("DHT sensor (%s) initialised on GPIO pin %d", self.dht_model, self.dht_pin)
        except (ImportError, RuntimeError) as exc:
            logger.warning("adafruit_dht not available: %s", exc)

    def read(self) -> SensorReading:
        """Return the latest weather sensor reading."""
        temperature: float = 20.0
        humidity: float = 50.0
        is_raining: bool = False

        if self._dht is not None:
            try:
                temperature = float(self._dht.temperature)  # type: ignore[attr-defined]
                humidity = float(self._dht.humidity)  # type: ignore[attr-defined]
            except RuntimeError as exc:
                # DHT sensors occasionally fail to respond; just keep previous values.
                logger.debug("DHT read error (non-fatal): %s", exc)

        if self._gpio is not None:
            try:
                # Most rain detector modules output LOW when rain is detected.
                is_raining = not bool(
                    self._gpio.input(self.rain_pin)  # type: ignore[attr-defined]
                )
            except RuntimeError as exc:
                logger.debug("Rain sensor read error: %s", exc)

        return SensorReading(
            timestamp=datetime.now(),
            temperature_celsius=temperature,
            humidity_pct=max(0.0, min(100.0, humidity)),
            is_raining=is_raining,
        )

    def close(self) -> None:
        if self._dht is not None:
            try:
                self._dht.exit()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._dht = None
        if self._gpio is not None:
            try:
                self._gpio.cleanup(self.rain_pin)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._gpio = None
