"""Soil moisture sensor implementation for Raspberry Pi.

Supports both resistive (analog via ADC) and capacitive soil moisture sensors.
Requires the RPi.GPIO and optionally adafruit-blinka packages on Raspberry Pi hardware.
"""

from __future__ import annotations

import logging
from datetime import datetime

from irrigation.sensors.base import SensorInterface, SensorReading

logger = logging.getLogger(__name__)

# Resistive sensor calibration defaults (ADC counts).
_ADC_DRY = 3.3   # Voltage equivalent when bone-dry (placeholder for raw ADC value normalisation)
_ADC_WET = 1.2   # Voltage equivalent when fully saturated


class SoilMoistureSensor(SensorInterface):
    """Raspberry Pi soil moisture sensor (resistive or capacitive).

    On real hardware this class drives a GPIO digital output pin that powers
    the sensor and reads the analog voltage via an ADC (e.g. ADS1115).  The
    ADC reading is converted to a 0–100 % volumetric moisture estimate using
    a two-point dry/wet calibration.

    When the ``RPi.GPIO`` / ``busio`` packages are not available the sensor
    falls back to returning a fixed placeholder value so that the rest of the
    stack can be exercised on a development machine.

    Args:
        pin: GPIO BCM pin number for a digital moisture sensor output.
        adc_channel: ADC channel number (for analog sensors).
        dry_voltage: ADC voltage reading in dry conditions.
        wet_voltage: ADC voltage reading in fully saturated conditions.
    """

    def __init__(
        self,
        pin: int = 4,
        adc_channel: int = 0,
        dry_voltage: float = _ADC_DRY,
        wet_voltage: float = _ADC_WET,
    ) -> None:
        self.pin = pin
        self.adc_channel = adc_channel
        self.dry_voltage = dry_voltage
        self.wet_voltage = wet_voltage
        self._gpio: object | None = None
        self._setup_hardware()

    def _setup_hardware(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN)
            self._gpio = GPIO
            logger.info("Soil moisture sensor initialised on GPIO pin %d", self.pin)
        except (ImportError, RuntimeError):
            logger.warning(
                "RPi.GPIO not available – soil moisture sensor running in stub mode."
            )

    def _read_voltage(self) -> float | None:
        """Read voltage from the ADC.  Returns None when hardware is unavailable."""
        if self._gpio is None:
            return None
        try:
            import busio  # type: ignore[import]
            import board  # type: ignore[import]
            import adafruit_ads1x15.ads1115 as ADS  # type: ignore[import]
            from adafruit_ads1x15.analog_in import AnalogIn  # type: ignore[import]

            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)
            channel = AnalogIn(ads, self.adc_channel)
            return float(channel.voltage)
        except (ImportError, RuntimeError) as exc:
            logger.debug("ADC read failed: %s", exc)
            return None

    def _voltage_to_moisture_pct(self, voltage: float) -> float:
        """Convert ADC voltage to a 0–100 % moisture estimate."""
        # Clamp the voltage to the calibration range.
        voltage = max(self.wet_voltage, min(self.dry_voltage, voltage))
        pct = (self.dry_voltage - voltage) / (self.dry_voltage - self.wet_voltage) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

    def read(self) -> SensorReading:
        """Return the current soil moisture reading."""
        voltage = self._read_voltage()
        if voltage is not None:
            moisture = self._voltage_to_moisture_pct(voltage)
        else:
            moisture = 0.0  # Hardware unavailable; downstream code should use simulation.

        return SensorReading(
            timestamp=datetime.now(),
            soil_moisture_pct=moisture,
        )

    def close(self) -> None:
        if self._gpio is not None:
            try:
                self._gpio.cleanup(self.pin)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._gpio = None
