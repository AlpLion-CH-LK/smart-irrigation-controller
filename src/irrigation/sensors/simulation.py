"""Simulated sensor implementations for testing and development.

These classes reproduce realistic sensor behaviour without requiring physical
hardware, making it possible to develop and test the RL agent on any machine.
"""

from __future__ import annotations

import math
import random
from datetime import datetime

from irrigation.sensors.base import SensorInterface, SensorReading


class SimulatedSoilMoistureSensor(SensorInterface):
    """Soil moisture sensor backed by a simple water-balance simulation.

    The model tracks a soil moisture percentage that:
    - Decreases over time due to evapotranspiration (ET).
    - Increases when ``irrigate()`` is called.
    - Increases when ``apply_rain()`` is called.

    Args:
        initial_moisture_pct: Starting soil moisture (0–100 %).
        et_rate_per_hour: Evapotranspiration rate in % per hour.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        initial_moisture_pct: float = 50.0,
        et_rate_per_hour: float = 2.0,
        seed: int | None = None,
    ) -> None:
        self._moisture = initial_moisture_pct
        self.et_rate_per_hour = et_rate_per_hour
        self._last_read: datetime = datetime.now()
        if seed is not None:
            random.seed(seed)

    @property
    def moisture_pct(self) -> float:
        return self._moisture

    def _apply_et(self) -> None:
        """Apply evapotranspiration since the last reading."""
        now = datetime.now()
        elapsed_hours = (now - self._last_read).total_seconds() / 3600.0
        self._moisture = max(0.0, self._moisture - self.et_rate_per_hour * elapsed_hours)
        self._last_read = now

    def irrigate(self, amount_pct: float) -> None:
        """Add soil moisture as if irrigation were applied.

        Args:
            amount_pct: Moisture increase in percentage points.
        """
        self._moisture = min(100.0, self._moisture + amount_pct)

    def apply_rain(self, rainfall_mm: float) -> None:
        """Increase moisture as if rain fell.

        A simple heuristic: 1 mm of rain ≈ 0.5 % soil moisture increase.

        Args:
            rainfall_mm: Rainfall in millimetres.
        """
        self._moisture = min(100.0, self._moisture + rainfall_mm * 0.5)

    def read(self) -> SensorReading:
        self._apply_et()
        noise = random.gauss(0, 0.3)
        noisy_moisture = max(0.0, min(100.0, self._moisture + noise))
        return SensorReading(
            timestamp=datetime.now(),
            soil_moisture_pct=round(noisy_moisture, 1),
        )


class SimulatedWeatherSensor(SensorInterface):
    """Weather sensor that generates realistic diurnal temperature cycles.

    Args:
        base_temp_celsius: Mean daily temperature.
        temp_amplitude: Half-range of the diurnal temperature swing.
        base_humidity_pct: Mean relative humidity.
        is_rainy_season: Whether to simulate more frequent rainfall.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        base_temp_celsius: float = 28.0,
        temp_amplitude: float = 5.0,
        base_humidity_pct: float = 70.0,
        is_rainy_season: bool = False,
        seed: int | None = None,
    ) -> None:
        self.base_temp = base_temp_celsius
        self.temp_amplitude = temp_amplitude
        self.base_humidity = base_humidity_pct
        self.is_rainy_season = is_rainy_season
        if seed is not None:
            random.seed(seed)
        self._rain_active: bool = False

    def _temperature_at(self, hour: float) -> float:
        """Return estimated temperature for the given hour of day (0–24)."""
        # Peak temperature around 14:00 local time.
        radians = 2 * math.pi * (hour - 14) / 24
        return self.base_temp + self.temp_amplitude * math.cos(radians)

    def set_rain(self, raining: bool) -> None:
        """Manually set rain state (useful in controlled test scenarios)."""
        self._rain_active = raining

    def read(self) -> SensorReading:
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

        temp = self._temperature_at(hour) + random.gauss(0, 0.5)
        humidity = self.base_humidity + random.gauss(0, 3.0)
        humidity = max(0.0, min(100.0, humidity))

        # Stochastic rain simulation.
        rain_prob = 0.15 if self.is_rainy_season else 0.03
        if self._rain_active or random.random() < rain_prob:
            is_raining = True
            rainfall_mm = random.uniform(0.5, 5.0)
        else:
            is_raining = False
            rainfall_mm = 0.0

        return SensorReading(
            timestamp=now,
            temperature_celsius=round(temp, 1),
            humidity_pct=round(humidity, 1),
            is_raining=is_raining,
            rainfall_mm=round(rainfall_mm, 2),
        )
