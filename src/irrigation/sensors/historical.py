"""Historical weather sensor using real NASA POWER Jaffna data.

Replaces SimulatedWeatherSensor + CombinedSimulatedSensor with a single
sensor backed by 20 years of real hourly weather records for Uduvil, Jaffna.

Each call to read():
  1. Samples a real weather record matching the current simulated hour
  2. Updates the soil sensor's ET rate using real ET₀ from that record
  3. Applies rain to the soil sensor if it was raining
  4. Reads the soil sensor (which applies ET and returns current moisture)
  5. Returns a complete SensorReading with all real weather values

ET₀ → soil drain rate conversion:
    Water loss = ET₀_mm × area_m² litres/hr
    Soil volume = area_m² × root_depth_m × 1000 litres
    Moisture %/hr = ET₀_mm / (root_depth_m × 10)
    (area cancels out — formula is independent of zone size)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from irrigation.sensors.base import SensorInterface, SensorReading
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor
from irrigation.weather.weather_data import WeatherDataLoader
from irrigation.zone_config import ZoneConfig


class HistoricalWeatherSensor(SensorInterface):
    """Real-data sensor for training — backed by NASA POWER hourly records.

    Combines soil moisture simulation with real Jaffna weather data.
    Replaces the CombinedSimulatedSensor used during development.

    Args:
        loader:       WeatherDataLoader with the hourly CSV loaded.
        soil_sensor:  SimulatedSoilMoistureSensor to update each step.
        zone:         ZoneConfig for ET₀ → moisture conversion.
        training_phase: 1 = dry months only, 2 = all months.
    """

    def __init__(
        self,
        loader: WeatherDataLoader,
        soil_sensor: SimulatedSoilMoistureSensor,
        zone: ZoneConfig,
        training_phase: int = 1,
    ) -> None:
        self.loader      = loader
        self.soil_sensor = soil_sensor
        self.zone        = zone
        self.training_phase = training_phase

        self._simulated_hour: float = 6.0   # start at 6am
        self._months = self._months_for_phase(training_phase)

    @staticmethod
    def _months_for_phase(phase: int) -> list[int] | None:
        """Return month filter for the curriculum phase."""
        if phase == 1:
            return [2, 3, 4]   # Feb, Mar, Apr — dry season
        return None             # all months

    def set_training_phase(self, phase: int) -> None:
        """Switch curriculum phase — updates which months are sampled."""
        self.training_phase = phase
        self._months = self._months_for_phase(phase)

    def reset(
        self,
        initial_moisture_pct: float = 50.0,
        initial_hour: float = 6.0,
    ) -> None:
        """Reset for a new episode."""
        self._simulated_hour = initial_hour
        self.soil_sensor.reset(initial_moisture_pct, initial_hour)

    def read(self) -> SensorReading:
        """Sample real weather, update ET rate, apply rain, read soil.

        Returns a SensorReading with:
            - soil_moisture_pct: from simulated soil (real ET applied)
            - temperature_celsius: from real NASA POWER record
            - humidity_pct: from real NASA POWER record
            - is_raining: from real NASA POWER record
            - rainfall_mm: from real NASA POWER record
        """
        hour = int(self._simulated_hour % 24)

        # Sample a real historical record for this hour
        record = self.loader.sample(hour=hour, months=self._months)

        # Convert ET₀ (mm/hr) → soil moisture drain rate (%/hr)
        # Formula: ET₀ / (root_depth_m × 10) — area_m² cancels out
        et_rate = record.et0_mm / (self.zone.root_depth_m * 10)
        self.soil_sensor.et_rate_per_hour = et_rate

        # Apply rain to soil before reading (rain arrives, then ET drains)
        if record.is_raining and record.rain_mm > 0:
            self.soil_sensor.apply_rain(record.rain_mm)

        # Read soil sensor — applies ET, returns current moisture
        soil = self.soil_sensor.read()

        self._simulated_hour += 1.0

        return SensorReading(
            timestamp=soil.timestamp,
            soil_moisture_pct=soil.soil_moisture_pct,
            temperature_celsius=record.temperature,
            humidity_pct=record.humidity_pct,
            is_raining=record.is_raining,
            rainfall_mm=record.rain_mm,
        )

    def close(self) -> None:
        pass
