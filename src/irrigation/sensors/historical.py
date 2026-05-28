"""Historical weather sensor using real NASA POWER Jaffna data.

Phase 1 — Yala only (Jan/Feb/Mar start, fixed year, sequential months)
    Agent learns basic irrigation in dry, predictable conditions.

Phase 2 — Maha only (Aug/Sep start, fixed year, sequential months)
    Agent learns to handle Northeast monsoon and reduce irrigation.

Phase 3 — Both seasons (random Yala or Maha start, random year per month)
    Agent generalises to maximum weather variability.

Sequential month flow:
    Phase 1 & 2: episode fixes one year, months advance sequentially
                 Feb(2005) → Mar(2005) → Apr(2005) → May(2005) ...
    Phase 3:     months advance sequentially, year picked randomly each month
                 Feb(2005) → Mar(2011) → Apr(2003) → May(2018) ...

ET₀ → soil drain rate conversion:
    Moisture %/hr = ET₀_mm / (root_depth_m × 10)
"""

from __future__ import annotations

import random

from irrigation.sensors.base import SensorInterface, SensorReading
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor
from irrigation.weather.weather_data import WeatherDataLoader
from irrigation.zone_config import ZoneConfig

# Approximate days per month — used to know when to advance to next month
_MONTH_DAYS = {
    1: 31, 2: 28, 3: 31, 4: 30,  5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

# Yala planting window: Jan, Feb, Mar (dry season)
_YALA_START_MONTHS = [1, 2, 3]

# Maha planting window: Aug, Sep (Northeast monsoon season)
_MAHA_START_MONTHS = [8, 9]


class HistoricalWeatherSensor(SensorInterface):
    """Season-aware sensor backed by 20 years of NASA POWER hourly records.

    Args:
        loader:          WeatherDataLoader with the hourly CSV loaded.
        soil_sensor:     SimulatedSoilMoistureSensor to update each step.
        zone:            ZoneConfig for ET₀ → moisture conversion.
        training_phase:  1 = Yala only, 2 = Maha only, 3 = both seasons.
    """

    def __init__(
        self,
        loader: WeatherDataLoader,
        soil_sensor: SimulatedSoilMoistureSensor,
        zone: ZoneConfig,
        training_phase: int = 1,
    ) -> None:
        self.loader         = loader
        self.soil_sensor    = soil_sensor
        self.zone           = zone
        self.training_phase = training_phase

        self._simulated_hour: float = 6.0
        self._current_month: int    = 2
        self._current_year: int     = 2010
        self._episode_year: int     = 2010
        self._hours_in_month: int   = 0

        self._rng = random.Random()

    def set_training_phase(self, phase: int) -> None:
        """Switch curriculum phase — takes effect on next reset()."""
        self.training_phase = phase

    def _pick_start(self) -> tuple[int, int]:
        """Pick (start_month, year) based on current training phase."""
        if self.training_phase == 1:
            month = self._rng.choice(_YALA_START_MONTHS)
        elif self.training_phase == 2:
            month = self._rng.choice(_MAHA_START_MONTHS)
        else:
            season = self._rng.choice(["yala", "maha"])
            month  = self._rng.choice(
                _YALA_START_MONTHS if season == "yala" else _MAHA_START_MONTHS
            )
        year = self._rng.choice(self.loader.get_years_for_month(month))
        return month, year

    def reset(
        self,
        initial_moisture_pct: float = 50.0,
        initial_hour: float = 6.0,
    ) -> None:
        """Reset for a new episode — picks new season start and year."""
        self._simulated_hour  = initial_hour
        self._hours_in_month  = 0

        self._current_month, self._episode_year = self._pick_start()
        self._current_year = self._episode_year

        self.soil_sensor.reset(initial_moisture_pct, initial_hour)

    def _advance_month_if_needed(self) -> None:
        """Move to next sequential month when current month's hours are exhausted."""
        limit = _MONTH_DAYS.get(self._current_month, 30) * 24
        if self._hours_in_month >= limit:
            self._current_month  = (self._current_month % 12) + 1
            self._hours_in_month = 0

            if self.training_phase == 3:
                # Phase 3: new random year for each new month
                self._current_year = self._rng.choice(
                    self.loader.get_years_for_month(self._current_month)
                )
            # Phase 1 & 2: keep the fixed episode year throughout

    def read(self) -> SensorReading:
        """Sample real weather for current sequential month, update soil, return reading."""
        hour = int(self._simulated_hour % 24)

        self._advance_month_if_needed()

        record = self.loader.sample_for_year(
            hour=hour,
            month=self._current_month,
            year=self._current_year,
        )

        # Convert ET₀ (mm/hr) → soil moisture drain rate (%/hr)
        et_rate = record.et0_mm / (self.zone.root_depth_m * 10)
        self.soil_sensor.et_rate_per_hour = et_rate

        if record.is_raining and record.rain_mm > 0:
            self.soil_sensor.apply_rain(record.rain_mm)

        soil = self.soil_sensor.read()

        self._simulated_hour  += 1.0
        self._hours_in_month  += 1

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
