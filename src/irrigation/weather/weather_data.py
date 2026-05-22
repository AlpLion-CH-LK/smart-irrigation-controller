"""Weather data loader for Uduvil, Jaffna historical hourly records.

Loads the NASA POWER hourly CSV and provides fast lookup by hour of day.
For curriculum learning, records can be filtered by month group:
    Phase 1 — dry months only (Feb, Mar, Apr)
    Phase 2 — all months (full 20 years)

Usage:
    loader = WeatherDataLoader()
    record = loader.sample(hour=14)                     # Phase 2 (all months)
    record = loader.sample(hour=14, months=[2, 3, 4])   # Phase 1 (dry only)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Default path to the downloaded NASA POWER hourly CSV
_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.csv"
)

# Jaffna season month groups for curriculum learning
FAVORABLE_MONTHS = [2, 3, 4]       # Feb, Mar, Apr — dry season
ALL_MONTHS       = list(range(1, 13))


class WeatherRecord(NamedTuple):
    """One hourly weather record from the NASA POWER dataset."""
    temperature:   float   # °C
    humidity_pct:  float   # %
    rain_mm:       float   # mm/hr
    is_raining:    bool    # rain_mm > 0.1
    wind_speed_ms: float   # m/s
    solar_rad_MJ:  float   # MJ/m²/hr
    et0_mm:        float   # mm/hr (FAO-56 Penman-Monteith)


class WeatherDataLoader:
    """Loads NASA POWER hourly CSV and provides random sampling by hour.

    Records are pre-indexed by (hour, month) for fast lookup during training.

    Args:
        csv_path: Path to the hourly CSV file. Defaults to the project data folder.
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV

        if not path.exists():
            raise FileNotFoundError(
                f"Hourly weather CSV not found at {path}.\n"
                f"Run: python scripts/fetch_weather_hourly.py"
            )

        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"]  = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month

        # Pre-index by (hour, month) → list of WeatherRecord tuples
        self._index: dict[tuple[int, int], list[WeatherRecord]] = {}
        for (hour, month), group in df.groupby(["hour", "month"]):
            records = [
                WeatherRecord(
                    temperature   = float(row.temperature),
                    humidity_pct  = float(row.humidity_pct),
                    rain_mm       = float(row.rain_mm),
                    is_raining    = bool(row.is_raining),
                    wind_speed_ms = float(row.wind_speed_ms),
                    solar_rad_MJ  = float(row.solar_rad_MJ),
                    et0_mm        = float(row.et0_mm),
                )
                for row in group.itertuples()
            ]
            self._index[(int(hour), int(month))] = records

        self._rng = __import__("random").Random()
        print(
            f"WeatherDataLoader: loaded {len(df):,} records "
            f"({df['datetime'].min().date()} → {df['datetime'].max().date()})"
        )

    def sample(
        self,
        hour: int,
        months: list[int] | None = None,
    ) -> WeatherRecord:
        """Return a random real weather record for the given hour of day.

        Args:
            hour:   Simulated hour of day (0–23).
            months: List of calendar months to sample from.
                    None = all months (Phase 2).
                    [2,3,4] = Feb, Mar, Apr only (Phase 1 dry season).

        Returns:
            A WeatherRecord with real Jaffna weather data.
        """
        month_list = months if months else ALL_MONTHS
        hour = int(hour) % 24

        # Collect all records for this hour across allowed months
        candidates: list[WeatherRecord] = []
        for m in month_list:
            candidates.extend(self._index.get((hour, m), []))

        if not candidates:
            # Fallback: use any month for this hour
            for m in ALL_MONTHS:
                candidates.extend(self._index.get((hour, m), []))

        return self._rng.choice(candidates)
