"""Base sensor interfaces and data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SensorReading:
    """A single reading from the sensor suite.

    Attributes:
        timestamp: When the reading was taken.
        soil_moisture_pct: Volumetric soil moisture as a percentage (0–100).
        temperature_celsius: Air temperature in degrees Celsius.
        humidity_pct: Relative humidity as a percentage (0–100).
        is_raining: Whether rain is currently detected.
        rainfall_mm: Recent rainfall in millimetres (if measurable).
    """

    timestamp: datetime = field(default_factory=datetime.now)
    soil_moisture_pct: float = 0.0
    temperature_celsius: float = 20.0
    humidity_pct: float = 50.0
    is_raining: bool = False
    rainfall_mm: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.soil_moisture_pct <= 100.0):
            raise ValueError(
                f"soil_moisture_pct must be between 0 and 100, got {self.soil_moisture_pct}"
            )
        if not (0.0 <= self.humidity_pct <= 100.0):
            raise ValueError(
                f"humidity_pct must be between 0 and 100, got {self.humidity_pct}"
            )
        if self.rainfall_mm < 0.0:
            raise ValueError(f"rainfall_mm must be >= 0, got {self.rainfall_mm}")


class SensorInterface(ABC):
    """Abstract base class for all sensor implementations."""

    @abstractmethod
    def read(self) -> SensorReading:
        """Take a sensor reading and return the result."""

    def close(self) -> None:
        """Release any hardware resources held by the sensor."""
