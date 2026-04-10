"""Base class and data types for crop profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MoistureThresholds:
    """Soil moisture thresholds for a crop at different growth stages.

    All values are volumetric soil moisture percentages (0–100).

    Attributes:
        wilting_point: Below this level the plant experiences severe stress.
        stress_threshold: Below this level mild water stress begins.
        optimal_min: Lower bound of the ideal moisture range.
        optimal_max: Upper bound of the ideal moisture range.
        field_capacity: Approximate field capacity; above this, drainage occurs.
    """

    wilting_point: float
    stress_threshold: float
    optimal_min: float
    optimal_max: float
    field_capacity: float


class CropProfile(ABC):
    """Abstract base class for crop-specific irrigation parameters.

    Subclass this to add a new crop type and register it in
    :mod:`irrigation.crops`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable crop name."""

    @property
    @abstractmethod
    def moisture_thresholds(self) -> MoistureThresholds:
        """Soil moisture thresholds for this crop."""

    @property
    @abstractmethod
    def peak_water_demand_mm_per_day(self) -> float:
        """Estimated daily water demand (in mm) during peak growth."""

    @property
    @abstractmethod
    def growing_season_days(self) -> int:
        """Approximate length of the growing season in days."""

    def stress_level(self, soil_moisture_pct: float) -> float:
        """Return a 0–1 water-stress indicator for the given soil moisture.

        - 0.0 → no stress (moisture is in the optimal range or above).
        - 1.0 → maximum stress (moisture is at or below the wilting point).

        Args:
            soil_moisture_pct: Current volumetric soil moisture (0–100).

        Returns:
            Float stress indicator in the range [0, 1].
        """
        t = self.moisture_thresholds
        if soil_moisture_pct >= t.optimal_min:
            return 0.0
        if soil_moisture_pct <= t.wilting_point:
            return 1.0
        # Linear interpolation between wilting point and optimal minimum.
        return (t.optimal_min - soil_moisture_pct) / (t.optimal_min - t.wilting_point)

    def needs_irrigation(self, soil_moisture_pct: float) -> bool:
        """Return True when irrigation should be considered.

        Args:
            soil_moisture_pct: Current volumetric soil moisture (0–100).
        """
        return soil_moisture_pct < self.moisture_thresholds.stress_threshold
