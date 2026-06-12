"""Zone configuration for one irrigation zone (one sensor + one valve area).

A zone is the physical area controlled by a single soil moisture sensor
and a single irrigation valve. All water calculations are per zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from irrigation.config_loader import load_config

# Load zone defaults from config.yaml → zone section
_z = load_config()["zone"]


@dataclass
class ZoneConfig:
    """Configuration for one irrigation zone.

    Defaults come from config.yaml → zone section.
    Override by passing explicit values (e.g. for a different farm size).

    Args:
        area_m2: Zone area in square metres.
        irrigation_type: "drip" (90% efficient) or "sprinkler" (75% efficient).
        max_litres_per_event: Hard safety limit per irrigation event.
        emergency_min_litres: Minimum to apply when soil is near wilting point.
        root_depth_m: Crop root zone depth used for moisture calculations.
    """

    area_m2:              float = _z["area_m2"]
    irrigation_type:      str   = _z["irrigation_type"]
    max_litres_per_event: float = _z["max_litres_per_event"]
    emergency_min_litres: float = _z["emergency_min_litres"]
    root_depth_m:         float = _z["root_depth_m"]

    @property
    def efficiency(self) -> float:
        """Water delivery efficiency (0-1)."""
        return 0.90 if self.irrigation_type == "drip" else 0.75

    @property
    def moisture_per_litre(self) -> float:
        """Soil moisture % increase per gross litre of water applied.

        Derived from zone area and root depth:
            soil_volume = area_m2 × root_depth_m × 1000 L/m³
            moisture_per_litre = efficiency / soil_volume × 100

        Example (3m², drip, 0.3m root depth):
            soil_volume = 900 L
            moisture_per_litre = 0.9 / 900 × 100 = 0.10 %/L
        """
        soil_volume_litres = self.area_m2 * self.root_depth_m * 1000
        return (self.efficiency / soil_volume_litres) * 100
