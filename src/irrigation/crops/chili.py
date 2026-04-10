"""Chili (Capsicum annuum) crop profile.

Based on FAO irrigation recommendations for hot peppers / chilli peppers.

References:
- FAO Irrigation and Drainage Paper 56 (Allen et al., 1998)
- "Chilli pepper production guide", Sri Lanka Dept. of Agriculture
"""

from __future__ import annotations

from irrigation.crops.base import CropProfile, MoistureThresholds


class ChiliProfile(CropProfile):
    """Crop profile for chili peppers (Capsicum annuum / Capsicum frutescens).

    Chili peppers are sensitive to both drought and waterlogging.  They
    perform best with consistent soil moisture in the 50–75 % range.
    Water stress during flowering and fruiting stages reduces yield
    significantly, making precise irrigation critical.

    Growth stages (approximate):
        - Germination / establishment: days 0–20
        - Vegetative growth:           days 20–60
        - Flowering:                   days 60–90
        - Fruit development:           days 90–120
        - Ripening:                    days 120–150
    """

    @property
    def name(self) -> str:
        return "Chili Pepper"

    @property
    def moisture_thresholds(self) -> MoistureThresholds:
        return MoistureThresholds(
            wilting_point=20.0,
            stress_threshold=40.0,
            optimal_min=50.0,
            optimal_max=75.0,
            field_capacity=85.0,
        )

    @property
    def peak_water_demand_mm_per_day(self) -> float:
        return 5.0  # Tropical conditions, peak flowering/fruiting stage

    @property
    def growing_season_days(self) -> int:
        return 150
