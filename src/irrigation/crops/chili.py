"""Chili (Capsicum annuum) crop profile.

Based on FAO irrigation recommendations for hot peppers / chilli peppers.

References:
- FAO Irrigation and Drainage Paper 56 (Allen et al., 1998)
- "Chilli pepper production guide", Sri Lanka Dept. of Agriculture
"""

from __future__ import annotations

from irrigation.config_loader import load_config
from irrigation.crops.base import CropProfile, MoistureThresholds

# Stage-specific ETc water requirements — loaded from config.yaml → crop section
# Source: Sri Lanka Dept. of Agriculture + FAO Paper 56 ETc calculations.
# To change water requirements: edit config.yaml → crop → etc_litres_per_day
_etc = load_config()["crop"]["etc_litres_per_day"]
_STAGE_ETC_LITRES_PER_DAY = [
    _etc["stage_0"],   # Germination — 1.8 mm/day × 3m² zone
    _etc["stage_1"],   # Vegetative  — 3.2 mm/day × 3m² zone
    _etc["stage_2"],   # Flowering   — 4.7 mm/day × 3m² zone (PEAK)
    _etc["stage_3"],   # Fruit dev   — 4.5 mm/day × 3m² zone
    _etc["stage_4"],   # Maturity    — 3.8 mm/day × 3m² zone
]


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
            stress_threshold=40.0, # MDA level - 52.5
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
    
    def stage_for_day(self, day: int) -> int:
        """Map day since planting to chili growth stage (0-4)."""
        if day < 20:
            return 0  # Germination
        if day < 60:
            return 1  # Vegetative
        if day < 90:
            return 2  # Flowering
        if day < 120:
            return 3  # Fruit development
        return 4      # Maturity

    def moisture_thresholds_for_stage(self, stage: int) -> MoistureThresholds:

        """ Stage-specific thresholds based on FAO Paper 56. """

        if stage == 0:  # Germination (days 0-20)
            return MoistureThresholds(
                wilting_point=20.0,
                stress_threshold=45.0,
                optimal_min=55.0,
                optimal_max=70.0,
                field_capacity=85.0,
            )
        if stage == 1:  # Vegetative (days 20-60)
            return MoistureThresholds(
                wilting_point=20.0,
                stress_threshold=45.0,
                optimal_min=50.0,
                optimal_max=70.0,
                field_capacity=85.0,
            )
        if stage == 2:  # Flowering (days 60-90)
            return MoistureThresholds(
                wilting_point=25.0,
                stress_threshold=55.0,
                optimal_min=65.0,
                optimal_max=75.0,
                field_capacity=85.0,
            )
        if stage == 3:  # Fruit development (days 90-120)
            return MoistureThresholds(
                wilting_point=25.0,
                stress_threshold=55.0,
                optimal_min=60.0,
                optimal_max=75.0,
                field_capacity=85.0,
            )
        if stage == 4:  # Maturity (days 120-150)
            return MoistureThresholds(
                wilting_point=20.0,
                stress_threshold=35.0,
                optimal_min=35.0,
                optimal_max=50.0,
                field_capacity=85.0,
            )
        return self.moisture_thresholds

    def optimal_litres_per_day(self, stage: int) -> float:
        """Stage-specific water requirement for a standard 3m² Jaffna chilli bed."""
        return _STAGE_ETC_LITRES_PER_DAY[min(stage, 4)]
    