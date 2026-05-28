"""Dynamic plant growth stage tracker for chilli irrigation RL.

Replaces the fixed calendar-based stage progression with a health-conditioned
model: a stage only advances when the plant has accumulated enough healthy
moisture hours — not just because a certain number of days have passed.

Stage advance requires BOTH:
    1. Minimum hours elapsed in this stage
    2. Health ratio >= required threshold  (OR max hours forced advance)

If the plant was stressed, stage is delayed up to max hours. After max hours,
the stage is forced to advance regardless (plant is stunted but alive).
"""

from __future__ import annotations

from dataclasses import dataclass
from irrigation.config_loader import load_config
from irrigation.crops.base import CropProfile

# Load stage timing parameters from config.yaml → stages section
_cfg = load_config()
_s   = _cfg["stages"]

_STAGE_KEYS = {
    0: "germination",
    1: "vegetative",
    2: "flowering",
    3: "fruit_development",
    4: "maturity",
}

# ---------------------------------------------------------------------------
# Stage timing and health thresholds — loaded from config.yaml
# Edit min_days, max_days, health_required there — reflects here automatically.
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    min_hours: int        # minimum hours before stage can advance
    max_hours: int        # maximum hours — forced advance after this
    health_required: float  # health ratio needed for normal advance (0.0-1.0)


# Built from config.yaml stages section — days converted to hours here
STAGE_CONFIGS: dict[int, StageConfig] = {
    idx: StageConfig(
        min_hours=_s[key]["min_days"] * 24,       # convert days → hours
        max_hours=_s[key]["max_days"] * 24,
        health_required=_s[key]["health_required"],
    )
    for idx, key in _STAGE_KEYS.items()
}

# Human-readable stage names for reporting
STAGE_NAMES = {
    0: "Germination",
    1: "Vegetative",
    2: "Flowering",
    3: "Fruit Development",
    4: "Maturity",
}


class DynamicStageTracker:
    """Tracks plant growth stage based on accumulated health, not calendar days.

    Each hourly step contributes to a running health ratio for the current stage:
        Moisture in optimal range  → +1.0 contribution
        Moisture in stress range   → +0.5 contribution (growing slowly)
        Moisture below stress      → +0.0 contribution (no growth)

    The stage advances when health_ratio >= required AND min_hours elapsed.
    If health never reaches the threshold, stage is forced at max_hours
    (plant is stunted but still alive — death is handled by PlantVitalityTracker).

    Args:
        crop: Crop profile providing stage-specific moisture thresholds.
    """

    def __init__(self, crop: CropProfile) -> None:
        self.crop = crop
        self.current_stage: int = 0
        self._hours_in_stage: int = 0
        self._health_sum: float = 0.0
        self._stage_advanced: bool = False
        self._actual_stage_hours: dict[int, int] = {}   # stage → hours taken

    def reset(self) -> None:
        """Reset for a new episode."""
        self.current_stage = 0
        self._hours_in_stage = 0
        self._health_sum = 0.0
        self._stage_advanced = False
        self._actual_stage_hours = {}

    def update(self, moisture_pct: float) -> bool:
        """Update tracker for one hourly step.

        Args:
            moisture_pct: Current soil moisture percentage.

        Returns:
            True if the stage advanced this step, False otherwise.
        """
        t = self.crop.moisture_thresholds_for_stage(self.current_stage)
        self._hours_in_stage += 1
        self._stage_advanced = False

        # Accumulate health contribution for this hour
        if t.optimal_min <= moisture_pct <= t.optimal_max:
            self._health_sum += 1.0          # full growth
        elif moisture_pct >= t.stress_threshold:
            self._health_sum += 0.5          # slow growth
        # below stress_threshold → 0.0 contribution (no growth)

        # Check if stage should advance
        if self.current_stage < 4 and self._should_advance():
            self._advance()
            return True

        return False

    def _should_advance(self) -> bool:
        """Return True if conditions to advance to next stage are met."""
        cfg = STAGE_CONFIGS[self.current_stage]

        # Must spend minimum hours in stage regardless of health
        if self._hours_in_stage < cfg.min_hours:
            return False

        # Normal advance: health ratio meets threshold
        health_ratio = self._health_sum / self._hours_in_stage
        if health_ratio >= cfg.health_required:
            return True

        # Forced advance: exceeded maximum hours (plant is stunted)
        if self._hours_in_stage >= cfg.max_hours:
            return True

        return False

    def _advance(self) -> None:
        """Move to the next growth stage."""
        self._actual_stage_hours[self.current_stage] = self._hours_in_stage
        self.current_stage += 1
        self._hours_in_stage = 0
        self._health_sum = 0.0
        self._stage_advanced = True

    @property
    def health_ratio(self) -> float:
        """Health ratio for the current stage so far (0.0 → 1.0)."""
        if self._hours_in_stage == 0:
            return 1.0
        return round(self._health_sum / self._hours_in_stage, 4)

    @property
    def hours_in_stage(self) -> int:
        """Hours spent in the current stage."""
        return self._hours_in_stage

    @property
    def days_in_stage(self) -> float:
        """Days spent in the current stage."""
        return round(self._hours_in_stage / 24, 1)

    @property
    def stage_name(self) -> str:
        """Human-readable name of the current stage."""
        return STAGE_NAMES[self.current_stage]

    def is_delayed(self) -> bool:
        """Return True if current stage is taking longer than the ideal minimum."""
        cfg = STAGE_CONFIGS[self.current_stage]
        return self._hours_in_stage > cfg.min_hours

    def summary(self) -> dict:
        """Return a summary dict for episode reporting."""
        completed = dict(self._actual_stage_hours)
        completed[self.current_stage] = self._hours_in_stage  # include current

        stage_days = {
            STAGE_NAMES[s]: round(h / 24, 1)
            for s, h in completed.items()
        }
        ideal_days = {
            STAGE_NAMES[s]: round(cfg.min_hours / 24, 1)
            for s, cfg in STAGE_CONFIGS.items()
        }
        delay_days = {
            name: round(stage_days.get(name, 0) - ideal_days.get(name, 0), 1)
            for name in STAGE_NAMES.values()
        }

        return {
            "final_dynamic_stage":  self.current_stage,
            "stage_name":           self.stage_name,
            "stage_days_taken":     stage_days,
            "stage_delay_days":     delay_days,
            "current_health_ratio": self.health_ratio,
        }
