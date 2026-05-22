"""Plant vitality tracker for the irrigation RL agent.

Tracks plant survival each step using stage-specific death thresholds.
If the plant stays below its critical moisture threshold for too many
consecutive hours, vitality drops to zero and the plant dies.

Key agronomic facts modelled:
  - Seedlings (Stage 0) die within 2 dry hours — no root system yet
  - Stage 2 (Flowering) death threshold is STRESS THRESHOLD (55%), not
    wilting point — flower drop at that level = total crop failure
  - Waterlogging above field capacity causes root suffocation (slower death)
  - Earlier death = larger penalty (wasted entire season investment)
"""

from __future__ import annotations

from dataclasses import dataclass

from irrigation.crops.base import CropProfile


# ---------------------------------------------------------------------------
# Stage-specific vitality configuration
# ---------------------------------------------------------------------------

@dataclass
class VitalityConfig:
    dry_threshold_type: str   # "wilting_point" or "stress_threshold"
    dry_hours_limit: int      # consecutive hours below dry threshold → dead
    wet_hours_limit: int      # consecutive hours above field capacity → dead
    vitality_drain: float     # vitality lost per dry hour (0.0–1.0)


VITALITY_CONFIGS: dict[int, VitalityConfig] = {
    0: VitalityConfig("wilting_point",    2,  4, 0.20),  # Germination — most fragile
    1: VitalityConfig("wilting_point",    6,  8, 0.12),  # Vegetative
    2: VitalityConfig("stress_threshold", 12, 10, 0.08), # Flowering — flower drop = death
    3: VitalityConfig("wilting_point",    12, 12, 0.06), # Fruit development
    4: VitalityConfig("wilting_point",    24, 24, 0.03), # Maturity — most resilient
}

# Terminal death penalty per stage — earlier death = larger penalty
DEATH_PENALTY: dict[int, float] = {
    0: -100.0,  # Germination — catastrophic, entire season wasted
    1: -80.0,   # Vegetative
    2: -60.0,   # Flowering — total crop failure
    3: -40.0,   # Fruit development
    4: -20.0,   # Maturity — some crop may be salvageable
}


class PlantVitalityTracker:
    """Tracks plant survival using stage-specific critical moisture thresholds.

    Vitality (0.0 → 1.0):
        1.0 = perfectly healthy
        0.5 = stressed but alive
        0.0 = dead → episode terminates

    Args:
        crop: Crop profile providing stage-specific moisture thresholds.
    """

    def __init__(self, crop: CropProfile) -> None:
        self.crop = crop
        self.vitality: float = 1.0
        self.is_dead: bool = False
        self._consecutive_dry: int = 0
        self._consecutive_wet: int = 0
        self._death_cause: str = ""
        self._death_stage: int = -1

    def reset(self) -> None:
        """Reset for a new episode."""
        self.vitality = 1.0
        self.is_dead = False
        self._consecutive_dry = 0
        self._consecutive_wet = 0
        self._death_cause = ""
        self._death_stage = -1

    def update(self, moisture_pct: float, stage: int) -> None:
        """Update vitality for one hourly step.

        Args:
            moisture_pct: Current soil moisture percentage.
            stage: Current dynamic growth stage (0-4).
        """
        if self.is_dead:
            return

        t   = self.crop.moisture_thresholds_for_stage(stage)
        cfg = VITALITY_CONFIGS[stage]

        # Determine the dry threshold for this stage
        dry_threshold = (
            t.stress_threshold
            if cfg.dry_threshold_type == "stress_threshold"
            else t.wilting_point
        )

        if moisture_pct < dry_threshold:
            # Drought stress — drain vitality
            self._consecutive_dry += 1
            self._consecutive_wet = 0
            self.vitality = max(0.0, self.vitality - cfg.vitality_drain)

            if self._consecutive_dry >= cfg.dry_hours_limit or self.vitality <= 0.0:
                self.vitality = 0.0
                self.is_dead = True
                self._death_cause = "drought"
                self._death_stage = stage

        elif moisture_pct > t.field_capacity:
            # Waterlogging — drain vitality at half rate (slower death)
            self._consecutive_wet += 1
            self._consecutive_dry = 0
            self.vitality = max(0.0, self.vitality - cfg.vitality_drain * 0.5)

            if self._consecutive_wet >= cfg.wet_hours_limit or self.vitality <= 0.0:
                self.vitality = 0.0
                self.is_dead = True
                self._death_cause = "waterlogging"
                self._death_stage = stage

        else:
            # Healthy range — reset counters and slowly recover
            self._consecutive_dry = 0
            self._consecutive_wet = 0
            self.vitality = min(1.0, self.vitality + 0.01)

    @property
    def death_penalty(self) -> float:
        """Return the terminal death penalty for the stage the plant died in."""
        if not self.is_dead:
            return 0.0
        return DEATH_PENALTY.get(self._death_stage, -50.0)

    def death_info(self) -> dict:
        """Return death event details for the info dict."""
        return {
            "plant_dead":   self.is_dead,
            "death_cause":  self._death_cause,
            "death_stage":  self._death_stage,
            "vitality":     round(self.vitality, 4),
        }
