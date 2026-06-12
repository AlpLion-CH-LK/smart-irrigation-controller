"""Plant health tracker for the irrigation RL agent.

Tracks cumulative plant health throughout one growing season episode.
Health is based on how well soil moisture was maintained within the
stage-specific optimal range at each hourly step.

This is a monitoring class — it does not modify rewards directly.
It adds health_score to the observation so the agent can see the
plant's condition and self-correct before it is too late.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irrigation.crops.base import CropProfile


@dataclass
class StageHealthRecord:
    """Health record for one growth stage."""
    stage: int
    optimal_hours: int = 0      # hours moisture in optimal range
    stress_hours: int = 0       # hours below stress threshold
    critical_hours: int = 0     # hours below wilting point
    waterlog_hours: int = 0     # hours above field capacity
    total_hours: int = 0        # total hours spent in this stage

    @property
    def health_score(self) -> float:
        """Health score for this stage (0.0 → 1.0)."""
        if self.total_hours == 0:
            return 1.0
        positive = self.optimal_hours
        negative = self.critical_hours * 2 + self.waterlog_hours
        raw = (positive - negative) / self.total_hours
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))


class PlantHealthTracker:
    """Tracks plant health across one full growing season episode.

    Updated every step via update(). Reset at episode start via reset().
    Provides health_score (0.0-1.0) for inclusion in the observation vector.

    Args:
        crop: Crop profile used to get stage-specific moisture thresholds.
    """

    def __init__(self, crop: CropProfile) -> None:
        self.crop = crop
        self._stage_records: list[StageHealthRecord] = []
        self._current_record: StageHealthRecord = StageHealthRecord(stage=0)
        self._total_steps: int = 0
        self._total_optimal: int = 0
        self._total_critical: int = 0
        self._total_waterlog: int = 0
        self._total_stress: int = 0
        self._total_water_litres: float = 0.0
        self._last_stage: int = 0

    def reset(self) -> None:
        """Reset all tracking for a new episode."""
        self._stage_records = []
        self._current_record = StageHealthRecord(stage=0)
        self._total_steps = 0
        self._total_optimal = 0
        self._total_critical = 0
        self._total_waterlog = 0
        self._total_stress = 0
        self._total_water_litres = 0.0
        self._last_stage = 0

    def update(
        self,
        moisture_pct: float,
        stage: int,
        water_litres: float,
    ) -> None:
        """Update health tracking for one step.

        Args:
            moisture_pct: Current soil moisture percentage.
            stage: Current growth stage (0-4).
            water_litres: Water applied this step.
        """
        # If stage advanced, save the old record and start a new one.
        if stage != self._last_stage:
            self._stage_records.append(self._current_record)
            self._current_record = StageHealthRecord(stage=stage)
            self._last_stage = stage

        t = self.crop.moisture_thresholds_for_stage(stage)
        self._current_record.total_hours += 1
        self._total_steps += 1
        self._total_water_litres += water_litres

        # Classify moisture condition this step
        if t.optimal_min <= moisture_pct <= t.optimal_max:
            self._current_record.optimal_hours += 1
            self._total_optimal += 1

        elif moisture_pct < t.wilting_point:
            self._current_record.critical_hours += 1
            self._total_critical += 1

        elif moisture_pct > t.field_capacity:
            self._current_record.waterlog_hours += 1
            self._total_waterlog += 1

        elif moisture_pct < t.stress_threshold:
            self._current_record.stress_hours += 1
            self._total_stress += 1

    @property
    def health_score(self) -> float:
        """Overall plant health score for this episode so far (0.0 → 1.0).

        1.0 = moisture in optimal range every step
        0.5 = mix of optimal and stress
        0.0 = critical stress or waterlogging throughout
        """
        if self._total_steps == 0:
            return 1.0

        positive = self._total_optimal
        negative = self._total_critical * 2 + self._total_waterlog
        raw = (positive - negative) / self._total_steps
        return round(max(0.0, min(1.0, (raw + 1.0) / 2.0)), 4)

    @property
    def stress_ratio(self) -> float:
        """Fraction of steps the plant was under stress (0.0 → 1.0)."""
        if self._total_steps == 0:
            return 0.0
        return round(self._total_stress / self._total_steps, 4)

    @property
    def critical_ratio(self) -> float:
        """Fraction of steps the plant was in critical moisture (0.0 → 1.0)."""
        if self._total_steps == 0:
            return 0.0
        return round(self._total_critical / self._total_steps, 4)

    def summary(self) -> dict:
        """Return a full episode health summary dict.

        Used in the info dict returned by gym_env.step() at episode end.
        """
        all_records = self._stage_records + [self._current_record]
        stage_scores = {
            f"stage_{r.stage}_health": round(r.health_score, 4)
            for r in all_records
            if r.total_hours > 0
        }

        return {
            "overall_health_score": self.health_score,
            "optimal_hours":        self._total_optimal,
            "stress_hours":         self._total_stress,
            "critical_hours":       self._total_critical,
            "waterlog_hours":       self._total_waterlog,
            "total_water_litres":   round(self._total_water_litres, 2),
            **stage_scores,
        }
