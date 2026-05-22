"""Gymnasium wrapper around IrrigationEnvironment for PPO training.

One episode = one full growing season. The simulated clock advances by
step_hours on every step, independent of real wall-clock time.

Step size: 1 hour (matches NASA POWER hourly data exactly)
Episode length: 150 days × 24 steps/day = 3,600 steps per episode

Curriculum Learning Phases:
    Phase 1 — Favorable months only (Feb–Apr, dry season Jaffna)
              Hot, dry, predictable. Agent learns basic irrigation skill.
    Phase 2 — All months randomly (dry + moderate + monsoon)
              Agent generalises to monsoon and varied conditions.

Action space: Box(0, 1, shape=(1,)) — agent outputs a fraction in [0, 1]
              mapped to [0, zone.max_litres_per_event] litres.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.base import CropProfile
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.dynamic_stage import DynamicStageTracker
from irrigation.rl.environment import IrrigationEnvironment
from irrigation.rl.health_tracker import PlantHealthTracker
from irrigation.rl.vitality import PlantVitalityTracker
from irrigation.sensors.simulation import (
    CombinedSimulatedSensor,
    SimulatedSoilMoistureSensor,
    SimulatedWeatherSensor,
)
from irrigation.zone_config import ZoneConfig


# ---------------------------------------------------------------------------
# Weather conditions per month group — based on Jaffna climate
# ---------------------------------------------------------------------------

# Phase 1 — Favorable: Feb, Mar, Apr (dry season)
# Hot, low humidity, minimal rain → clear reward signal for irrigation
_PHASE1_CONDITIONS = [
    {"base_temp": 32.0, "humidity": 65.0, "rainy": False},
    {"base_temp": 33.0, "humidity": 62.0, "rainy": False},
    {"base_temp": 31.0, "humidity": 67.0, "rainy": False},
]

# Phase 2 — All months: dry + moderate + monsoon
_PHASE2_CONDITIONS = [
    # Dry months (Feb–Apr)
    {"base_temp": 32.0, "humidity": 65.0, "rainy": False},
    {"base_temp": 33.0, "humidity": 62.0, "rainy": False},
    # Moderate months (May–Sep, Southwest monsoon)
    {"base_temp": 29.0, "humidity": 75.0, "rainy": False},
    {"base_temp": 28.0, "humidity": 78.0, "rainy": False},
    # Monsoon months (Oct–Jan, Northeast monsoon)
    {"base_temp": 27.0, "humidity": 85.0, "rainy": True},
    {"base_temp": 26.0, "humidity": 88.0, "rainy": True},
]


class IrrigationGymEnv(gymnasium.Env):
    """PPO-ready Gymnasium environment for irrigation control.

    Args:
        crop: Crop profile. Defaults to ChiliProfile.
        zone: Zone configuration. Defaults to ZoneConfig (3m² drip bed).
        step_hours: Simulated time per step in hours (default 1.0 = 1 hour).
        training_phase: Curriculum phase (1=favorable months, 2=all months).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        crop: CropProfile | None = None,
        zone: ZoneConfig | None = None,
        step_hours: float = 1.0,
        training_phase: int = 1,
    ) -> None:
        super().__init__()

        self.crop = crop or ChiliProfile()
        self.zone = zone or ZoneConfig()
        self.step_hours = step_hours
        self.training_phase = training_phase
        self._steps_per_day = int(24 / step_hours)
        self._max_steps = self.crop.growing_season_days * self._steps_per_day

        # Continuous action: fraction of max irrigation volume.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 9 normalized continuous observation values, all in [0, 1].
        # Index 7 = plant health_score (PlantHealthTracker)
        # Index 8 = plant vitality    (PlantVitalityTracker)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        self._soil_sensor = SimulatedSoilMoistureSensor(step_hours=step_hours)
        self._weather_sensor = SimulatedWeatherSensor(step_hours=step_hours)
        self._combined_sensor = CombinedSimulatedSensor(
            self._soil_sensor, self._weather_sensor
        )
        self._actuator = SimulatedActuator(
            soil_sensor=self._soil_sensor,
            moisture_per_litre=self.zone.moisture_per_litre,
        )
        self._env = IrrigationEnvironment(
            sensor=self._combined_sensor,
            actuator=self._actuator,
            crop=self.crop,
            planting_date=date.today(),
            zone=self.zone,
        )

        self._step = 0
        self._health_tracker   = PlantHealthTracker(self.crop)
        self._stage_tracker    = DynamicStageTracker(self.crop)
        self._vitality_tracker = PlantVitalityTracker(self.crop)

    def _pick_weather_conditions(self) -> dict:
        """Pick random weather conditions based on current training phase."""
        if self.training_phase == 1:
            return random.choice(_PHASE1_CONDITIONS)
        return random.choice(_PHASE2_CONDITIONS)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick weather conditions for this episode based on curriculum phase.
        conditions = self._pick_weather_conditions()
        self._weather_sensor.base_temp      = conditions["base_temp"]
        self._weather_sensor.base_humidity  = conditions["humidity"]
        self._weather_sensor.is_rainy_season = conditions["rainy"]

        initial_moisture = float(self.np_random.uniform(40.0, 80.0))
        self._soil_sensor.reset(initial_moisture_pct=initial_moisture, initial_hour=6.0)
        self._weather_sensor.reset(initial_hour=6.0)
        self._actuator.reset()

        self._step = 0
        self._env._sim_day = 0
        self._env._sim_stage = 0
        self._env._last_reading = None
        self._health_tracker.reset()
        self._stage_tracker.reset()
        self._vitality_tracker.reset()

        state = self._env.observe()
        obs = self._make_obs(state)
        return obs, {"phase": self.training_phase, "conditions": conditions}

    def _make_obs(self, state) -> np.ndarray:
        """Build (9,) observation: 7 env values + health_score + vitality."""
        base_obs = np.clip(
            state.to_observation(self.crop.growing_season_days), 0.0, 1.0
        )
        extras = np.array(
            [self._health_tracker.health_score,
             self._vitality_tracker.vitality],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, extras])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Sync simulated day and dynamic stage before the environment observes.
        self._env._sim_day   = self._step // self._steps_per_day
        self._env._sim_stage = self._stage_tracker.current_stage

        # Map agent output [0, 1] → [0, max_litres_per_event].
        water_fraction = float(np.clip(action[0], 0.0, 1.0))
        water_litres   = water_fraction * self.zone.max_litres_per_event

        next_state, reward, _ = self._env.step(water_litres)
        self._step += 1

        current_stage = self._stage_tracker.current_stage

        # Update dynamic stage tracker — may advance stage this step.
        stage_advanced = self._stage_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
        )

        # Update vitality — check plant death using current dynamic stage.
        self._vitality_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
            stage=current_stage,
        )

        # Update health tracker with dynamic stage (not calendar stage).
        self._health_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
            stage=current_stage,
            water_litres=water_litres,
        )

        # Apply death penalty and terminate if plant died.
        plant_dead = self._vitality_tracker.is_dead
        if plant_dead:
            reward += self._vitality_tracker.death_penalty

        obs = self._make_obs(next_state)
        terminated = self._step >= self._max_steps or plant_dead

        info = {
            "day":              next_state.current_day,
            "calendar_stage":   next_state.growth_stage,
            "dynamic_stage":    current_stage,
            "stage_name":       self._stage_tracker.stage_name,
            "days_in_stage":    self._stage_tracker.days_in_stage,
            "stage_advanced":   stage_advanced,
            "health_ratio":     self._stage_tracker.health_ratio,
            "moisture":         next_state.soil_moisture_pct,
            "water_litres":     water_litres,
            "health_score":     self._health_tracker.health_score,
            "stress_ratio":     self._health_tracker.stress_ratio,
            "vitality":         self._vitality_tracker.vitality,
            "phase":            self.training_phase,
        }

        # At episode end add full summaries from all trackers.
        if terminated:
            info.update(self._health_tracker.summary())
            info.update(self._stage_tracker.summary())
            info.update(self._vitality_tracker.death_info())

        return obs, float(reward), terminated, False, info
