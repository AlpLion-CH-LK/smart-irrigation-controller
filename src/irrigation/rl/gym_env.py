"""Gymnasium wrapper around IrrigationEnvironment for PPO training.

One episode = one full growing season. The simulated clock advances by
step_hours on every step, independent of real wall-clock time.

Action space: Box(0, 1, shape=(1,)) — agent outputs a fraction in [0, 1]
              mapped to [0, zone.max_litres_per_event] litres.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.base import CropProfile
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.environment import IrrigationEnvironment
from irrigation.sensors.simulation import (
    CombinedSimulatedSensor,
    SimulatedSoilMoistureSensor,
    SimulatedWeatherSensor,
)
from irrigation.zone_config import ZoneConfig


class IrrigationGymEnv(gymnasium.Env):
    """PPO-ready Gymnasium environment for irrigation control.

    Args:
        crop: Crop profile. Defaults to ChiliProfile.
        zone: Zone configuration. Defaults to ZoneConfig (3m² drip bed).
        step_hours: Simulated time per step in hours (default 0.5 = 30 min).
        base_temp_celsius: Mean daily temperature for weather simulation.
        is_rainy_season: Whether to simulate more frequent rainfall.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        crop: CropProfile | None = None,
        zone: ZoneConfig | None = None,
        step_hours: float = 0.5,
        base_temp_celsius: float = 28.0,
        is_rainy_season: bool = False,
    ) -> None:
        super().__init__()

        self.crop = crop or ChiliProfile()
        self.zone = zone or ZoneConfig()
        self.step_hours = step_hours
        self._steps_per_day = int(24 / step_hours)
        self._max_steps = self.crop.growing_season_days * self._steps_per_day

        # Continuous action: fraction of max irrigation volume.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 7 normalized continuous observation values, all in [0, 1].
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self._soil_sensor = SimulatedSoilMoistureSensor(step_hours=step_hours)
        self._weather_sensor = SimulatedWeatherSensor(
            base_temp_celsius=base_temp_celsius,
            is_rainy_season=is_rainy_season,
            step_hours=step_hours,
        )
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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        initial_moisture = float(self.np_random.uniform(40.0, 80.0))
        self._soil_sensor.reset(initial_moisture_pct=initial_moisture, initial_hour=6.0)
        self._weather_sensor.reset(initial_hour=6.0)
        self._actuator.reset()

        self._step = 0
        self._env._sim_day = 0
        self._env._last_reading = None

        state = self._env.observe()
        obs = np.clip(
            state.to_observation(self.crop.growing_season_days), 0.0, 1.0
        )
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Sync simulated day before the environment observes.
        self._env._sim_day = self._step // self._steps_per_day

        # Map agent output [0, 1] → [0, max_litres_per_event].
        water_fraction = float(np.clip(action[0], 0.0, 1.0))
        water_litres = water_fraction * self.zone.max_litres_per_event

        next_state, reward, _ = self._env.step(water_litres)
        self._step += 1

        obs = np.clip(
            next_state.to_observation(self.crop.growing_season_days), 0.0, 1.0
        )
        terminated = self._step >= self._max_steps
        info = {
            "day": next_state.current_day,
            "stage": next_state.growth_stage,
            "moisture": next_state.soil_moisture_pct,
            "water_litres": water_litres,
        }
        return obs, float(reward), terminated, False, info
