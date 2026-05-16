"""RL environment for the irrigation controller.

The environment wraps the physical (or simulated) sensors and actuators and
exposes a Gym-like interface that the PPO agent interacts with.

Observation space (7 continuous values, normalized):
    - Soil moisture      soil_moisture_pct / 100
    - Temperature        temperature_celsius / 40
    - Humidity           humidity_pct / 100
    - Time of day        hour / 24
    - Is raining         0 or 1
    - Growth stage       stage / 4
    - Current day        current_day / growing_season_days

Action space:
    Four discrete actions defined in IrrigationAction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from irrigation.actuators.base import (
    ActuatorInterface,
    IrrigationAction,
    IrrigationCommand,
)
from irrigation.crops.base import CropProfile
from irrigation.rl.reward import RewardFunction
from irrigation.sensors.base import SensorInterface, SensorReading

logger = logging.getLogger(__name__)



@dataclass
class IrrigationState:
    """The continuous state observed by the RL agent.

    Attributes:
        soil_moisture_pct: Raw soil moisture percentage (0-100).
        temperature_celsius: Air temperature in degrees Celsius.
        humidity_pct: Relative humidity percentage (0-100).
        hour: Hour of day with fractional minutes (0-24).
        is_raining: Whether rain is currently detected.
        growth_stage: Current growth stage index (0-4).
        current_day: Days since planting (0 to growing_season_days).
    """

    soil_moisture_pct: float
    temperature_celsius: float
    humidity_pct: float
    hour: float
    is_raining: bool
    growth_stage: int
    current_day: int

    def to_observation(self, growing_season_days: int = 150) -> np.ndarray:
        """Return a normalized float32 array for the PPO policy network."""
        obs = np.array(
            [
                self.soil_moisture_pct / 100.0,
                self.temperature_celsius / 40.0,
                self.humidity_pct / 100.0,
                self.hour / 24.0,
                float(self.is_raining),
                self.growth_stage / 4.0,
                self.current_day / growing_season_days,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)


class IrrigationEnvironment:
    """Gym-inspired environment wrapping sensors, actuators and reward logic.

    Args:
        sensor: Sensor implementation (real or simulated).
        actuator: Actuator implementation (real or simulated).
        crop: Crop profile providing stage thresholds and season length.
        planting_date: The date the crop was planted. Defaults to today.
    """

    def __init__(
        self,
        sensor: SensorInterface,
        actuator: ActuatorInterface,
        crop: CropProfile,
        planting_date: date | None = None,
    ) -> None:
        self.sensor = sensor
        self.actuator = actuator
        self.crop = crop
        self.planting_date = planting_date or date.today()
        self.reward_fn = RewardFunction(crop)
        self._last_reading: SensorReading | None = None
        # Set by IrrigationGymEnv during training to override wall-clock day.
        self._sim_day: int | None = None

    def _current_day(self) -> int:
        """Days since planting, clamped to the season length."""
        if self._sim_day is not None:
            return self._sim_day
        days = (date.today() - self.planting_date).days
        return max(0, min(days, self.crop.growing_season_days))

    def _growth_stage(self, current_day: int) -> int:
        """Map a day count to a growth stage index (0-4)."""
        return self.crop.stage_for_day(current_day)

    def observe(self) -> IrrigationState:
        """Take a fresh sensor reading and return the current state."""
        reading = self.sensor.read()
        self._last_reading = reading
        current_day = self._current_day()
        return IrrigationState(
            soil_moisture_pct=reading.soil_moisture_pct,
            temperature_celsius=reading.temperature_celsius,
            humidity_pct=reading.humidity_pct,
            hour=reading.timestamp.hour + reading.timestamp.minute / 60.0,
            is_raining=reading.is_raining,
            growth_stage=self._growth_stage(current_day),
            current_day=current_day,
        )

    def step(
        self, action: IrrigationAction
    ) -> tuple[IrrigationState, float, bool]:
        """Execute an action and return (next_state, reward, done).

        Args:
            action: The action chosen by the agent.

        Returns:
            A three-tuple of (next_state, reward, done). done is always
            False for this continuous-control environment.
        """
        command = IrrigationCommand(action=action)
        if self._last_reading is None:
            self.observe()
        assert self._last_reading is not None
        is_raining = self._last_reading.is_raining

        self.actuator.execute(command)

        next_state = self.observe()

        reward = self.reward_fn.compute(
            soil_moisture_pct=next_state.soil_moisture_pct,
            command=command,
            is_raining=is_raining,
            stage=next_state.growth_stage,
        )

        logger.debug(
            "action=%s moisture=%.1f%% stage=%d reward=%.4f",
            action.name,
            next_state.soil_moisture_pct,
            next_state.growth_stage,
            reward,
        )
        return next_state, reward, False

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions."""
        return len(IrrigationAction)

    @property
    def observation_size(self) -> int:
        """Number of values in the observation vector fed to the PPO network."""
        return 7
