"""RL environment for the irrigation controller.

The environment wraps the physical (or simulated) sensors and actuators and
exposes a Gym-like interface that the :class:`~irrigation.rl.agent.QLearningAgent`
interacts with.

State space (discrete bins):
    - Soil moisture level  (n_soil_bins bins, default 10)
    - Temperature          (n_temp_bins bins, default 5)
    - Time of day          (n_time_bins slots, default 8 × 3-hour windows)
    - Is raining           (2: False / True)

Action space:
    Four discrete actions defined in :class:`~irrigation.actuators.base.IrrigationAction`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

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

# Temperature bins: < 15, 15-20, 20-25, 25-30, ≥ 30 °C
_TEMP_BIN_EDGES = [15.0, 20.0, 25.0, 30.0]


@dataclass
class IrrigationState:
    """The discrete state observed by the RL agent.

    Attributes:
        soil_moisture_bin: Discretised soil moisture level.
        temperature_bin: Discretised temperature.
        time_bin: Time-of-day slot (3-hour windows: 0=00-03, …, 7=21-24).
        is_raining: Whether rain is currently detected.
    """

    soil_moisture_bin: int
    temperature_bin: int
    time_bin: int
    is_raining: bool

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to a hashable tuple for use as a Q-table key."""
        return (
            self.soil_moisture_bin,
            self.temperature_bin,
            self.time_bin,
            int(self.is_raining),
        )


class IrrigationEnvironment:
    """Gym-inspired environment wrapping sensors, actuators and reward logic.

    Args:
        sensor: Sensor implementation (real or simulated).
        actuator: Actuator implementation (real or simulated).
        crop: Crop profile for reward shaping.
        n_soil_bins: Number of discrete soil-moisture bins.
        n_temp_bins: Number of discrete temperature bins.
        n_time_bins: Number of time-of-day bins (must be a divisor of 24).
    """

    def __init__(
        self,
        sensor: SensorInterface,
        actuator: ActuatorInterface,
        crop: CropProfile,
        n_soil_bins: int = 10,
        n_temp_bins: int = 5,
        n_time_bins: int = 8,
    ) -> None:
        self.sensor = sensor
        self.actuator = actuator
        self.crop = crop
        self.n_soil_bins = n_soil_bins
        self.n_temp_bins = n_temp_bins
        self.n_time_bins = n_time_bins
        self.reward_fn = RewardFunction(crop)
        self._last_reading: SensorReading | None = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _discretise_moisture(self, moisture_pct: float) -> int:
        """Map a moisture percentage to a bin index."""
        bin_size = 100.0 / self.n_soil_bins
        idx = int(moisture_pct / bin_size)
        return min(idx, self.n_soil_bins - 1)

    def _discretise_temperature(self, temp_celsius: float) -> int:
        """Map a temperature to a bin index using fixed edges."""
        for i, edge in enumerate(_TEMP_BIN_EDGES):
            if temp_celsius < edge:
                return i
        return len(_TEMP_BIN_EDGES)

    def _discretise_time(self, dt: datetime) -> int:
        """Map a datetime to a time-of-day bin."""
        hours_per_bin = 24 / self.n_time_bins
        return int(dt.hour / hours_per_bin) % self.n_time_bins

    def observe(self) -> IrrigationState:
        """Take a fresh sensor reading and return the discretised state."""
        reading = self.sensor.read()
        self._last_reading = reading
        return IrrigationState(
            soil_moisture_bin=self._discretise_moisture(reading.soil_moisture_pct),
            temperature_bin=self._discretise_temperature(reading.temperature_celsius),
            time_bin=self._discretise_time(reading.timestamp),
            is_raining=reading.is_raining,
        )

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def step(
        self, action: IrrigationAction
    ) -> tuple[IrrigationState, float, bool]:
        """Execute an action and return (next_state, reward, done).

        Args:
            action: The action chosen by the agent.

        Returns:
            A three-tuple of (next_state, reward, done).  ``done`` is always
            ``False`` for this continuous-control environment; episode
            termination is handled externally.
        """
        command = IrrigationCommand(action=action)
        # Ensure we have a fresh reading to determine rain state before irrigating.
        if self._last_reading is None:
            self.observe()
        is_raining = self._last_reading.is_raining if self._last_reading else False

        # Execute the irrigation command.
        self.actuator.execute(command)

        # Observe the resulting state.
        next_state = self.observe()

        # Re-read the continuous moisture value for reward computation.
        moisture_after = (
            self._last_reading.soil_moisture_pct if self._last_reading else 50.0
        )

        reward = self.reward_fn.compute(
            soil_moisture_pct=moisture_after,
            command=command,
            is_raining=is_raining,
        )

        logger.debug(
            "action=%s moisture=%.1f%% reward=%.4f",
            action.name,
            moisture_after,
            reward,
        )
        return next_state, reward, False

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions."""
        return len(IrrigationAction)

    @property
    def state_shape(self) -> tuple[int, int, int, int]:
        """Shape of the Q-table (soil_bins, temp_bins, time_bins, rain_states)."""
        return (self.n_soil_bins, self.n_temp_bins, self.n_time_bins, 2)
