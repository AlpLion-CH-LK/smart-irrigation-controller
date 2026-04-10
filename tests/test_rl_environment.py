"""Tests for the RL environment and reward function."""

from __future__ import annotations

import pytest

from irrigation.actuators.base import IrrigationAction, IrrigationCommand
from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.environment import IrrigationEnvironment, IrrigationState
from irrigation.rl.reward import RewardFunction
from irrigation.sensors.simulation import (
    SimulatedSoilMoistureSensor,
    SimulatedWeatherSensor,
)
from irrigation.sensors.base import SensorReading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(initial_moisture: float = 50.0) -> IrrigationEnvironment:
    soil = SimulatedSoilMoistureSensor(initial_moisture_pct=initial_moisture, seed=42)
    weather = SimulatedWeatherSensor(seed=42)
    actuator = SimulatedActuator(soil_sensor=soil)
    crop = ChiliProfile()

    class _CombinedSensor:
        def read(self):
            s = soil.read()
            w = weather.read()
            return SensorReading(
                soil_moisture_pct=s.soil_moisture_pct,
                temperature_celsius=w.temperature_celsius,
                humidity_pct=w.humidity_pct,
                is_raining=w.is_raining,
                rainfall_mm=w.rainfall_mm,
            )

    return IrrigationEnvironment(
        sensor=_CombinedSensor(),
        actuator=actuator,
        crop=crop,
    )


# ---------------------------------------------------------------------------
# IrrigationState tests
# ---------------------------------------------------------------------------

class TestIrrigationState:
    def test_to_tuple_length(self):
        state = IrrigationState(
            soil_moisture_bin=3,
            temperature_bin=2,
            time_bin=5,
            is_raining=False,
        )
        t = state.to_tuple()
        assert len(t) == 4

    def test_to_tuple_values(self):
        state = IrrigationState(
            soil_moisture_bin=3,
            temperature_bin=2,
            time_bin=5,
            is_raining=True,
        )
        assert state.to_tuple() == (3, 2, 5, 1)


# ---------------------------------------------------------------------------
# IrrigationEnvironment tests
# ---------------------------------------------------------------------------

class TestIrrigationEnvironment:
    def test_observe_returns_state(self):
        env = _make_env()
        state = env.observe()
        assert isinstance(state, IrrigationState)

    def test_state_bins_in_range(self):
        env = _make_env()
        state = env.observe()
        assert 0 <= state.soil_moisture_bin < env.n_soil_bins
        assert 0 <= state.temperature_bin < env.n_temp_bins
        assert 0 <= state.time_bin < env.n_time_bins

    def test_step_returns_tuple(self):
        env = _make_env()
        env.observe()  # Prime last_reading
        next_state, reward, done = env.step(IrrigationAction.NO_IRRIGATION)
        assert isinstance(next_state, IrrigationState)
        assert isinstance(reward, float)
        assert done is False

    def test_n_actions(self):
        env = _make_env()
        assert env.n_actions == len(IrrigationAction)

    def test_state_shape(self):
        env = _make_env()
        shape = env.state_shape
        assert len(shape) == 4
        assert shape[0] == env.n_soil_bins
        assert shape[1] == env.n_temp_bins
        assert shape[2] == env.n_time_bins
        assert shape[3] == 2  # rain: True / False


# ---------------------------------------------------------------------------
# RewardFunction tests
# ---------------------------------------------------------------------------

class TestRewardFunction:
    def setup_method(self):
        self.crop = ChiliProfile()
        self.reward_fn = RewardFunction(self.crop)

    def test_optimal_moisture_no_action_positive_reward(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        cmd = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        reward = self.reward_fn.compute(soil_moisture_pct=mid, command=cmd)
        assert reward > 0.0

    def test_drought_stress_penalises_reward(self):
        t = self.crop.moisture_thresholds
        # Reward at dry soil (stress) vs reward at optimal
        cmd = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        reward_stressed = self.reward_fn.compute(
            soil_moisture_pct=t.wilting_point, command=cmd
        )
        reward_optimal = self.reward_fn.compute(
            soil_moisture_pct=t.optimal_min, command=cmd
        )
        assert reward_stressed < reward_optimal

    def test_irrigation_reduces_reward(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        cmd_none = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        cmd_short = IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT)
        reward_none = self.reward_fn.compute(soil_moisture_pct=mid, command=cmd_none)
        reward_irrigate = self.reward_fn.compute(soil_moisture_pct=mid, command=cmd_short)
        # Water cost should make irrigation-with-optimal-soil less rewarding.
        assert reward_none > reward_irrigate

    def test_rain_bonus_for_skipping_irrigation(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        cmd_none = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        reward_no_rain = self.reward_fn.compute(
            soil_moisture_pct=mid, command=cmd_none, is_raining=False
        )
        reward_rain = self.reward_fn.compute(
            soil_moisture_pct=mid, command=cmd_none, is_raining=True
        )
        assert reward_rain > reward_no_rain

    def test_overwatering_penalised(self):
        t = self.crop.moisture_thresholds
        cmd = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        reward_optimal = self.reward_fn.compute(
            soil_moisture_pct=t.optimal_max, command=cmd
        )
        reward_overwater = self.reward_fn.compute(
            soil_moisture_pct=t.field_capacity + 5.0, command=cmd
        )
        assert reward_optimal > reward_overwater
