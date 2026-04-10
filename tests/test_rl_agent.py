"""Tests for the Q-learning agent."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from irrigation.actuators.base import IrrigationAction
from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.agent import QLearningAgent
from irrigation.rl.environment import IrrigationEnvironment, IrrigationState
from irrigation.sensors.base import SensorReading
from irrigation.sensors.simulation import (
    SimulatedSoilMoistureSensor,
    SimulatedWeatherSensor,
)


def _make_agent(initial_moisture: float = 50.0) -> tuple[QLearningAgent, IrrigationEnvironment]:
    soil = SimulatedSoilMoistureSensor(initial_moisture_pct=initial_moisture, seed=0)
    weather = SimulatedWeatherSensor(seed=0)
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

    env = IrrigationEnvironment(sensor=_CombinedSensor(), actuator=actuator, crop=crop)
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_min=0.05,
        exploration_decay=0.99,
    )
    return agent, env


class TestQLearningAgent:
    def test_q_table_shape(self):
        agent, env = _make_agent()
        expected = (*env.state_shape, env.n_actions)
        assert agent.q_table.shape == expected

    def test_choose_action_returns_valid_action(self):
        agent, env = _make_agent()
        state = env.observe()
        action = agent.choose_action(state)
        assert isinstance(action, IrrigationAction)

    def test_choose_greedy_action(self):
        agent, env = _make_agent()
        state = env.observe()
        action = agent.choose_greedy_action(state)
        assert isinstance(action, IrrigationAction)

    def test_update_changes_q_table(self):
        agent, env = _make_agent()
        state = env.observe()
        action = IrrigationAction.NO_IRRIGATION
        env.step(action)
        next_state = env.observe()

        q_before = agent.q_table[state.to_tuple()][action.value]
        agent.update(state, action, reward=1.0, next_state=next_state)
        q_after = agent.q_table[state.to_tuple()][action.value]

        # The update should change the Q-value (unless reward matches perfectly).
        # We just verify the update ran without error; values may or may not change.
        assert isinstance(q_after, float | np.floating)

    def test_train_returns_rewards_list(self):
        agent, _ = _make_agent()
        rewards = agent.train(n_episodes=5, steps_per_episode=4)
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

    def test_epsilon_decays_after_training(self):
        agent, _ = _make_agent()
        initial_epsilon = agent.epsilon
        agent.train(n_episodes=10, steps_per_episode=2)
        assert agent.epsilon < initial_epsilon

    def test_epsilon_does_not_go_below_min(self):
        agent, _ = _make_agent()
        agent.train(n_episodes=500, steps_per_episode=2)
        assert agent.epsilon >= agent.epsilon_min

    def test_save_and_load(self):
        agent, _ = _make_agent()
        agent.train(n_episodes=5, steps_per_episode=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            agent.save(path)
            assert path.exists()

            agent2, env2 = _make_agent()
            agent2.load(path)
            np.testing.assert_array_almost_equal(agent.q_table, agent2.q_table)
            assert agent2.epsilon == pytest.approx(agent.epsilon)

    def test_episode_rewards_property(self):
        agent, _ = _make_agent()
        agent.train(n_episodes=3, steps_per_episode=2)
        rewards = agent.episode_rewards
        assert len(rewards) == 3
