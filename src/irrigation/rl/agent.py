"""Q-learning agent for irrigation control.

The agent learns a policy that maps discrete environment states to irrigation
actions.  It uses tabular Q-learning with ε-greedy exploration and supports
persistence (save / load) so that a trained policy can survive system reboots.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from irrigation.actuators.base import IrrigationAction
from irrigation.rl.environment import IrrigationEnvironment, IrrigationState

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Tabular Q-learning agent for irrigation control.

    Args:
        env: The :class:`~irrigation.rl.environment.IrrigationEnvironment` to
            interact with.
        learning_rate: Q-update step size (α).
        discount_factor: Future reward discount (γ).
        exploration_rate: Initial ε for ε-greedy exploration.
        exploration_min: Minimum ε after decay.
        exploration_decay: Multiplicative decay applied to ε after each episode.
    """

    def __init__(
        self,
        env: IrrigationEnvironment,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_min: float = 0.05,
        exploration_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay

        # Initialise Q-table to small random values to break symmetry.
        self.q_table: np.ndarray = np.zeros(
            (*env.state_shape, env.n_actions), dtype=np.float32
        )
        self.q_table += np.random.uniform(0, 0.01, self.q_table.shape).astype(np.float32)

        self._episode_rewards: list[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: IrrigationState) -> IrrigationAction:
        """Choose an action using ε-greedy policy.

        Args:
            state: Current environment state.

        Returns:
            The chosen :class:`~irrigation.actuators.base.IrrigationAction`.
        """
        if np.random.random() < self.epsilon:
            return IrrigationAction(np.random.randint(self.env.n_actions))

        s = state.to_tuple()
        action_idx = int(np.argmax(self.q_table[s]))
        return IrrigationAction(action_idx)

    def choose_greedy_action(self, state: IrrigationState) -> IrrigationAction:
        """Choose the best known action without any exploration.

        Args:
            state: Current environment state.

        Returns:
            The greedy :class:`~irrigation.actuators.base.IrrigationAction`.
        """
        s = state.to_tuple()
        action_idx = int(np.argmax(self.q_table[s]))
        return IrrigationAction(action_idx)

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------

    def update(
        self,
        state: IrrigationState,
        action: IrrigationAction,
        reward: float,
        next_state: IrrigationState,
    ) -> None:
        """Apply the Q-learning update rule.

        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

        Args:
            state: State before taking the action.
            action: Action that was taken.
            reward: Reward received.
            next_state: State after taking the action.
        """
        s = state.to_tuple()
        s_next = next_state.to_tuple()
        a = action.value

        current_q = float(self.q_table[s][a])
        best_next_q = float(np.max(self.q_table[s_next]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q

        self.q_table[s][a] = current_q + self.alpha * td_error

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 1000, steps_per_episode: int = 48) -> list[float]:
        """Train the agent for a number of episodes.

        Each episode represents one day of control decisions at a
        30-minute resolution (48 steps by default).

        Args:
            n_episodes: Number of training episodes.
            steps_per_episode: Number of steps (decisions) per episode.

        Returns:
            A list of total rewards per episode.
        """
        episode_rewards: list[float] = []

        for episode in range(n_episodes):
            state = self.env.observe()
            total_reward = 0.0

            for _ in range(steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            # Decay exploration rate.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg = np.mean(episode_rewards[-100:])
                logger.info(
                    "Episode %d/%d – avg reward (last 100): %.3f, ε=%.4f",
                    episode + 1,
                    n_episodes,
                    avg,
                    self.epsilon,
                )

        self._episode_rewards = episode_rewards
        return episode_rewards

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the Q-table and hyper-parameters to disk.

        Args:
            path: File path (will be created including parent directories).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Q-table saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved Q-table from disk.

        Args:
            path: File path to load from.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        with open(path, "rb") as f:
            payload: dict = pickle.load(f)  # noqa: S301 – only loads our own files
        self.q_table = payload["q_table"]
        self.epsilon = float(payload["epsilon"])
        self.alpha = float(payload["alpha"])
        self.gamma = float(payload["gamma"])
        logger.info("Q-table loaded from %s", path)

    @property
    def episode_rewards(self) -> list[float]:
        """Reward history from the last training run."""
        return list(self._episode_rewards)
